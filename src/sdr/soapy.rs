//! SoapySDR backend: any device a SoapySDR module drives.
//!
//! SoapySDR's stream carries no per-sample frequency tag, so a hop
//! cannot be told apart from the samples still buffered from the last
//! channel. Each retune therefore stops the stream, tunes, and starts it
//! again, which empties the driver's buffers; the first `settle` of
//! samples after that is discarded while the synthesiser locks. That
//! costs a few milliseconds a hop, and in exchange a transmitter is
//! never reported at the frequency of the hop after the one it was
//! heard on.

use super::{
    IqBuffer, IqPacket, SdrHandle, SdrSource, SourceConfig, Stopper, deliver, queue_depth,
};
use anyhow::Context;
use num_complex::Complex32;
use soapysdr::{Device, Direction, ErrorCode, RxStream};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, SyncSender};
use std::thread;
use std::time::{Duration, Instant};
use tracing::{error, warn};

/// How long a single stream read may wait for samples.
const READ_TIMEOUT: Duration = Duration::from_millis(100);

/// Consecutive read timeouts before the device is taken to be gone.
/// Fifty of [`READ_TIMEOUT`] is five seconds of silence.
const TIMEOUT_BAILOUT: u32 = 50;

/// Consecutive stream errors (other than overflow and timeout) before
/// the capture gives up.
const ERROR_BAILOUT: u32 = 5;

/// A capture on an open SoapySDR device.
///
/// The device is opened and configured (gain, antenna, settings) once
/// by the caller and shared by every capture it starts; each capture
/// sets the sample rate its config asks for and opens its own stream.
pub struct SoapySource {
    pub device: Device,
    pub channel: usize,
    /// Samples per packet.
    pub block_size: usize,
    /// Discarded after every (re)start of the stream.
    pub settle: Duration,
}

/// Set the RX sample rate and return the rate the device actually runs,
/// which can differ from the request on a device with a rate grid.
pub fn apply_sample_rate(device: &Device, channel: usize, rate_hz: f64) -> anyhow::Result<f64> {
    device
        .set_sample_rate(Direction::Rx, channel, rate_hz)
        .with_context(|| format!("setting sample rate {:.3} MSPS", rate_hz / 1e6))?;
    Ok(device
        .sample_rate(Direction::Rx, channel)
        .unwrap_or(rate_hz))
}

/// The fastest RX rate the device reports, if it reports any.
pub fn max_sample_rate(device: &Device, channel: usize) -> Option<f64> {
    device
        .get_sample_rate_range(Direction::Rx, channel)
        .ok()?
        .iter()
        .map(|r| r.maximum)
        .max_by(f64::total_cmp)
}

impl SdrSource for SoapySource {
    fn start(self: Box<Self>, config: SourceConfig) -> anyhow::Result<SdrHandle> {
        let SoapySource {
            device,
            channel,
            block_size,
            settle,
        } = *self;
        anyhow::ensure!(
            !config.channels_hz.is_empty(),
            "a capture needs at least one channel"
        );
        let block_size = block_size.max(1024);
        let sample_rate = apply_sample_rate(&device, channel, config.sample_rate_hz)?;
        let stream = device
            .rx_stream::<Complex32>(&[channel])
            .context("opening the RX stream")?;

        let depth = queue_depth(block_size);
        let (tx, receiver) = std::sync::mpsc::sync_channel::<IqPacket>(depth);
        let (pool_tx, pool_rx) = std::sync::mpsc::sync_channel::<Vec<Complex32>>(depth);
        let stop = Arc::new(AtomicBool::new(false));
        let capture = Capture {
            device,
            channel,
            stream,
            sample_rate,
            block_size,
            settle_samples: (sample_rate * settle.as_secs_f64()).round() as usize,
            scratch: vec![Complex32::default(); block_size.min(65_536)],
            channels_hz: config.channels_hz,
            dwell: config.dwell,
            tx,
            pool_tx,
            pool_rx,
            stop: stop.clone(),
        };
        let thread = thread::Builder::new()
            .name("soapy-capture".into())
            .spawn(move || {
                if let Err(e) = capture.run() {
                    error!("SoapySDR capture ended: {e:#}");
                }
            })?;
        Ok(SdrHandle::new(receiver, Stopper::new(stop, thread)))
    }
}

struct Capture {
    device: Device,
    channel: usize,
    stream: RxStream<Complex32>,
    sample_rate: f64,
    block_size: usize,
    settle_samples: usize,
    /// Where the settle's samples are read to and thrown away.
    scratch: Vec<Complex32>,
    channels_hz: Vec<f64>,
    dwell: Duration,
    tx: SyncSender<IqPacket>,
    pool_tx: SyncSender<Vec<Complex32>>,
    pool_rx: Receiver<Vec<Complex32>>,
    stop: Arc<AtomicBool>,
}

/// What filling one block came to.
enum Fill {
    Full,
    /// The stream lost samples part-way; the block was discarded.
    Overflowed,
    Stopped,
}

impl Capture {
    fn run(mut self) -> anyhow::Result<()> {
        let hopping = self.channels_hz.len() > 1;
        let mut next = 0usize;
        let mut overrun = false;
        let mut errors = 0u32;
        let mut timeouts = 0u32;

        while !self.stopped() {
            let commanded = self.channels_hz[next];
            next = (next + 1) % self.channels_hz.len();
            let tuned = self.retune(commanded)?;
            let deadline = Instant::now() + self.dwell;

            loop {
                if self.stopped() {
                    break;
                }
                if hopping && Instant::now() >= deadline {
                    break;
                }
                let mut block = self
                    .pool_rx
                    .try_recv()
                    .unwrap_or_else(|_| Vec::with_capacity(self.block_size));
                match self.fill(&mut block, &mut errors, &mut timeouts)? {
                    Fill::Stopped => break,
                    Fill::Overflowed => {
                        overrun = true;
                        let _ = self.pool_tx.try_send(block);
                        continue;
                    }
                    Fill::Full => {}
                }
                let pkt = IqPacket {
                    samples: IqBuffer::new_pooled(block, self.pool_tx.clone()),
                    center_frequency_hz: tuned,
                    sample_rate_hz: self.sample_rate as f32,
                    overrun: std::mem::take(&mut overrun),
                };
                if !deliver(&self.tx, pkt, &self.stop) {
                    self.halt();
                    return Ok(());
                }
            }
            if !hopping {
                break;
            }
        }
        self.halt();
        Ok(())
    }

    fn stopped(&self) -> bool {
        self.stop.load(Ordering::SeqCst)
    }

    fn halt(&mut self) {
        if self.stream.active() {
            let _ = self.stream.deactivate(None);
        }
    }

    /// Stop the stream, tune, restart it and discard the settle. Returns
    /// the frequency the device reports it landed on.
    fn retune(&mut self, commanded: f64) -> anyhow::Result<f64> {
        self.halt();
        self.device
            .set_frequency(Direction::Rx, self.channel, commanded, ())
            .with_context(|| format!("tuning to {:.3} MHz", commanded / 1e6))?;
        self.stream
            .activate(None)
            .context("starting the RX stream")?;
        let tuned = self
            .device
            .frequency(Direction::Rx, self.channel)
            .ok()
            .filter(|f| f.is_finite() && *f > 0.0)
            .unwrap_or(commanded);

        let mut left = self.settle_samples;
        while left > 0 && !self.stopped() {
            let n = left.min(self.scratch.len());
            match self
                .stream
                .read(&mut [&mut self.scratch[..n]], micros(READ_TIMEOUT))
            {
                Ok(0) => break,
                Ok(got) => left = left.saturating_sub(got),
                Err(e) if e.code == ErrorCode::Overflow => {}
                Err(e) if e.code == ErrorCode::Timeout => break,
                Err(e) => return Err(e).context("reading the RX stream"),
            }
        }
        Ok(tuned)
    }

    /// Read until `block` holds `block_size` contiguous samples.
    fn fill(
        &mut self,
        block: &mut Vec<Complex32>,
        errors: &mut u32,
        timeouts: &mut u32,
    ) -> anyhow::Result<Fill> {
        block.clear();
        block.resize(self.block_size, Complex32::default());
        let mut filled = 0;
        while filled < self.block_size {
            if self.stopped() {
                return Ok(Fill::Stopped);
            }
            let read = self
                .stream
                .read(&mut [&mut block[filled..]], micros(READ_TIMEOUT));
            // A driver may return nothing without calling it a timeout;
            // either way no samples came, and it counts toward the
            // silence bailout rather than looping on the spot.
            let nothing = match &read {
                Ok(0) => true,
                Err(e) => e.code == ErrorCode::Timeout,
                Ok(_) => false,
            };
            if nothing {
                *timeouts += 1;
                anyhow::ensure!(
                    *timeouts < TIMEOUT_BAILOUT,
                    "no samples for {:?}; the device stopped streaming",
                    READ_TIMEOUT * TIMEOUT_BAILOUT
                );
                continue;
            }
            match read {
                Ok(n) => {
                    filled += n;
                    *errors = 0;
                    *timeouts = 0;
                }
                Err(e) if e.code == ErrorCode::Overflow => return Ok(Fill::Overflowed),
                Err(e) => {
                    *errors += 1;
                    anyhow::ensure!(
                        *errors < ERROR_BAILOUT,
                        "{ERROR_BAILOUT} consecutive stream errors, the last: {e}"
                    );
                    warn!("SoapySDR stream error: {e}");
                    return Ok(Fill::Overflowed);
                }
            }
        }
        Ok(Fill::Full)
    }
}

fn micros(d: Duration) -> i64 {
    d.as_micros() as i64
}
