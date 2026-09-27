//! Aaronia Spectran V6 backend, over `sdr-aaronia-rs`'s unified source.
//!
//! The unified source is async; a capture runs it on its own thread
//! with its own tokio runtime and passes packets to the viewer over the
//! same channel every backend uses.
//!
//! Unlike SoapySDR, the RTSA stream stamps every packet with the
//! frequency it was captured at, so a hop does not have to restart
//! anything: packets still carrying the previous centre are recognised
//! and dropped, and the post-retune drain only saves reading them.

use super::{IqBuffer, IqPacket, SdrHandle, SdrSource, SourceConfig, Stopper};
use anyhow::Context;
use num_complex::Complex32;
use sdr_aaronia_rs::http_streaming::StreamFormat;
use sdr_aaronia_rs::{SourceType, SpectranSource, SpectranSourceBuilder};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, SyncSender, TrySendError};
use std::thread;
use std::time::{Duration, Instant};
use tracing::{error, info, warn};

/// How far a packet's reported capture frequency may sit from the
/// commanded channel and still count as that channel.
///
/// The device reports where it is actually tuned, which is not exactly
/// what was asked for: measured on a V6 ECO its reference is a steady
/// −0.79 ppm off, −873 Hz at 1.1 GHz rising to −4.71 kHz at 5.93 GHz.
/// 100 kHz clears that by 20x at the top of the range and stays far
/// below the megahertz between any two hops, which is all it has to
/// separate.
const STALE_TOLERANCE_HZ: f64 = 100_000.0;

/// How long to drain the stream after a retune before reading for real.
///
/// An optimisation, not the correctness mechanism: the capture-frequency
/// check rejects stale packets exactly however long the device takes.
/// Measured on a V6 ECO over RTSA HTTP at 61.44 MSPS, a retune is
/// carried by the signal within 24 ms median and 39 ms worst case over
/// 24 hops; 20 ms is below that on purpose, because the drain is pure
/// added time on every hop and a longer one measured slower end to end.
const RETUNE_SETTLE: Duration = Duration::from_millis(20);

/// Empty reads on one hop before moving on to the next.
const EMPTY_READ_BAILOUT: u32 = 16;

/// Consecutive read or retune failures before the source is taken to
/// be dead and the capture ends, rather than spinning on it.
const ERROR_BAILOUT: u32 = 5;

/// How long a held (non-hopping) read waits before checking the stop
/// flag again.
const HOLD_READ_BUDGET: Duration = Duration::from_millis(50);

/// How long the capture may go without a packet it can deliver before
/// it ends. An RTSA server whose analyzer is stopped still answers HTTP
/// and opens `/stream`, then sends no bytes; without this the viewer
/// waits for its first packet forever. Packets that all report the
/// wrong frequency count as silence too, for the same reason.
const SILENCE_BAILOUT: Duration = Duration::from_secs(5);

/// Which way to reach the Spectran.
#[derive(Debug, Clone)]
pub enum Transport {
    /// The HTTP server block of a running RTSA-Suite.
    Http { url: String, format: StreamFormat },
    /// Direct USB through the native AARTSAAPI SDK; `serial` picks one
    /// of several attached devices.
    Sdk { serial: Option<String> },
}

/// A capture from a Spectran V6.
pub struct AaroniaSource {
    pub transport: Transport,
    pub reference_level_dbm: f64,
    /// Samples requested per read.
    pub block_size: usize,
}

impl SdrSource for AaroniaSource {
    fn start(self: Box<Self>, config: SourceConfig) -> anyhow::Result<SdrHandle> {
        anyhow::ensure!(
            !config.channels_hz.is_empty(),
            "a capture needs at least one channel"
        );
        let block_size = self.block_size.max(1024);
        let depth = super::queue_depth(block_size);
        let (tx, receiver) = std::sync::mpsc::sync_channel::<IqPacket>(depth);
        let (pool_tx, pool_rx) = std::sync::mpsc::sync_channel::<Vec<Complex32>>(depth);
        let stop = Arc::new(AtomicBool::new(false));
        let pump = Pump {
            block_size,
            channels_hz: config.channels_hz,
            dwell: config.dwell,
            tx,
            pool_tx,
            pool_rx,
            stop: stop.clone(),
        };
        let transport = self.transport;
        let reference_level_dbm = self.reference_level_dbm;
        let sample_rate_hz = config.sample_rate_hz;

        let thread = thread::Builder::new()
            .name("aaronia-capture".into())
            .spawn(move || {
                let runtime = match tokio::runtime::Builder::new_multi_thread()
                    .enable_all()
                    .thread_name("aaronia-pump")
                    .build()
                {
                    Ok(rt) => rt,
                    Err(e) => {
                        error!("Aaronia capture could not start a runtime: {e}");
                        return;
                    }
                };
                if let Err(e) =
                    runtime.block_on(pump.run(transport, sample_rate_hz, reference_level_dbm))
                {
                    error!("Aaronia capture ended: {e:#}");
                }
            })?;
        Ok(SdrHandle::new(receiver, Stopper::new(stop, thread)))
    }
}

struct Pump {
    block_size: usize,
    channels_hz: Vec<f64>,
    dwell: Duration,
    tx: SyncSender<IqPacket>,
    pool_tx: SyncSender<Vec<Complex32>>,
    pool_rx: Receiver<Vec<Complex32>>,
    stop: Arc<AtomicBool>,
}

impl Pump {
    async fn run(
        self,
        transport: Transport,
        sample_rate_hz: f64,
        reference_level_dbm: f64,
    ) -> anyhow::Result<()> {
        let mut builder = SpectranSourceBuilder::new();
        builder
            .center_frequency_hz(self.channels_hz[0])
            .sample_rate_hz(sample_rate_hz)
            .reference_level_dbm(reference_level_dbm);
        match transport {
            Transport::Http { url, format } => {
                info!("Aaronia HTTP source: {url} (format: {})", format.as_str());
                builder.http_source(url).stream_format(format);
            }
            Transport::Sdk { serial } => {
                builder.force_source_type(SourceType::NativeSdk);
                match serial {
                    Some(s) => {
                        info!("Aaronia native SDK source (serial = {s})");
                        builder.device_serial(s);
                    }
                    None => info!("Aaronia native SDK source (first device found)"),
                }
            }
        }
        let mut source = builder.build().await.context("opening the Spectran")?;
        info!("Aaronia source: {:?}", source.source_info());
        source
            .start_streaming()
            .await
            .context("starting the Spectran stream")?;
        let result = self.hop(&mut source).await;
        let _ = source.stop_streaming().await;
        result
    }

    fn stopped(&self) -> bool {
        self.stop.load(Ordering::SeqCst)
    }

    /// Visit each channel in turn for `dwell`, or hold a lone channel
    /// until stopped.
    async fn hop(&self, source: &mut SpectranSource) -> anyhow::Result<()> {
        let hopping = self.channels_hz.len() > 1;
        let mut next = 0usize;
        let mut current = f64::NAN;
        let mut retune_failures = 0u32;
        // Across hops, so a dead source eventually ends the capture
        // instead of failing once per hop forever. Empty reads are per
        // hop on purpose: a quiet channel is not a dead source.
        let mut read_errors = 0u32;
        // When a packet was last handed on, and the frequency of the
        // last one rejected since, to say why if the silence runs out.
        let mut last_delivered = Instant::now();
        let mut last_rejected_hz: Option<f64> = None;

        while !self.stopped() {
            let channel = self.channels_hz[next];
            next = (next + 1) % self.channels_hz.len();

            // One channel is tuned once; re-tuning it every lap would
            // cost the settle for nothing.
            if channel != current {
                if let Err(e) = source.set_center_frequency_hz(channel).await {
                    retune_failures += 1;
                    anyhow::ensure!(
                        retune_failures < ERROR_BAILOUT,
                        "{ERROR_BAILOUT} consecutive retune failures, the last: {e}"
                    );
                    warn!("Aaronia retune to {:.3} MHz failed: {e}", channel / 1e6);
                    tokio::time::sleep(Duration::from_millis(100)).await;
                    continue;
                }
                retune_failures = 0;
                current = channel;
                drain(source, self.block_size, RETUNE_SETTLE).await;
            }

            let deadline = Instant::now() + self.dwell;
            let mut empty_reads = 0u32;
            loop {
                if self.stopped() || (hopping && Instant::now() >= deadline) {
                    break;
                }
                let budget = if hopping {
                    deadline.saturating_duration_since(Instant::now())
                } else {
                    HOLD_READ_BUDGET
                };
                let mut buf = self
                    .pool_rx
                    .try_recv()
                    .unwrap_or_else(|_| Vec::with_capacity(self.block_size));
                buf.clear();
                // Bounded by the dwell: an unbounded read waits out the
                // source's 30 s read timeout on a stalled server and
                // starves every remaining hop.
                let n = match source
                    .read_samples_deadline(&mut buf, self.block_size, budget)
                    .await
                {
                    Ok(n) => {
                        read_errors = 0;
                        n
                    }
                    // A dwell that ends before samples arrive is an empty
                    // read, not a broken source.
                    Err(sdr_aaronia_rs::Error::Io(ref e))
                        if e.kind() == std::io::ErrorKind::TimedOut =>
                    {
                        0
                    }
                    Err(e) => {
                        read_errors += 1;
                        anyhow::ensure!(
                            read_errors < ERROR_BAILOUT,
                            "{ERROR_BAILOUT} consecutive read errors, the last: {e}"
                        );
                        warn!("Aaronia read error: {e}");
                        break;
                    }
                };
                if last_delivered.elapsed() >= SILENCE_BAILOUT {
                    match last_rejected_hz {
                        Some(hz) => anyhow::bail!(
                            "no usable samples for {SILENCE_BAILOUT:?}: packets report {:.3} MHz, not the commanded {:.3} MHz",
                            hz / 1e6,
                            channel / 1e6
                        ),
                        None => anyhow::bail!(
                            "no samples for {SILENCE_BAILOUT:?}; is the analyzer running in RTSA-Suite?"
                        ),
                    }
                }
                if n == 0 {
                    let _ = self.pool_tx.try_send(buf);
                    if hopping {
                        empty_reads += 1;
                        if empty_reads >= EMPTY_READ_BAILOUT {
                            warn!(
                                "{EMPTY_READ_BAILOUT} empty reads on {:.3} MHz; moving on",
                                channel / 1e6
                            );
                            break;
                        }
                        // Not once the dwell is over: the next pass
                        // moves on, and the sleep would only delay it.
                        if Instant::now() < deadline {
                            tokio::time::sleep(Duration::from_millis(5)).await;
                        }
                    }
                    continue;
                }
                empty_reads = 0;
                // Tag the packet with where the stream says it was
                // captured, not where it was just told to go: the first
                // reads after a hop can still carry the old centre, and
                // labelling those with the new one reports a transmitter
                // where it was never received. Stale ones are dropped.
                let captured_hz = source.capture_frequency_hz();
                let Some(center) = packet_frequency(captured_hz, channel) else {
                    last_rejected_hz = Some(captured_hz);
                    let _ = self.pool_tx.try_send(buf);
                    continue;
                };
                last_delivered = Instant::now();
                last_rejected_hz = None;
                let pkt = IqPacket {
                    samples: IqBuffer::new_pooled(buf, self.pool_tx.clone()),
                    center_frequency_hz: center,
                    // Read per packet: over HTTP the rate the device
                    // streams is only known once packets flow, and snaps
                    // to its decimation ladder.
                    sample_rate_hz: source.sample_rate_hz() as f32,
                    overrun: source.take_overrun(),
                };
                if !self.deliver(pkt).await {
                    return Ok(());
                }
            }
        }
        Ok(())
    }

    /// Hand `pkt` to the viewer, waiting while its queue is full; see
    /// [`super::deliver`]. `false` means stop.
    async fn deliver(&self, mut pkt: IqPacket) -> bool {
        loop {
            match self.tx.try_send(pkt) {
                Ok(()) => return true,
                Err(TrySendError::Disconnected(_)) => return false,
                Err(TrySendError::Full(p)) => {
                    if self.stopped() {
                        return false;
                    }
                    pkt = p;
                    tokio::time::sleep(Duration::from_millis(1)).await;
                }
            }
        }
    }
}

/// The frequency to stamp a packet with, or `None` for one still
/// carrying the previous centre. A source that reports no capture
/// frequency (0.0 — the native SDK) is trusted to be where it was told.
fn packet_frequency(captured_hz: f64, commanded_hz: f64) -> Option<f64> {
    if captured_hz > 0.0 {
        ((captured_hz - commanded_hz).abs() <= STALE_TOLERANCE_HZ).then_some(captured_hz)
    } else {
        Some(commanded_hz)
    }
}

/// Read and discard until `settle` has passed, so the hop's real reads
/// start on fresh samples.
async fn drain(source: &mut SpectranSource, block_size: usize, settle: Duration) {
    let deadline = Instant::now() + settle;
    let mut scratch = Vec::with_capacity(block_size);
    loop {
        let left = deadline.saturating_duration_since(Instant::now());
        if left.is_zero() {
            break;
        }
        scratch.clear();
        match tokio::time::timeout(left, source.read_samples(&mut scratch, block_size)).await {
            Ok(Ok(_)) => continue,
            Ok(Err(_)) | Err(_) => break,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_packet_within_tolerance_keeps_its_reported_frequency() {
        // The V6 ECO's -0.79 ppm at 5.93 GHz.
        assert_eq!(
            packet_frequency(5_930_995_294.0, 5_931_000_000.0),
            Some(5_930_995_294.0)
        );
    }

    #[test]
    fn the_tolerance_is_where_the_line_falls() {
        let cmd = 5_865_000_000.0;
        let edge = STALE_TOLERANCE_HZ;
        assert!(packet_frequency(cmd - edge + 1.0, cmd).is_some());
        assert!(packet_frequency(cmd + edge - 1.0, cmd).is_some());
        assert!(packet_frequency(cmd - edge - 1.0, cmd).is_none());
        assert!(packet_frequency(cmd + edge + 1.0, cmd).is_none());
    }

    #[test]
    fn a_packet_from_the_previous_hop_is_dropped() {
        assert_eq!(packet_frequency(5_800_000_000.0, 5_849_000_000.0), None);
    }

    #[test]
    fn a_source_without_capture_frequency_is_trusted() {
        assert_eq!(
            packet_frequency(0.0, 5_865_000_000.0),
            Some(5_865_000_000.0)
        );
    }
}
