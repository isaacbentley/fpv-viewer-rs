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
        let mtu = stream.mtu().ok().filter(|&m| m > 0).unwrap_or(block_size);

        let depth = queue_depth(block_size);
        let (tx, receiver) = std::sync::mpsc::sync_channel::<IqPacket>(depth);
        let (pool_tx, pool_rx) = std::sync::mpsc::sync_channel::<Vec<Complex32>>(depth);
        let stop = Arc::new(AtomicBool::new(false));
        let capture = Capture {
            device,
            channel,
            stream,
            mtu,
            mtu_rem: 0,
            sample_rate,
            block_size,
            settle_samples: (sample_rate * settle.as_secs_f64()).round() as usize,
            scratch: vec![Complex32::default(); block_size.max(mtu).min(65_536)],
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
    /// Driver transfer buffer size (`getStreamMTU`), used to keep reads aligned
    /// to driver ring-buffer entries across `deactivate`/`activate`.
    mtu: usize,
    /// Samples remaining in the driver's currently acquired MTU buffer.
    ///
    /// Drivers such as `SoapyHackRF` wrap `acquireReadBuffer`/`releaseReadBuffer`
    /// in `readStream` using a `remainderHandle`/`remainderSamps` pair, and
    /// `activateStream` resets `buf_count = 0` without resetting
    /// `remainderHandle = -1`. If `deactivateStream` is called while
    /// `remainderSamps > 0`, the first read after the next `activateStream`
    /// finishes the stale remainder buffer and calls `releaseReadBuffer`,
    /// underflowing `buf_count` from `0` to `-1` (`0xFFFFFFFF`) so the driver
    /// spins through uninitialised ring buffers without waiting for hardware.
    mtu_rem: usize,
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
            let mut delivered_on_hop = false;

            loop {
                if self.stopped() {
                    break;
                }
                if hopping && delivered_on_hop && Instant::now() >= deadline {
                    break;
                }
                let mut block = self
                    .pool_rx
                    .try_recv()
                    .unwrap_or_else(|_| Vec::with_capacity(self.block_size));
                match self.fill(&mut block, &mut overrun, &mut errors, &mut timeouts)? {
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
                delivered_on_hop = true;
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
            // Drain any in-hand MTU remainder buffer before deactivating so the
            // driver releases `remainderHandle` while `buf_count >= 1`.
            while self.mtu_rem > 0 {
                let n = self.mtu_rem.min(self.scratch.len());
                match read_aligned(
                    &mut self.stream,
                    self.mtu,
                    &mut self.mtu_rem,
                    &mut self.scratch[..n],
                    Duration::ZERO,
                ) {
                    Ok(got) if got > 0 => {}
                    _ => {
                        self.mtu_rem = 0;
                        break;
                    }
                }
            }
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
            match read_aligned(
                &mut self.stream,
                self.mtu,
                &mut self.mtu_rem,
                &mut self.scratch[..n],
                READ_TIMEOUT,
            ) {
                Ok(0) => break,
                Ok(got) => left = left.saturating_sub(got),
                Err(e) if e.code == ErrorCode::Overflow => {}
                Err(e) if e.code == ErrorCode::Timeout => break,
                Err(e) => return Err(e).context("reading the RX stream"),
            }
        }
        Ok(tuned)
    }

    /// Read until `block` holds `block_size` contiguous samples (or fewer if
    /// an overflow interrupted a non-empty partial fill).
    fn fill(
        &mut self,
        block: &mut Vec<Complex32>,
        overrun: &mut bool,
        errors: &mut u32,
        timeouts: &mut u32,
    ) -> anyhow::Result<Fill> {
        block.clear();
        block.resize(self.block_size, Complex32::default());
        let mut filled = 0;
        let mut empty_overflows = 0u32;
        while filled < self.block_size {
            if self.stopped() {
                return Ok(Fill::Stopped);
            }
            let read = read_aligned(
                &mut self.stream,
                self.mtu,
                &mut self.mtu_rem,
                &mut block[filled..],
                READ_TIMEOUT,
            );
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
                    empty_overflows = 0;
                }
                Err(e) if e.code == ErrorCode::Overflow => {
                    *overrun = true;
                    if filled > 0 {
                        block.truncate(filled);
                        return Ok(Fill::Full);
                    }
                    empty_overflows += 1;
                    anyhow::ensure!(
                        empty_overflows < ERROR_BAILOUT,
                        "{ERROR_BAILOUT} consecutive empty overflows; the device cannot keep up"
                    );
                    return Ok(Fill::Overflowed);
                }
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

/// Advance `mtu_rem` after a read of `got` samples against a driver MTU of `mtu`.
fn advance_mtu_rem(mtu: usize, mtu_rem: usize, got: usize) -> usize {
    if got == 0 || mtu == 0 {
        return mtu_rem;
    }
    if mtu_rem > 0 {
        mtu_rem.saturating_sub(got)
    } else {
        (mtu - (got % mtu)) % mtu
    }
}

/// Read at most one driver MTU boundary worth of samples so a single
/// `readStream` call never consumes a remainder buffer and acquires a
/// new buffer in the same call.
fn read_aligned(
    stream: &mut RxStream<Complex32>,
    mtu: usize,
    mtu_rem: &mut usize,
    dst: &mut [Complex32],
    timeout: Duration,
) -> Result<usize, soapysdr::Error> {
    let step = if *mtu_rem > 0 { *mtu_rem } else { mtu.max(1) };
    let n = dst.len().min(step);
    match stream.read(&mut [&mut dst[..n]], micros(timeout)) {
        Ok(got) => {
            *mtu_rem = advance_mtu_rem(mtu, *mtu_rem, got);
            Ok(got)
        }
        Err(e) => {
            if e.code == ErrorCode::Overflow {
                *mtu_rem = 0;
            }
            Err(e)
        }
    }
}

fn micros(d: Duration) -> i64 {
    d.as_micros() as i64
}

#[cfg(test)]
mod tests {
    use super::*;
    use fpv_drone_analog::scanner::{ScanProgress, detect_packets_per_hop};

    #[test]
    fn mtu_remainder_tracks_hackrf_settle_and_blocks_exact() {
        let mtu = 131_072;
        let mut rem = 0;

        // 2 ms settle at 20 MS/s = 40,000 samples -> leaves 91,072 in current MTU buffer.
        rem = advance_mtu_rem(mtu, rem, 40_000);
        assert_eq!(rem, 91_072);

        // First 65,536-sample block takes 65,536 from the remainder -> leaves 25,536.
        rem = advance_mtu_rem(mtu, rem, 65_536);
        assert_eq!(rem, 25_536);

        // Second 65,536-sample block is split by read_aligned into 25,536 (finishing
        // the first MTU buffer) followed by 40,000 (from the second MTU buffer).
        rem = advance_mtu_rem(mtu, rem, 25_536);
        assert_eq!(rem, 0);
        rem = advance_mtu_rem(mtu, rem, 40_000);
        assert_eq!(rem, 91_072);

        // Draining in 65,536-sample scratch chunks before deactivate reaches 0.
        rem = advance_mtu_rem(mtu, rem, 65_536);
        assert_eq!(rem, 25_536);
        rem = advance_mtu_rem(mtu, rem, 25_536);
        assert_eq!(rem, 0);
    }

    #[test]
    #[ignore = "requires a physical HackRF One connected via USB with SoapyHackRF"]
    fn live_hackrf_soapy_qualification() {
        let devs = soapysdr::enumerate("driver=hackrf").expect("enumerate SoapySDR hackrf");
        assert!(
            !devs.is_empty(),
            "expected at least one HackRF via SoapySDR"
        );
        println!("Found {} SoapySDR HackRF device(s): {}", devs.len(), devs[0]);

        let device = Device::new("driver=hackrf").expect("open driver=hackrf");
        let hw = device.hardware_key().expect("hardware_key");
        let drv = device.driver_key().expect("driver_key");
        let max_rate = max_sample_rate(&device, 0).expect("max_sample_rate");
        let gains = device.list_gains(Direction::Rx, 0).expect("list_gains");
        let antennas = device.antennas(Direction::Rx, 0).expect("antennas");
        let bw = device.bandwidth(Direction::Rx, 0).unwrap_or(0.0);
        println!(
            "Opened {hw} ({drv}), max_rate = {:.2} MS/s, bw = {:.2} MHz, gains = {gains:?}, antennas = {antennas:?}",
            max_rate / 1e6,
            bw / 1e6
        );
        assert_eq!(max_rate, 20_000_000.0);

        // Configure gains and bias-tee off
        device.write_setting("bias_tx", "false").expect("bias_tx=false");
        device.set_gain_element(Direction::Rx, 0, "AMP", 0.0).expect("AMP=0");
        device.set_gain_element(Direction::Rx, 0, "LNA", 16.0).expect("LNA=16");
        device.set_gain_element(Direction::Rx, 0, "VGA", 20.0).expect("VGA=20");

        let actual_rate = apply_sample_rate(&device, 0, 20_000_000.0).expect("apply 20 MS/s");
        assert_eq!(actual_rate, 20_000_000.0);

        // 1. Single-channel hold test (1 second at 20 MS/s)
        let source = Box::new(SoapySource {
            device: device.clone(),
            channel: 0,
            block_size: 65_536,
            settle: Duration::from_millis(2),
        });
        let t0 = Instant::now();
        let handle = source
            .start(SourceConfig::hold(20_000_000.0, 5_800_000_000.0))
            .expect("start single-channel SoapySource");

        let first = handle
            .receiver
            .recv_timeout(Duration::from_secs(2))
            .expect("first packet");
        let ttfp = t0.elapsed();
        assert_eq!(first.samples.len(), 65_536);
        assert!((first.center_frequency_hz - 5_800_000_000.0).abs() < 1e3);
        assert_eq!(first.sample_rate_hz, 20_000_000.0);
        let power: f32 =
            first.samples.iter().map(|c| c.norm_sqr()).sum::<f32>() / first.samples.len() as f32;
        println!(
            "[1/3] Single-channel hold: TTFB = {:?}, first packet len = {}, mean power = {:.6}",
            ttfp,
            first.samples.len(),
            power
        );
        assert!(power > 0.0 && power.is_finite());

        let stream_start = Instant::now();
        let mut pkt_count = 1usize;
        let mut overruns = usize::from(first.overrun);
        while stream_start.elapsed() < Duration::from_secs(1) {
            let pkt = handle
                .receiver
                .recv_timeout(Duration::from_secs(1))
                .expect("sustained packet");
            pkt_count += 1;
            if pkt.overrun {
                overruns += 1;
            }
        }
        let elapsed = stream_start.elapsed();
        handle.stop();
        let msps = (pkt_count as f64 * 65_536.0) / elapsed.as_secs_f64() / 1e6;
        println!(
            "[1/3] Sustained 1s: {} packets ({:.2} MS/s), overruns = {}",
            pkt_count, msps, overruns
        );
        assert!(msps > 18.0 && msps < 22.0, "expected ~20 MS/s, got {msps:.2} MS/s");

        // 2. Full 16-hop 5.8 GHz FPV band scan on the SAME cloned device
        let hops = crate::build_scan_hops(20_000_000.0, crate::ScanBands::Band58);
        assert_eq!(hops.len(), 16);
        let budget_pkts = detect_packets_per_hop(20_000_000.0);
        println!(
            "[2/3] Testing full 16-hop 5.8 GHz scan (budget = {} pkts/hop, dwell = 20 ms)...",
            budget_pkts
        );
        let source2 = Box::new(SoapySource {
            device: device.clone(),
            channel: 0,
            block_size: 65_536,
            settle: Duration::from_millis(2),
        });
        let handle2 = source2
            .start(SourceConfig {
                sample_rate_hz: 20_000_000.0,
                channels_hz: hops.clone(),
                dwell: Duration::from_millis(20),
            })
            .expect("start 16-hop SoapySource");

        let mut progress = ScanProgress::new(&hops, budget_pkts);
        let mut hop_runs: Vec<(u64, usize, Duration)> = Vec::new();
        let mut cur_freq: Option<u64> = None;
        let mut cur_count = 0usize;
        let mut hop_start = Instant::now();
        let mut sweep_completed_at: Option<(usize, Duration)> = None;
        let sweep_t0 = Instant::now();

        for i in 0..(hops.len() * 8) {
            let pkt = handle2
                .receiver
                .recv_timeout(Duration::from_secs(2))
                .expect("hop packet");
            let f = crate::snap_to_nearest_hop(pkt.center_frequency_hz, &hops);
            if cur_freq != Some(f) {
                if let Some(prev) = cur_freq {
                    hop_runs.push((prev, cur_count, hop_start.elapsed()));
                }
                cur_freq = Some(f);
                cur_count = 0;
                hop_start = Instant::now();
            }
            cur_count += 1;
            if let Some(obs) = progress.observe(f)
                && obs.sweep_complete
            {
                sweep_completed_at = Some((i + 1, sweep_t0.elapsed()));
                hop_runs.push((f, cur_count, hop_start.elapsed()));
                break;
            }
        }
        handle2.stop();

        println!(
            "[2/3] 16-hop scan completed at: {:?} ({} hop runs recorded)",
            sweep_completed_at,
            hop_runs.len()
        );
        assert_eq!(hop_runs.len(), 16, "expected all 16 hops in one sweep pass");
        assert!(
            sweep_completed_at.is_some(),
            "16-hop 5.8 GHz sweep at 20 MS/s must complete in one pass"
        );
        for &(freq, count, dur) in &hop_runs {
            assert!(
                count >= budget_pkts,
                "hop {freq} delivered only {count} packets (< budget {budget_pkts})"
            );
            assert!(
                dur >= Duration::from_millis(15),
                "hop {freq} delivered {count} packets in {dur:?} (faster than hardware clock!)"
            );
        }

        // 3. Test 16 MS/s (stepped-down rate) + live detector pass
        let short_hops = hops[..4].to_vec();
        let rate_16m = apply_sample_rate(&device, 0, 16_000_000.0).expect("apply 16 MS/s");
        assert_eq!(rate_16m, 16_000_000.0);
        let source3 = Box::new(SoapySource {
            device: device.clone(),
            channel: 0,
            block_size: 65_536,
            settle: Duration::from_millis(2),
        });
        let handle3 = source3
            .start(SourceConfig {
                sample_rate_hz: 16_000_000.0,
                channels_hz: short_hops.clone(),
                dwell: Duration::from_millis(20),
            })
            .expect("start 16 MS/s SoapySource");

        let mut progress16 = ScanProgress::new(&short_hops, budget_pkts);
        let mut completed16 = false;
        let detector = fpv_drone_analog::scanner::receiver_detector();
        let mut integrator = fpv_drone_analog::detector::SpectralIntegrator::new(4);
        let mut probes = Vec::new();

        for _ in 0..32 {
            let pkt = handle3
                .receiver
                .recv_timeout(Duration::from_secs(2))
                .expect("16 MS/s hop packet");
            assert!(pkt.samples.iter().all(|c| c.re.is_finite() && c.im.is_finite()));
            let f = crate::snap_to_nearest_hop(pkt.center_frequency_hz, &short_hops);
            if let Some(obs) = progress16.observe(f) {
                let _hits = detector.detect_from_iq_integrated_with_probes(
                    &pkt.samples,
                    f,
                    pkt.sample_rate_hz.round() as u32,
                    &mut integrator,
                    &mut probes,
                );
                assert!(!probes.is_empty(), "detector should populate probe energies");
                if obs.sweep_complete {
                    completed16 = true;
                    break;
                }
            }
        }
        handle3.stop();
        println!("[3/3] 16 MS/s 4-hop detector integration completed = {}", completed16);
        assert!(completed16, "16 MS/s 4-hop sweep must complete");
    }
}

