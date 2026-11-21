//! The viewer's SDR sources, behind one shape.
//!
//! Two backends deliver live IQ: [`soapy`] (any SoapySDR device —
//! HackRF, USRP via SoapyUHD, LimeSDR, …) and [`aaronia`] (a Spectran V6
//! over RTSA HTTP or the native SDK). Each runs its own capture thread
//! and hands [`IqPacket`]s to the viewer through an [`SdrHandle`], so the
//! scan and decode loops never care which one is plugged in.

#[cfg(feature = "aaronia")]
pub mod aaronia;
#[cfg(feature = "soapy")]
pub mod soapy;

use num_complex::Complex32;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, SyncSender};
use std::thread::JoinHandle;
use std::time::Duration;

/// A buffer of IQ samples that returns itself to its source's pool
/// when dropped, so the capture loop reuses allocations instead of
/// making a fresh one per packet.
#[derive(Debug)]
pub struct IqBuffer {
    vec: Option<Vec<Complex32>>,
    recycler: Option<SyncSender<Vec<Complex32>>>,
}

impl IqBuffer {
    /// A buffer with no pool behind it; it drops normally.
    pub fn new_unpooled(vec: Vec<Complex32>) -> Self {
        Self {
            vec: Some(vec),
            recycler: None,
        }
    }

    /// A buffer that goes back to `recycler` when dropped.
    #[cfg(any(test, feature = "soapy", feature = "aaronia"))]
    pub fn new_pooled(vec: Vec<Complex32>, recycler: SyncSender<Vec<Complex32>>) -> Self {
        Self {
            vec: Some(vec),
            recycler: Some(recycler),
        }
    }
}

impl Drop for IqBuffer {
    fn drop(&mut self) {
        if let Some(mut vec) = self.vec.take()
            && let Some(recycler) = &self.recycler
        {
            vec.clear();
            let _ = recycler.try_send(vec);
        }
    }
}

impl std::ops::Deref for IqBuffer {
    type Target = [Complex32];
    fn deref(&self) -> &Self::Target {
        // `vec` is only taken in `drop`.
        self.vec.as_deref().unwrap_or_default()
    }
}

/// One block of IQ, tagged with where and how fast it was captured.
///
/// `center_frequency_hz` is where the radio says it was tuned for these
/// samples, which can sit a few kHz off the commanded channel (a
/// synthesiser's fractional offset); DSP uses it, hop bookkeeping snaps
/// it back to the plan.
#[derive(Debug)]
pub struct IqPacket {
    pub samples: IqBuffer,
    pub center_frequency_hz: f64,
    pub sample_rate_hz: f32,
    /// Samples were lost between the previous packet and this one.
    pub overrun: bool,
}

/// What a capture should do.
#[derive(Debug, Clone)]
pub struct SourceConfig {
    pub sample_rate_hz: f64,
    /// One entry holds that channel for as long as the capture runs;
    /// several are visited in turn, `dwell` each, round and round.
    pub channels_hz: Vec<f64>,
    /// How long to stay on each channel of a hop list, after the
    /// retune has settled. Ignored for a single channel.
    pub dwell: Duration,
}

impl SourceConfig {
    /// Hold one channel indefinitely.
    pub fn hold(sample_rate_hz: f64, center_hz: f64) -> Self {
        Self {
            sample_rate_hz,
            channels_hz: vec![center_hz],
            dwell: Duration::ZERO,
        }
    }
}

/// An SDR backend, ready to start capturing.
pub trait SdrSource: Send {
    fn start(self: Box<Self>, config: SourceConfig) -> anyhow::Result<SdrHandle>;
}

/// A running capture.
///
/// Packets arrive on `receiver` until the capture ends. [`Self::stop`]
/// asks the capture thread to finish and waits for it, so the hardware
/// is released by the time it returns and the next capture can open it.
pub struct SdrHandle {
    pub receiver: Receiver<IqPacket>,
    stopper: Stopper,
}

impl SdrHandle {
    pub(crate) fn new(receiver: Receiver<IqPacket>, stopper: Stopper) -> Self {
        Self { receiver, stopper }
    }

    /// Split into the packet stream and the stop control, for a caller
    /// that hands the receiver to another thread.
    pub fn split(self) -> (Receiver<IqPacket>, Stopper) {
        (self.receiver, self.stopper)
    }

    pub fn stop(self) {
        self.stopper.stop();
    }
}

/// Ends a capture: raises its stop flag and joins its thread.
pub struct Stopper {
    flag: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl Stopper {
    pub(crate) fn new(flag: Arc<AtomicBool>, thread: JoinHandle<()>) -> Self {
        Self {
            flag,
            thread: Some(thread),
        }
    }

    pub fn stop(mut self) {
        self.finish();
    }

    fn finish(&mut self) {
        self.flag.store(true, Ordering::SeqCst);
        if let Some(thread) = self.thread.take()
            && thread.join().is_err()
        {
            tracing::error!("SDR capture thread panicked");
        }
    }
}

impl Drop for Stopper {
    fn drop(&mut self) {
        self.finish();
    }
}

/// Hand `pkt` to the consumer, waiting while its queue is full.
///
/// Waits in short sleeps rather than a blocking `send` so a raised
/// stop flag is seen even when nobody is reading: [`Stopper::stop`]
/// joins this thread, and a blocking send would never return. `false`
/// means stop — the flag went up or the consumer is gone.
#[cfg(feature = "soapy")]
pub(crate) fn deliver(tx: &SyncSender<IqPacket>, mut pkt: IqPacket, stop: &AtomicBool) -> bool {
    use std::sync::mpsc::TrySendError;
    loop {
        match tx.try_send(pkt) {
            Ok(()) => return true,
            Err(TrySendError::Disconnected(_)) => return false,
            Err(TrySendError::Full(p)) => {
                if stop.load(Ordering::SeqCst) {
                    return false;
                }
                pkt = p;
                std::thread::sleep(Duration::from_millis(1));
            }
        }
    }
}

/// How many packets a capture may queue ahead of the viewer: 64 MiB of
/// samples, and never fewer than one or more than 64 packets.
#[cfg(any(feature = "soapy", feature = "aaronia"))]
pub(crate) fn queue_depth(block_size: usize) -> usize {
    (64 * 1024 * 1024 / (block_size * std::mem::size_of::<Complex32>())).clamp(1, 64)
}

/// Largest I or Q magnitude let into the pipeline.
///
/// A finite sample can still be poison: the FM discriminator multiplies
/// neighbouring samples, so anything past about 1.8e19 overflows `f32`
/// to `inf - inf` and comes out NaN, which the filters with memory
/// behind it (deemphasis, the detector's spectral averages) then hold.
/// 1e9 is far above any real capture (a normalised stream peaks near 1,
/// an unscaled 16-bit one at 32,768) and far below where squares and
/// FIR sums overflow. No radio's ADC produces such a value, but a
/// crafted file or a network peer can.
pub const MAX_IQ_COMPONENT: f32 = 1.0e9;

/// Whether `s` can go into the pipeline as it is. One test covers NaN,
/// infinity and out-of-range, since every comparison with NaN is false.
#[inline]
pub fn iq_sample_is_sane(s: &Complex32) -> bool {
    s.re.abs() <= MAX_IQ_COMPONENT && s.im.abs() <= MAX_IQ_COMPONENT
}

/// `s` made safe to process: a NaN or infinite part becomes zero, a
/// finite one is clamped to ±[`MAX_IQ_COMPONENT`].
#[inline]
pub fn sanitize_iq_sample(s: Complex32) -> Complex32 {
    let part = |v: f32| {
        if v.is_finite() {
            v.clamp(-MAX_IQ_COMPONENT, MAX_IQ_COMPONENT)
        } else {
            0.0
        }
    };
    Complex32::new(part(s.re), part(s.im))
}

/// [`sanitize_iq_sample`] over a block, in place; `true` if anything
/// had to change. Every source runs its samples through this (or the
/// per-sample form) before they reach the detector or a decoder.
pub fn sanitize_iq_in_place(iq: &mut [Complex32]) -> bool {
    if iq.iter().all(iq_sample_is_sane) {
        return false;
    }
    for s in iq.iter_mut() {
        *s = sanitize_iq_sample(*s);
    }
    true
}

/// Whether a packet's reported sample rate can be the capture that was
/// asked for: within a factor of two of it either way.
///
/// A device may run a nearby rate from its own grid, and the planner
/// follows that. A rate far from the request is not this capture at
/// all, and since it sizes buffers and filters downstream, a packet
/// claiming one is dropped rather than believed.
pub fn rate_is_plausible(reported_hz: f64, requested_hz: f64) -> bool {
    reported_hz.is_finite()
        && requested_hz.is_finite()
        && requested_hz > 0.0
        && reported_hz >= requested_hz * 0.5
        && reported_hz <= requested_hz * 2.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clean_samples_are_left_alone() {
        let mut iq = vec![
            Complex32::new(0.5, -0.5),
            Complex32::new(-32_768.0, 32_767.0),
            Complex32::new(MAX_IQ_COMPONENT, -MAX_IQ_COMPONENT),
        ];
        let before = iq.clone();
        assert!(!sanitize_iq_in_place(&mut iq));
        assert_eq!(iq, before);
        assert!(!sanitize_iq_in_place(&mut []));
    }

    #[test]
    fn hostile_samples_are_zeroed_or_clamped() {
        let mut iq = vec![
            Complex32::new(1.0, 2.0),
            Complex32::new(f32::NAN, 2.0),
            Complex32::new(1.0, f32::INFINITY),
            Complex32::new(f32::NEG_INFINITY, f32::NAN),
            Complex32::new(3.0e38, -1.0e30),
        ];
        assert!(sanitize_iq_in_place(&mut iq));
        assert_eq!(
            iq,
            vec![
                Complex32::new(1.0, 2.0),
                Complex32::new(0.0, 2.0),
                Complex32::new(1.0, 0.0),
                Complex32::new(0.0, 0.0),
                Complex32::new(MAX_IQ_COMPONENT, -MAX_IQ_COMPONENT),
            ]
        );
        // What the bound is for: the discriminator's product of two
        // neighbouring samples stays finite, so its angle is a number.
        let worst = Complex32::new(MAX_IQ_COMPONENT, MAX_IQ_COMPONENT);
        assert!((worst * worst.conj()).arg().is_finite());
        let unclamped = Complex32::new(1.0e30, 1.0e30);
        assert!((unclamped * unclamped.conj()).arg().is_nan());
    }

    #[test]
    fn a_rate_near_the_request_is_plausible() {
        let asked = 61_440_000.0;
        for reported in [61_440_000.0, 61_439_998.5, 30_720_000.0, 92_160_000.0] {
            assert!(rate_is_plausible(reported, asked), "{reported}");
        }
        assert!(rate_is_plausible(20_000_000.0, 25_000_000.0));
    }

    #[test]
    fn a_rate_far_from_the_request_is_not() {
        let asked = 61_440_000.0;
        for reported in [
            1.0,
            0.0,
            -61_440_000.0,
            15_360_000.0,
            245_760_000.0,
            f64::INFINITY,
            f64::NAN,
            u32::MAX as f64,
        ] {
            assert!(!rate_is_plausible(reported, asked), "{reported}");
        }
        assert!(!rate_is_plausible(1.0, 0.0));
        assert!(!rate_is_plausible(1.0, f64::NAN));
    }

    #[test]
    fn a_pooled_buffer_returns_to_its_pool() {
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        let buf = IqBuffer::new_pooled(vec![Complex32::new(1.0, 0.0); 4], tx);
        assert_eq!(buf.len(), 4);
        drop(buf);
        let back = rx.try_recv().expect("buffer came back");
        assert!(back.is_empty() && back.capacity() >= 4);
    }

    #[test]
    fn an_unpooled_buffer_reads_its_samples() {
        let buf = IqBuffer::new_unpooled(vec![Complex32::new(0.5, -0.5); 3]);
        assert_eq!(buf[2], Complex32::new(0.5, -0.5));
    }
}
