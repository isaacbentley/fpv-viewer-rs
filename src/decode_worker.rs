use orecchiette_fpv_drone_analog_rs::decode::{
    DecodeConfigError, DecoderConfig, StreamingFpvDecoder,
};
use orecchiette_fpv_drone_analog_rs::timing::Standard;
use std::sync::{Arc, mpsc};
use std::thread;

pub type FrameSlot = Arc<std::sync::Mutex<Option<Vec<u32>>>>;

pub fn lock_slot(slot: &FrameSlot) -> std::sync::MutexGuard<'_, Option<Vec<u32>>> {
    slot.lock().unwrap_or_else(|e| e.into_inner())
}

/// Transport ownership stays outside the hardware-independent decoder crate.
pub struct IqChunk {
    pub samples: Arc<orecchiette_sdr_source_rs::PooledIqBuffer>,
    pub discontinuous: bool,
}

pub struct DecodeWorkerConfig {
    pub decoder: DecoderConfig,
    pub display_mhz: f64,
    #[allow(dead_code)]
    pub denoise_model: Option<String>,
}

pub struct DecodeWorker {
    config: DecodeWorkerConfig,
    decoder: StreamingFpvDecoder,
    frame_buf: Vec<u32>,
    producer_slot: FrameSlot,
    frame_tx: mpsc::SyncSender<()>,
    recycle_rx: mpsc::Receiver<Vec<u32>>,
    worker_frames: Arc<std::sync::atomic::AtomicU64>,
    observed_timing_fields: Arc<std::sync::atomic::AtomicU64>,
    snap_tx: mpsc::Sender<(String, Vec<u8>, u32, u32)>,
    frame_count: u64,
    pal_debug_done: bool,
    #[cfg(feature = "neural-vsr")]
    denoise_on: Arc<std::sync::atomic::AtomicBool>,
}

impl DecodeWorker {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        config: DecodeWorkerConfig,
        producer_slot: FrameSlot,
        frame_tx: mpsc::SyncSender<()>,
        recycle_rx: mpsc::Receiver<Vec<u32>>,
        worker_frames: Arc<std::sync::atomic::AtomicU64>,
        observed_timing_fields: Arc<std::sync::atomic::AtomicU64>,
        snap_tx: mpsc::Sender<(String, Vec<u8>, u32, u32)>,
        #[cfg(feature = "neural-vsr")] denoise_on: Arc<std::sync::atomic::AtomicBool>,
    ) -> Result<Self, DecodeConfigError> {
        #[allow(unused_mut)]
        let mut decoder = StreamingFpvDecoder::new(config.decoder.clone())?;
        #[cfg(feature = "neural-vsr")]
        if let Some(model) = &config.denoise_model {
            if let Err(error) = decoder.load_neural_restorer(model, true) {
                eprintln!("Denoiser unavailable: {error} — continuing without it.");
            }
            decoder.set_restoration_enabled(denoise_on.load(std::sync::atomic::Ordering::Relaxed));
        }
        let frame_buf = vec![0; decoder.reconstructor().width * decoder.reconstructor().height];
        Ok(Self {
            config,
            decoder,
            frame_buf,
            producer_slot,
            frame_tx,
            recycle_rx,
            worker_frames,
            observed_timing_fields,
            snap_tx,
            frame_count: 0,
            pal_debug_done: false,
            #[cfg(feature = "neural-vsr")]
            denoise_on,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn spawn(
        config: DecodeWorkerConfig,
        iq_rx: mpsc::Receiver<IqChunk>,
        producer_slot: FrameSlot,
        frame_tx: mpsc::SyncSender<()>,
        recycle_rx: mpsc::Receiver<Vec<u32>>,
        worker_frames: Arc<std::sync::atomic::AtomicU64>,
        observed_timing_fields: Arc<std::sync::atomic::AtomicU64>,
        snap_tx: mpsc::Sender<(String, Vec<u8>, u32, u32)>,
        #[cfg(feature = "neural-vsr")] denoise_on: Arc<std::sync::atomic::AtomicBool>,
    ) -> Result<thread::JoinHandle<()>, DecodeConfigError> {
        let worker = Self::new(
            config,
            producer_slot,
            frame_tx,
            recycle_rx,
            worker_frames,
            observed_timing_fields,
            snap_tx,
            #[cfg(feature = "neural-vsr")]
            denoise_on,
        )?;
        Ok(thread::spawn(move || worker.run(iq_rx)))
    }

    pub fn run(mut self, iq_rx: mpsc::Receiver<IqChunk>) {
        while let Ok(chunk) = iq_rx.recv() {
            if !self.process_chunk(chunk) {
                return;
            }
        }
    }

    fn process_chunk(&mut self, chunk: IqChunk) -> bool {
        #[cfg(feature = "neural-vsr")]
        self.decoder
            .set_restoration_enabled(self.denoise_on.load(std::sync::atomic::Ordering::Relaxed));
        self.decoder.push_iq(&chunk.samples, chunk.discontinuous);
        if self.config.decoder.debug
            && self.config.decoder.standard == Standard::Pal
            && !self.pal_debug_done
        {
            let samples = self.decoder.demodulated_chunk();
            if !samples.is_empty() {
                let min = samples.iter().copied().fold(f32::INFINITY, f32::min);
                let max = samples.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mean = samples.iter().sum::<f32>() / samples.len() as f32;
                eprintln!(
                    "[PAL DEBUG] first demod chunk: min={min:.4} max={max:.4} mean={mean:.4}"
                );
                self.pal_debug_done = true;
            }
        }
        loop {
            let timing = match self.decoder.next_field_into(&mut self.frame_buf) {
                Ok(Some(timing)) => timing,
                Ok(None) => break,
                Err(error) => {
                    eprintln!("Decode worker failed: {error}");
                    return false;
                }
            };
            self.frame_count += 1;
            self.worker_frames
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if timing.has_observed_timing_evidence {
                self.observed_timing_fields
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }

            let metrics_due = self.config.decoder.debug
                && (self.frame_count <= 3 || self.frame_count.is_multiple_of(30));
            if metrics_due {
                let cnr = self
                    .decoder
                    .cnr_db()
                    .map(|v| format!("{v:.1} dB"))
                    .unwrap_or_else(|| "n/a".into());
                let lock = self
                    .decoder
                    .pll_phase_error_rms()
                    .map(|err| format!(" | PLLerr: {err:.2} rad"))
                    .unwrap_or_default();
                println!("[DEBUG LINK] CNRest: {cnr}{lock}");
                let timing_state = if !self.decoder.reconstructor().timing_tracker().is_locked {
                    "Searching"
                } else if timing.coasted_field_count > 0 {
                    "Coasting"
                } else {
                    "Tracking"
                };
                println!(
                    "[DEBUG METRICS] Frame {} | Std: {:?} | LinePer: {:.2} samples | SyncQ: {:.2} | Y_avg: {:+.3} | HistD: {} | State: {} | Evidence: {} | Coasted: {}",
                    self.frame_count,
                    self.decoder.reconstructor().video_standard(),
                    self.decoder.reconstructor().line_period_samples(),
                    self.decoder.reconstructor().latest_sync_quality(),
                    self.decoder.reconstructor().latest_mean_amplitude(),
                    self.decoder.reconstructor().history_depth(),
                    timing_state,
                    timing.has_observed_timing_evidence,
                    timing.coasted_field_count,
                );
            }

            let width = self.decoder.reconstructor().width;
            let height = self.decoder.reconstructor().height;

            if self.config.decoder.debug
                && (self.frame_count <= 3 || (30..=32).contains(&self.frame_count))
            {
                let path = format!(
                    "fpv_frame_{:.0}MHz_{}.png",
                    self.config.display_mhz, self.frame_count
                );
                let mut rgb_buf = vec![0u8; width * height * 3];
                for (i, &pixel) in self.frame_buf.iter().enumerate() {
                    rgb_buf[i * 3] = ((pixel >> 16) & 0xFF) as u8;
                    rgb_buf[i * 3 + 1] = ((pixel >> 8) & 0xFF) as u8;
                    rgb_buf[i * 3 + 2] = (pixel & 0xFF) as u8;
                }
                let _ = self
                    .snap_tx
                    .send((path, rgb_buf, width as u32, height as u32));
            }

            let displaced =
                lock_slot(&self.producer_slot).replace(std::mem::take(&mut self.frame_buf));
            self.frame_buf = displaced
                .or_else(|| self.recycle_rx.try_recv().ok())
                .unwrap_or_else(|| vec![0u32; width * height]);

            match self.frame_tx.try_send(()) {
                Ok(()) | Err(mpsc::TrySendError::Full(())) => {}
                Err(mpsc::TrySendError::Disconnected(())) => return false,
            }
        }
        // The application chooses its latency budget. The decoder performs
        // the discard and invalidates all affected timing/picture references.
        let work_rate = self.decoder.plan().work_rate() as usize;
        if self.decoder.buffered_samples() > work_rate / 12 {
            self.decoder.discard_pending_except(work_rate / 60);
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex;
    use orecchiette_fpv_drone_analog_rs::synthetic::{
        SyntheticVideoConfig, TestPattern, generate_iq,
    };
    use orecchiette_fpv_drone_analog_rs::vbi::FieldParity;
    use std::sync::atomic::{AtomicU64, Ordering};

    fn worker(is_pal: bool, decim: usize) -> (DecodeWorker, mpsc::Receiver<()>) {
        let config = DecodeWorkerConfig {
            decoder: DecoderConfig {
                plan: orecchiette_fpv_drone_analog_rs::decode::DecodePlan::new(
                    15_360_000 * decim as u32,
                    3_000_000.0,
                    10_000_000.0,
                    orecchiette_fpv_drone_analog_rs::decode::DemodulationMode::Discriminator,
                )
                .unwrap(),
                frequency_offset_hz: 0.0,
                standard: Standard::from_is_pal(is_pal),
                deemphasis_tau_s: 0.75e-6,
                temporal_window: 3,
                debug: false,
            },
            display_mhz: 5865.0,
            denoise_model: None,
        };
        let (frame_tx, frame_rx) = mpsc::sync_channel(1);
        let (_, recycle_rx) = mpsc::channel();
        let (snap_tx, _) = mpsc::channel();
        let worker = DecodeWorker::new(
            config,
            Arc::new(std::sync::Mutex::new(None)),
            frame_tx,
            recycle_rx,
            Arc::new(AtomicU64::new(0)),
            Arc::new(AtomicU64::new(0)),
            snap_tx,
            #[cfg(feature = "neural-vsr")]
            Arc::new(std::sync::atomic::AtomicBool::new(false)),
        );
        (worker.unwrap(), frame_rx)
    }

    fn iq(is_pal: bool, decim: usize) -> Vec<Complex<f32>> {
        generate_iq(
            &SyntheticVideoConfig {
                sample_rate: 15_360_000 * decim as u32,
                is_pal,
                deviation_hz: 3_000_000.0,
                pattern: TestPattern::Bars,
                start_field: FieldParity::First,
                noise_sigma: 0.0,
                dc_offset: 0.0,
            },
            4,
            0.0,
        )
    }

    fn chunk(samples: &[Complex<f32>], discontinuous: bool) -> IqChunk {
        IqChunk {
            samples: Arc::new(orecchiette_sdr_source_rs::PooledIqBuffer::new_unpooled(
                samples.to_vec(),
            )),
            discontinuous,
        }
    }

    #[test]
    fn production_worker_is_invariant_to_iq_chunk_boundaries_and_compaction() {
        for is_pal in [false, true] {
            for decim in [1, 2] {
                let samples = iq(is_pal, decim);
                let (mut small, _small_rx) = worker(is_pal, decim);
                let (mut large, _large_rx) = worker(is_pal, decim);
                for part in samples.chunks(16_381) {
                    assert!(small.process_chunk(chunk(part, false)));
                }
                for part in samples.chunks(131_071) {
                    assert!(large.process_chunk(chunk(part, false)));
                }
                let count = small.worker_frames.load(Ordering::Relaxed);
                assert!(
                    count >= 3,
                    "PAL={is_pal} decim={decim}: only {count} fields"
                );
                assert_eq!(large.worker_frames.load(Ordering::Relaxed), count);
                assert_eq!(small.observed_timing_fields.load(Ordering::Relaxed), count);
                assert_eq!(large.observed_timing_fields.load(Ordering::Relaxed), count);
                assert_eq!(
                    small.decoder.sample_coordinate(),
                    large.decoder.sample_coordinate()
                );
                assert_eq!(
                    small.decoder.reconstructor().timing_tracker(),
                    large.decoder.reconstructor().timing_tracker()
                );
                assert_eq!(
                    small.decoder.reconstructor().latest_sync_positions(),
                    large.decoder.reconstructor().latest_sync_positions()
                );
                assert_eq!(
                    *lock_slot(&small.producer_slot),
                    *lock_slot(&large.producer_slot)
                );
                assert_eq!(
                    small
                        .decoder
                        .reconstructor()
                        .timing_tracker()
                        .continuity_epoch,
                    0
                );
                assert!(
                    small.decoder.buffered_samples() < small.decoder.sample_coordinate() as usize,
                    "test must exercise compaction"
                );
            }
        }
    }

    #[test]
    fn source_gap_resets_once_and_accounts_for_discarded_tail() {
        let samples = iq(false, 2);
        let (mut worker, _frame_rx) = worker(false, 2);
        for part in samples.chunks(131_071) {
            assert!(worker.process_chunk(chunk(part, false)));
        }
        assert!(worker.decoder.reconstructor().timing_tracker().is_locked);
        let discarded_end =
            worker.decoder.sample_coordinate() + worker.decoder.buffered_samples() as u64;
        assert!(worker.process_chunk(chunk(&samples[..1_000], true)));
        assert_eq!(worker.decoder.sample_coordinate(), discarded_end);
        assert_eq!(
            worker
                .decoder
                .reconstructor()
                .timing_tracker()
                .continuity_epoch,
            1
        );
        assert_eq!(worker.decoder.reconstructor().history_depth(), 0);
        assert!(!worker.decoder.reconstructor().timing_tracker().is_locked);
        for part in samples[1_000..].chunks(16_381) {
            assert!(worker.process_chunk(chunk(part, false)));
        }
        assert!(worker.decoder.reconstructor().timing_tracker().is_locked);
        assert_eq!(
            worker
                .decoder
                .reconstructor()
                .timing_tracker()
                .continuity_epoch,
            1
        );
        assert!(worker.decoder.sample_coordinate() > discarded_end);
    }
}
