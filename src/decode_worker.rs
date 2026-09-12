use num_complex::Complex;
use orecchiette_fpv_drone_analog_rs::ddc::StreamingDDC;
use orecchiette_fpv_drone_analog_rs::demod::{Deemphasis, PllFmDemod, fm_demod_into};
use orecchiette_fpv_drone_analog_rs::levels::estimate_cnr_db;
use orecchiette_fpv_drone_analog_rs::timing::{DecodeStep, TimedDemodSlice};
use orecchiette_fpv_drone_analog_rs::video::FrameReconstructor;
use std::sync::{Arc, mpsc};
use std::thread;

pub type FrameSlot = Arc<std::sync::Mutex<Option<Vec<u32>>>>;

/// Lock a [`FrameSlot`], ignoring poisoning.
pub fn lock_slot(slot: &FrameSlot) -> std::sync::MutexGuard<'_, Option<Vec<u32>>> {
    slot.lock().unwrap_or_else(|e| e.into_inner())
}

/// One capture chunk on its way to the decoder, and whether the signal
/// reaching it runs on from the last one.
pub struct IqChunk {
    pub samples: Arc<orecchiette_sdr_source_rs::PooledIqBuffer>,
    pub discontinuous: bool,
}

pub struct DecodeWorkerConfig {
    pub sample_rate: u32,
    pub freq_offset: f32,
    pub ddc_cutoff: f32,
    pub decim: usize,
    pub work_rate: u32,
    pub use_pll: bool,
    pub pll_loop_bw_hz: f32,
    pub fm_deviation: f32,
    pub deemphasis_tau: f32,
    pub is_pal: bool,
    pub temporal_window: usize,
    pub debug: bool,
    pub display_mhz: f64,
    pub decode_fir_taps: usize,
    #[allow(dead_code)]
    pub denoise_model: Option<String>,
}

pub struct DecodeWorker {
    pub config: DecodeWorkerConfig,
    pub reconstructor: FrameReconstructor,
    ddc: StreamingDDC,
    shifted_iq: Vec<Complex<f32>>,
    iq_carry: Option<Complex<f32>>,
    demod_scratch: Vec<f32>,
    deemph: Option<Deemphasis>,
    pll: Option<PllFmDemod>,
    demod_buffer: Vec<f32>,
    demod_start: usize,
    current_sample_coordinate: u64,
    is_discontinuous: bool,
    compact_after: usize,
    live_region_cap: usize,
    keep_on_skip: usize,
    min_samples_per_field: usize,
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
    stashed_restorer: Option<Box<dyn orecchiette_fpv_drone_analog_rs::video::NeuralRestorer>>,
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
    ) -> Self {
        #[allow(unused_mut)]
        let mut reconstructor = FrameReconstructor::new(
            config.work_rate,
            config.is_pal,
            config.fm_deviation,
            config.debug,
        )
        .with_temporal_window(config.temporal_window);

        #[cfg(feature = "neural-vsr")]
        let stashed_restorer = if let Some(model) = &config.denoise_model {
            reconstructor = reconstructor.with_neural_restorer(model, true);
            if reconstructor.neural_restorer.is_none() {
                eprintln!("Denoiser unavailable: could not load {model} — continuing without it.");
            }
            if denoise_on.load(std::sync::atomic::Ordering::Relaxed) {
                None
            } else {
                reconstructor.neural_restorer.take()
            }
        } else {
            None
        };

        let ddc = if config.decim > 1 {
            StreamingDDC::with_taps(
                config.freq_offset,
                config.sample_rate,
                config.ddc_cutoff,
                config.decode_fir_taps,
            )
        } else {
            StreamingDDC::new(config.freq_offset, config.sample_rate, config.ddc_cutoff)
        };

        let deemph = (config.deemphasis_tau.is_finite() && config.deemphasis_tau > 0.0)
            .then(|| Deemphasis::new(config.work_rate, config.deemphasis_tau));

        let pll = config.use_pll.then(|| {
            PllFmDemod::new(
                config.work_rate,
                config.pll_loop_bw_hz,
                config.fm_deviation * 1.2,
            )
        });

        let min_samples_per_field = {
            let line_rate = if config.is_pal { 15_625.0f32 } else { 15_734.0 };
            let lines = if config.is_pal { 288 } else { 240 } + 22;
            ((config.work_rate as f32 / line_rate) * lines as f32) as usize
        };

        let compact_after = config.work_rate as usize / 30;
        let live_region_cap = config.work_rate as usize / 12;
        let keep_on_skip = config.work_rate as usize / 60;
        let frame_buf = vec![0u32; reconstructor.width * reconstructor.height];

        Self {
            config,
            reconstructor,
            ddc,
            shifted_iq: Vec::new(),
            iq_carry: None,
            demod_scratch: Vec::new(),
            deemph,
            pll,
            demod_buffer: Vec::new(),
            demod_start: 0,
            current_sample_coordinate: 0,
            is_discontinuous: false,
            compact_after,
            live_region_cap,
            keep_on_skip,
            min_samples_per_field,
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
            stashed_restorer,
            #[cfg(feature = "neural-vsr")]
            denoise_on,
        }
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
    ) -> thread::JoinHandle<()> {
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
        );
        thread::spawn(move || worker.run(iq_rx))
    }

    pub fn handle_discontinuity(&mut self) {
        self.iq_carry = None;
        self.ddc.reset();
        if let Some(p) = self.pll.as_mut() {
            p.reset();
        }
        if let Some(d) = self.deemph.as_mut() {
            d.reset();
        }
        self.demod_buffer.clear();
        self.demod_start = 0;
        self.is_discontinuous = true;
        self.reconstructor.forget_history();
        #[cfg(feature = "neural-vsr")]
        {
            self.reconstructor.hidden_state = None;
        }
    }

    pub fn run(mut self, iq_rx: mpsc::Receiver<IqChunk>) {
        while let Ok(iq_chunk) = iq_rx.recv() {
            if iq_chunk.discontinuous {
                self.handle_discontinuity();
            }

            self.shifted_iq.clear();
            if self.pll.is_none()
                && let Some(prev) = self.iq_carry
            {
                self.shifted_iq.push(prev);
            }
            self.ddc.process_into_decimated(
                &iq_chunk.samples,
                &mut self.shifted_iq,
                self.config.decim,
            );
            self.iq_carry = self.shifted_iq.last().copied();

            match self.pll.as_mut() {
                Some(p) => p.process_into(&self.shifted_iq, &mut self.demod_scratch),
                None => fm_demod_into(&self.shifted_iq, &mut self.demod_scratch),
            }
            if let Some(d) = self.deemph.as_mut() {
                d.process_in_place(&mut self.demod_scratch);
            }

            #[cfg(feature = "neural-vsr")]
            {
                let want = self.denoise_on.load(std::sync::atomic::Ordering::Relaxed);
                let have = self.reconstructor.neural_restorer.is_some();
                if want && !have {
                    self.reconstructor.neural_restorer = self.stashed_restorer.take();
                    self.reconstructor.hidden_state = None;
                } else if !want && have {
                    self.stashed_restorer = self.reconstructor.neural_restorer.take();
                }
                if self.reconstructor.neural_restorer.is_some()
                    && let Some(cnr) = estimate_cnr_db(&self.shifted_iq)
                {
                    self.reconstructor.set_neural_noise_level(cnr);
                }
            }

            if self.config.debug
                && self.config.is_pal
                && !self.pal_debug_done
                && !self.demod_scratch.is_empty()
            {
                let min = self
                    .demod_scratch
                    .iter()
                    .cloned()
                    .fold(f32::INFINITY, f32::min);
                let max = self
                    .demod_scratch
                    .iter()
                    .cloned()
                    .fold(f32::NEG_INFINITY, f32::max);
                let mean = self.demod_scratch.iter().sum::<f32>() / self.demod_scratch.len() as f32;
                eprintln!(
                    "[PAL DEBUG] first demod chunk: min={:.4} max={:.4} mean={:.4}",
                    min, max, mean
                );
                self.pal_debug_done = true;
            }

            self.demod_buffer.extend_from_slice(&self.demod_scratch);

            while self.demod_buffer.len() - self.demod_start >= self.min_samples_per_field {
                let slice = TimedDemodSlice::new(
                    &self.demod_buffer[self.demod_start..],
                    self.current_sample_coordinate,
                    self.config.work_rate,
                    self.is_discontinuous,
                );
                self.is_discontinuous = false;

                match self
                    .reconstructor
                    .reconstruct_timed_into(slice, &mut self.frame_buf)
                {
                    Ok(DecodeStep::NeedMoreData { .. }) => break,
                    Ok(DecodeStep::Advance {
                        consumed_samples,
                        field,
                    }) => {
                        self.demod_start += consumed_samples;
                        self.current_sample_coordinate += consumed_samples as u64;

                        if let Some(timing) = field {
                            self.frame_count += 1;
                            self.worker_frames
                                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            if timing.has_observed_timing_evidence {
                                self.observed_timing_fields
                                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            }

                            let metrics_due = self.config.debug
                                && (self.frame_count <= 3 || self.frame_count.is_multiple_of(30));
                            if metrics_due {
                                let cnr = estimate_cnr_db(&self.shifted_iq)
                                    .map(|v| format!("{v:.1} dB"))
                                    .unwrap_or_else(|| "n/a".into());
                                let lock = self
                                    .pll
                                    .as_ref()
                                    .map(|p| format!(" | PLLerr: {:.2} rad", p.phase_error_rms()))
                                    .unwrap_or_default();
                                println!("[DEBUG LINK] CNRest: {cnr}{lock}");
                                let timing_state = if !self.reconstructor.timing_tracker().is_locked
                                {
                                    "Searching"
                                } else if timing.coasted_field_count > 0 {
                                    "Coasting"
                                } else {
                                    "Tracking"
                                };
                                println!(
                                    "[DEBUG METRICS] Frame {} | Std: {:?} | LinePer: {:.2}s | SyncQ: {:.2} | Y_avg: {:+.3} | HistD: {} | State: {} | Evidence: {} | Coasted: {}",
                                    self.frame_count,
                                    self.reconstructor.video_standard(),
                                    self.reconstructor.line_period_samples(),
                                    self.reconstructor.latest_sync_quality(),
                                    self.reconstructor.latest_mean_amplitude(),
                                    self.reconstructor.history_depth(),
                                    timing_state,
                                    timing.has_observed_timing_evidence,
                                    timing.coasted_field_count,
                                );
                            }

                            let width = self.reconstructor.width;
                            let height = self.reconstructor.height;

                            if self.config.debug
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
                                let _ =
                                    self.snap_tx
                                        .send((path, rgb_buf, width as u32, height as u32));
                            }

                            let displaced = lock_slot(&self.producer_slot)
                                .replace(std::mem::take(&mut self.frame_buf));
                            self.frame_buf = displaced
                                .or_else(|| self.recycle_rx.try_recv().ok())
                                .unwrap_or_else(|| vec![0u32; width * height]);

                            match self.frame_tx.try_send(()) {
                                Ok(()) | Err(mpsc::TrySendError::Full(())) => {}
                                Err(mpsc::TrySendError::Disconnected(())) => return,
                            }
                        }
                    }
                    Err(e) => {
                        if self.config.debug {
                            eprintln!("[DECODE ERROR] {:?}", e);
                        }
                        let fallback = self.reconstructor.samples_per_line;
                        self.demod_start += fallback;
                        self.current_sample_coordinate += fallback as u64;
                        break;
                    }
                }
            }

            if self.demod_start > self.compact_after {
                self.demod_buffer.drain(0..self.demod_start);
                self.demod_start = 0;
            }

            if self.demod_buffer.len() - self.demod_start > self.live_region_cap {
                let new_start = self.demod_buffer.len().saturating_sub(self.keep_on_skip);
                let skipped = new_start - self.demod_start;
                self.demod_start = new_start;
                self.current_sample_coordinate += skipped as u64;
                self.reconstructor.forget_history();
                #[cfg(feature = "neural-vsr")]
                {
                    self.reconstructor.hidden_state = None;
                }
            }
        }
    }
}
