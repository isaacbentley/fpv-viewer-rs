//! Where the decode worker's time goes, per stage, without an SDR.
//!
//! Follows `run_viewer_pipeline`'s per-channel worker stage for stage —
//! the same down-converter cutoff and tap count, decimation rule,
//! demodulator choice, deemphasis, reconstructor and field guard the
//! live path builds — and feeds it synthetic NTSC at each sample rate
//! the Aaronia backend offers, in the same 65,536-sample chunks the
//! driver delivers. It is not a copy of the worker: buffer compaction
//! is simpler here and the worker's safety valve, debug telemetry and
//! snapshot encoder are all absent, so these are the costs of a decode
//! that is keeping up, not of one falling behind.
//!
//! The number that matters is CPU seconds spent per second of signal.
//! Above 1.0 the worker cannot keep up and the live path drops chunks,
//! which breaks the down-converter's continuity and glitches the
//! picture.
//!
//! Run with: cargo run --release --example profile_decode

use num_complex::Complex;
use orecchiette_fpv_drone_analog_rs::ddc::StreamingDDC;
use orecchiette_fpv_drone_analog_rs::demod::{
    DEFAULT_DEEMPHASIS_TAU_S, Deemphasis, PllFmDemod, fm_demod_into,
};
use orecchiette_fpv_drone_analog_rs::synthetic::{SyntheticVideoConfig, TestPattern, generate_iq};
use orecchiette_fpv_drone_analog_rs::vbi::FieldParity;
use orecchiette_fpv_drone_analog_rs::video::FrameReconstructor;
use std::time::{Duration, Instant};

/// Live defaults, from `LiveArgs`.
const FM_DEVIATION: f32 = 5_000_000.0;
const LUMA_HEADROOM_HZ: f32 = 2_000_000.0;
const TEMPORAL_WINDOW: usize = 5;
/// The driver's block size.
const CHUNK: usize = 65_536;
/// `DemodKind::Auto` picks the PLL at or above this rate.
const PLL_AUTO_MIN_SAMPLE_RATE_HZ: u32 = 25_000_000;
/// Seconds of signal to push through per rate.
const TARGET_SECONDS: f64 = 1.0;

#[derive(Default)]
struct Stages {
    ddc: Duration,
    demod: Duration,
    deemph: Duration,
    buffer: Duration,
    reconstruct: Duration,
}

impl Stages {
    fn total(&self) -> Duration {
        self.ddc + self.demod + self.deemph + self.buffer + self.reconstruct
    }
}

fn main() {
    println!(
        "decode worker, per stage, NTSC, {} MHz deviation, {}-field temporal window\n\
         chunks of {CHUNK} samples, {TARGET_SECONDS:.1} s of signal per rate\n",
        FM_DEVIATION / 1e6,
        TEMPORAL_WINDOW
    );

    for rate in [15_360_000u32, 30_720_000, 61_440_000] {
        run(rate, 1);
    }

    println!(
        "same input, but the down-converter decimates as far as the channel allows\n\
         (the FIR then runs only on kept samples, and everything downstream sees the lower rate)\n"
    );
    let cutoff = FM_DEVIATION + LUMA_HEADROOM_HZ;
    for rate in [15_360_000u32, 30_720_000, 61_440_000] {
        run(rate, decode_decimation(rate, cutoff));
    }
}

/// The decimation rule the worker itself uses, so the two cannot drift.
/// Duplicated rather than imported because this crate is a binary.
fn decode_decimation(sample_rate: u32, ddc_cutoff_hz: f32) -> usize {
    const DECODE_FIR_TAPS: usize = 127;
    const FIR_TRANSITION_K: f32 = 2.33;
    if sample_rate == 0 || !ddc_cutoff_hz.is_finite() || ddc_cutoff_hz <= 0.0 {
        return 1;
    }
    let transition = FIR_TRANSITION_K * sample_rate as f32 / DECODE_FIR_TAPS as f32;
    ((sample_rate as f32 / (2.0 * ddc_cutoff_hz + transition)).floor() as usize).max(1)
}

fn run(rate: u32, decim: usize) {
    {
        let cfg = SyntheticVideoConfig {
            sample_rate: rate,
            is_pal: false,
            deviation_hz: FM_DEVIATION,
            pattern: TestPattern::Bars,
            start_field: FieldParity::First,
            noise_sigma: 0.05,
            dc_offset: 0.0,
        };
        // Four fields is ~67 ms; cycled below to reach the target.
        let iq = generate_iq(&cfg, 4, 0.0);

        // Exactly what the worker builds.
        let bandwidth_hz = (FM_DEVIATION + LUMA_HEADROOM_HZ) * 2.0;
        // Same expression the worker uses, written as a clamp.
        let cutoff = (bandwidth_hz / 2.0).clamp(FM_DEVIATION, FM_DEVIATION + LUMA_HEADROOM_HZ);
        let mut ddc = if decim > 1 {
            StreamingDDC::with_taps(0.0, rate, cutoff, 127)
        } else {
            StreamingDDC::new(0.0, rate, cutoff)
        };
        // Everything after the down-converter sees the decimated rate.
        let work_rate = rate / decim as u32;
        let use_pll = work_rate >= PLL_AUTO_MIN_SAMPLE_RATE_HZ;
        let mut pll = use_pll.then(|| PllFmDemod::new(work_rate, 1.0e6, FM_DEVIATION * 1.2));
        let mut deemph = Deemphasis::new(work_rate, DEFAULT_DEEMPHASIS_TAU_S);
        let mut reconstructor = FrameReconstructor::new(work_rate, false, FM_DEVIATION, false)
            .with_temporal_window(TEMPORAL_WINDOW);
        let mut frame_buf = vec![0u32; reconstructor.width * reconstructor.height];
        // Same guard the worker uses: a field plus blanking must be
        // buffered before asking. `reconstruct_frame_into` does three
        // full-slice passes before the length check that would reject a
        // short buffer, so polling it per chunk pays that for nothing.
        let min_samples_per_field = {
            let line_rate = 15_734.0f32; // NTSC throughout this profile
            ((work_rate as f32 / line_rate) * (240.0 + 22.0)) as usize
        };

        let mut shifted: Vec<Complex<f32>> = Vec::new();
        let mut scratch: Vec<f32> = Vec::new();
        let mut demod_buffer: Vec<f32> = Vec::new();
        let mut demod_start = 0usize;
        let mut carry: Option<Complex<f32>> = None;

        let mut st = Stages::default();
        let mut frames = 0u64;
        let want = (rate as f64 * TARGET_SECONDS) as usize;
        let mut done = 0usize;
        let mut pos = 0usize;

        while done < want {
            let end = (pos + CHUNK).min(iq.len());
            let chunk = &iq[pos..end];
            pos = if end == iq.len() { 0 } else { end };
            done += chunk.len();

            let t = Instant::now();
            shifted.clear();
            if !use_pll && let Some(p) = carry {
                shifted.push(p);
            }
            ddc.process_into_decimated(chunk, &mut shifted, decim);
            carry = shifted.last().copied();
            st.ddc += t.elapsed();

            let t = Instant::now();
            match pll.as_mut() {
                Some(p) => p.process_into(&shifted, &mut scratch),
                None => fm_demod_into(&shifted, &mut scratch),
            }
            st.demod += t.elapsed();

            let t = Instant::now();
            deemph.process_in_place(&mut scratch);
            st.deemph += t.elapsed();

            let t = Instant::now();
            demod_buffer.extend_from_slice(&scratch);
            st.buffer += t.elapsed();

            let t = Instant::now();
            while demod_buffer.len() - demod_start >= min_samples_per_field {
                let Some(consumed) = reconstructor
                    .reconstruct_frame_into(&demod_buffer[demod_start..], &mut frame_buf)
                else {
                    break;
                };
                frames += 1;
                demod_start += consumed;
                if demod_start > demod_buffer.len() / 2 {
                    demod_buffer.drain(..demod_start);
                    demod_start = 0;
                }
            }
            st.reconstruct += t.elapsed();
        }

        let secs = done as f64 / rate as f64;
        let tot = st.total().as_secs_f64() / secs;
        println!(
            "{:.2} MSPS{}   demodulator {}   {} frames   {:.2} CPU-seconds per second of signal  {}",
            rate as f64 / 1e6,
            if decim > 1 {
                format!(" /{decim} -> {:.2}", work_rate as f64 / 1e6)
            } else {
                String::new()
            },
            if use_pll { "PLL" } else { "discriminator" },
            frames,
            tot,
            if tot > 1.0 {
                "CANNOT KEEP UP"
            } else {
                "keeps up"
            }
        );
        for (name, d) in [
            ("down-convert + filter", st.ddc),
            ("demodulate", st.demod),
            ("deemphasis", st.deemph),
            ("buffer copy", st.buffer),
            ("reconstruct frames", st.reconstruct),
        ] {
            let per = d.as_secs_f64() / secs;
            println!("    {name:<22} {per:5.2} s/s  {:5.1}%", per / tot * 100.0);
        }
        println!();
    }
}
