//! Profile the production StreamingFpvDecoder without an SDR or UI.
//! Run with: cargo run --release --example profile_decode
//! Stage figures are elapsed wall seconds per second of input signal.
use clap::Parser;
use orecchiette_fpv_drone_analog_rs::decode::{
    DecodePlan, DecoderConfig, DemodulationMode, LUMA_HEADROOM_HZ, StreamingFpvDecoder,
};
use orecchiette_fpv_drone_analog_rs::demod::DEFAULT_DEEMPHASIS_TAU_S;
use orecchiette_fpv_drone_analog_rs::synthetic::{SyntheticVideoConfig, TestPattern, generate_iq};
use orecchiette_fpv_drone_analog_rs::timing::Standard;
use orecchiette_fpv_drone_analog_rs::vbi::FieldParity;
use std::num::NonZeroUsize;
use std::time::Instant;

#[derive(Parser)]
#[command(about = "Profile PAL and NTSC decoding without an SDR or UI")]
struct Args {
    /// Input rates in samples per second (comma-separated).
    #[arg(long, value_delimiter = ',', default_value = "15360000,30720000,61440000",
        value_parser = clap::value_parser!(u32).range(1..))]
    sample_rate: Vec<u32>,
    /// Signal duration per standard and rate; fixture generation is not timed.
    #[arg(long, default_value = "3", value_parser = positive_seconds)]
    seconds: f64,
    /// Samples per decoder push.
    #[arg(long, default_value = "65536")]
    chunk_size: NonZeroUsize,
    /// Temporal reconstruction window; 1 disables temporal repair.
    #[arg(long, default_value = "5")]
    temporal_window: NonZeroUsize,
}

fn positive_seconds(value: &str) -> Result<f64, String> {
    let seconds: f64 = value.parse().map_err(|_| "expected a number".to_owned())?;
    if !seconds.is_finite() || seconds <= 0.0 {
        return Err("seconds must be finite and greater than zero".into());
    }
    Ok(seconds)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    println!(
        "{}-{}, temporal window {}, chunks {} samples; wall s/s excludes fixture generation, SDR and UI",
        std::env::consts::OS,
        std::env::consts::ARCH,
        args.temporal_window,
        args.chunk_size,
    );
    for standard in [Standard::Ntsc, Standard::Pal] {
        for &rate in &args.sample_rate {
            let target_samples = (args.seconds * rate as f64).ceil();
            if target_samples >= u64::MAX as f64 {
                return Err("requested signal duration is too large".into());
            }
            let target_samples = target_samples as u64;
            let deviation = 5_000_000.0;
            let iq = generate_iq(
                &SyntheticVideoConfig {
                    sample_rate: rate,
                    is_pal: standard == Standard::Pal,
                    deviation_hz: deviation,
                    pattern: TestPattern::Bars,
                    start_field: FieldParity::First,
                    noise_sigma: 0.05,
                    dc_offset: 0.0,
                },
                8,
                0.0,
            );
            let plan = DecodePlan::new(
                rate,
                deviation,
                2.0 * (deviation + LUMA_HEADROOM_HZ),
                DemodulationMode::Auto,
            )?;
            let mut decoder = StreamingFpvDecoder::new(DecoderConfig {
                plan: plan.clone(),
                frequency_offset_hz: 0.0,
                standard,
                deemphasis_tau_s: DEFAULT_DEEMPHASIS_TAU_S,
                temporal_window: args.temporal_window.get(),
                debug: false,
            })?;
            decoder.set_profiling(true);
            let mut frame = vec![0; decoder.reconstructor().width * decoder.reconstructor().height];
            let mut done = 0;
            let mut fields = 0;
            let mut observed = 0;
            let started = Instant::now();
            while done < target_samples {
                // Replaying a fixture is a source discontinuity. Do not let a
                // synthetic phase jump masquerade as continuous timing evidence.
                for (index, chunk) in iq.chunks(args.chunk_size.get()).enumerate() {
                    let remaining = target_samples - done;
                    let chunk = &chunk[..remaining.min(chunk.len() as u64) as usize];
                    decoder.push_iq(chunk, done > 0 && index == 0);
                    done += chunk.len() as u64;
                    while let Some(timing) = decoder.next_field_into(&mut frame)? {
                        fields += 1;
                        observed += usize::from(timing.has_observed_timing_evidence);
                    }
                    if done == target_samples {
                        break;
                    }
                }
            }
            let wall_cost = started.elapsed().as_secs_f64() / (done as f64 / rate as f64);
            let signal_seconds = done as f64 / rate as f64;
            let stages = decoder.take_stage_timings();
            let cost = stages.total().as_secs_f64() / signal_seconds;
            println!(
                "{standard:?} {:.2} MSPS / {} -> {:.2} MSPS, {}, {fields} fields ({observed} observed), {cost:.3} stage s/s, {wall_cost:.3} wall s/s",
                rate as f64 / 1e6,
                plan.decimation(),
                plan.work_rate() as f64 / 1e6,
                if plan.use_pll() {
                    "PLL"
                } else {
                    "discriminator"
                }
            );
            for (label, time) in [
                ("DDC", stages.ddc),
                ("demod", stages.demod),
                ("deemphasis", stages.deemphasis),
                ("buffer", stages.buffering),
                ("reconstruct", stages.reconstruct),
            ] {
                println!(
                    "  {label:12} {:.3} s/s",
                    time.as_secs_f64() / signal_seconds
                );
            }
        }
    }
    Ok(())
}
