//! Profile the production StreamingFpvDecoder without an SDR or UI.
//! Run with: cargo run --release --example profile_decode
//! Stage figures are elapsed wall seconds per second of input signal.
use orecchiette_fpv_drone_analog_rs::decode::{
    DecodePlan, DecoderConfig, DemodulationMode, LUMA_HEADROOM_HZ, StreamingFpvDecoder,
};
use orecchiette_fpv_drone_analog_rs::demod::DEFAULT_DEEMPHASIS_TAU_S;
use orecchiette_fpv_drone_analog_rs::synthetic::{SyntheticVideoConfig, TestPattern, generate_iq};
use orecchiette_fpv_drone_analog_rs::timing::Standard;
use orecchiette_fpv_drone_analog_rs::vbi::FieldParity;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    for standard in [Standard::Ntsc, Standard::Pal] {
        for rate in [15_360_000, 30_720_000, 61_440_000] {
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
                temporal_window: 5,
                debug: false,
            })?;
            decoder.set_profiling(true);
            let mut frame = vec![0; decoder.reconstructor().width * decoder.reconstructor().height];
            let mut done = 0;
            let mut fields = 0;
            let mut observed = 0;
            while done < rate as usize {
                // Replaying a fixture is a source discontinuity. Do not let a
                // synthetic phase jump masquerade as continuous timing evidence.
                for (index, chunk) in iq.chunks(65_536).enumerate() {
                    decoder.push_iq(chunk, done > 0 && index == 0);
                    done += chunk.len();
                    while let Some(timing) = decoder.next_field_into(&mut frame)? {
                        fields += 1;
                        observed += usize::from(timing.has_observed_timing_evidence);
                    }
                }
            }
            let signal_seconds = done as f64 / rate as f64;
            let stages = decoder.take_stage_timings();
            let cost = stages.total().as_secs_f64() / signal_seconds;
            println!(
                "{standard:?} {:.2} MSPS / {} -> {:.2} MSPS, {}, {fields} fields ({observed} observed), {cost:.3} s/s",
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
