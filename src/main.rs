mod band_scan;

use band_scan::{BandScan, PanelMode, ScanWindow};
use clap::{Args as ClapArgs, Parser, Subcommand};
use minifb::{Key, Window, WindowOptions};
use num_complex::Complex;
use orecchiette_fpv_drone_analog_rs::ddc::StreamingDDC;
use orecchiette_fpv_drone_analog_rs::demod::{
    DEFAULT_DEEMPHASIS_TAU_S, Deemphasis, PllFmDemod, fm_demod, fm_demod_into,
};
use orecchiette_fpv_drone_analog_rs::detector::{
    AnalogFpvDetector, FpvDetector, ProbeEnergy, SpectralIntegrator,
};
use orecchiette_fpv_drone_analog_rs::levels::estimate_cnr_db;
use orecchiette_fpv_drone_analog_rs::lookup_channel_by_name;
use orecchiette_fpv_drone_analog_rs::types::SignalType;
use orecchiette_fpv_drone_analog_rs::video::{FrameReconstructor, detect_video_standard};
use orecchiette_sdr_source_rs::{DwellAdvice, SdrSource, SourceConfig};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::PathBuf;

use std::sync::{Arc, mpsc};
use std::thread;
use std::time::{Duration, Instant};

// ── CLI ────────────────────────────────────────────────────────────

#[derive(Parser, Debug)]
#[command(author, version, about = "Real-time Analog FPV Viewer")]
struct Cli {
    #[command(subcommand)]
    source: SourceCmd,
}

#[derive(Subcommand, Debug)]
enum SourceCmd {
    /// Replay a SigMF or raw IQ file.
    File(FileArgs),
    /// Live capture from an Ettus USRP B2xx.
    Usrp(UsrpArgs),
    /// Live capture from a HackRF One.
    Hackrf(HackrfArgs),
    /// Live capture from an Aaronia Spectran V6.
    #[cfg(feature = "aaronia")]
    #[command(subcommand)]
    Aaronia(AaroniaCmd),
}

// ── File subcommand ────────────────────────────────────────────────

#[derive(ClapArgs, Debug)]
struct FileArgs {
    /// Path to the SigMF `.sigmf-data` / `.sigmf-meta` or raw IQ file.
    input: PathBuf,
    #[arg(long)]
    sample_rate: Option<u32>,
    #[arg(long)]
    fm_deviation: Option<f32>,
    /// Video deemphasis time constant in seconds, undoing the VTX's
    /// pre-emphasis (which otherwise leaves high-frequency noise
    /// emphasized in the picture). 0 disables. Default 0.15 µs, from
    /// the analog crate's [`DEFAULT_DEEMPHASIS_TAU_S`], whose docs
    /// carry what each value costs the picture. Lengthen it if your
    /// transmitter's pre-emphasis is strong and the result looks
    /// noisy.
    #[arg(long, default_value_t = DEFAULT_DEEMPHASIS_TAU_S)]
    deemphasis_tau: f32,
    /// FM demodulator. 'auto' (default) uses the PLL only where its
    /// loop can track the deviation — a high enough decode rate *and* a
    /// deviation inside the loop's bandwidth. At analog FPV's ~5 MHz
    /// deviation that is never true at any rate the viewer decodes at,
    /// so 'auto' resolves to the discriminator: measured, the PLL is
    /// 9-25 dB worse there and also costs 6.49 dB at 4.2 MHz luma. A
    /// narrow deviation does not change that: it is exactly what lets
    /// the DDC decimate hard, and the loop's advantage falls with the
    /// decode rate (+9.9 dB at 25 MSPS, +1.8 at 6.25). 'pll' forces it
    /// and holds the decode rate up so the loop gets a rate it can use,
    /// at a CPU cost. Run with `--help` (long form, not `-h`) to see
    /// per-value details.
    #[arg(long, value_enum, ignore_case = true, default_value_t = DemodKind::Auto)]
    demod: DemodKind,
    /// Start with the neural temporal denoiser enabled (requires a
    /// build with `--features neural-vsr`). Toggle it live with `D`.
    #[arg(long)]
    denoise: bool,
    /// Path to the denoiser ONNX model.
    #[arg(long, default_value = "models/temporal_denoiser.onnx")]
    denoise_model: String,
    #[arg(long)]
    debug: bool,
    /// Number of fields kept in the temporal denoise / dropout-repair
    /// history (Phase A-E). 5 (default) gives ≈ +7 dB SNR on static
    /// scenes with ~83 ms latency; 1 disables temporal processing.
    /// Clamped to the range 1–8 (higher values allocate history the
    /// denoise never reads).
    #[arg(long, default_value_t = 5)]
    temporal_window: usize,
}

// ── Shared live-SDR flags ──────────────────────────────────────────

#[derive(ClapArgs, Debug, Clone)]
struct LiveArgs {
    /// FPV channel name (A1–A8, B1–B8, E1–E8, F1–F8, R1–R8, L1–L8,
    /// D1–D8) or raw frequency in Hz (e.g. 5865000000).
    ///
    /// Omit to auto-scan: the viewer captures wideband, detects all
    /// active analog FPV signals, and opens a window for each.
    #[arg(long)]
    channel: Option<String>,
    /// Force video standard instead of auto-detecting.
    #[arg(long, value_parser = parse_standard)]
    standard: Option<SignalType>,
    /// Override sample rate (Hz). Defaults per backend: USRP 25 MSPS,
    /// HackRF 20 MSPS (its USB 2.0 ceiling), Aaronia 61.44 MSPS (it runs 61.44 MHz over powers of two, down to 120 kSPS).
    #[arg(long)]
    sample_rate: Option<f64>,
    /// Which bands the auto-scan sweeps. '5.8' (default) covers
    /// 5.645-5.945 GHz; 'all' adds the 5.3 GHz L/D bands and 1.2 GHz,
    /// at proportionally more tunes per sweep. Ignored with --channel.
    #[arg(long, value_enum, ignore_case = true, default_value_t = ScanBands::Band58)]
    scan_bands: ScanBands,
    /// FM deviation (Hz). Defaults to 5 MHz for live SDR.
    #[arg(long, default_value_t = 5_000_000.0)]
    fm_deviation: f32,
    /// Video deemphasis time constant in seconds, undoing the VTX's
    /// pre-emphasis (which otherwise leaves high-frequency noise
    /// emphasized in the picture). 0 disables. Default 0.15 µs, from
    /// the analog crate's [`DEFAULT_DEEMPHASIS_TAU_S`], whose docs
    /// carry what each value costs the picture. Lengthen it if your
    /// transmitter's pre-emphasis is strong and the result looks
    /// noisy.
    #[arg(long, default_value_t = DEFAULT_DEEMPHASIS_TAU_S)]
    deemphasis_tau: f32,
    /// FM demodulator. 'auto' (default) uses the PLL only where its
    /// loop can track the deviation — a high enough decode rate *and* a
    /// deviation inside the loop's bandwidth. At analog FPV's ~5 MHz
    /// deviation that is never true at any rate the viewer decodes at,
    /// so 'auto' resolves to the discriminator: measured, the PLL is
    /// 9-25 dB worse there and also costs 6.49 dB at 4.2 MHz luma. A
    /// narrow deviation does not change that: it is exactly what lets
    /// the DDC decimate hard, and the loop's advantage falls with the
    /// decode rate (+9.9 dB at 25 MSPS, +1.8 at 6.25). 'pll' forces it
    /// and holds the decode rate up so the loop gets a rate it can use,
    /// at a CPU cost. Run with `--help` (long form, not `-h`) to see
    /// per-value details.
    #[arg(long, value_enum, ignore_case = true, default_value_t = DemodKind::Auto)]
    demod: DemodKind,
    /// Start with the neural temporal denoiser enabled (requires a
    /// build with `--features neural-vsr`). Toggle it live with `D`.
    /// It is a clear win on impaired links — dropouts especially — and
    /// a small cost on a clean signal, so it starts off by default.
    #[arg(long)]
    denoise: bool,
    /// Path to the denoiser ONNX model.
    #[arg(long, default_value = "models/temporal_denoiser.onnx")]
    denoise_model: String,
    /// Enable debug mode: saves startup frames 1-3 and steady-state
    /// frames 30-32 as PNGs, prints periodic metrics, and keeps
    /// running until quit (Q / Ctrl-C).
    #[arg(long)]
    debug: bool,
    /// Number of fields kept in the temporal denoise / dropout-repair
    /// history (Phase A-E). 5 (default) gives ≈ +7 dB SNR on static
    /// scenes with ~83 ms latency; 1 disables temporal processing.
    /// Clamped to the range 1–8 (higher values allocate history the
    /// denoise never reads).
    #[arg(long, default_value_t = 5)]
    temporal_window: usize,
}

/// Decode rate at/above which `DemodKind::Auto` selects the PLL, and
/// below which it selects the discriminator — the measured ~25 MSPS
/// crossover documented on [`DemodKind::Pll`] and in the library's
/// `weak_signal_sweep` example (below it the loop can't track a
/// typical ~5 MHz FPV deviation).
///
/// This is compared against the *decode* rate, which since the worker
/// started decimating (see [`decode_decimation`]) is normally well
/// below the capture rate — so on a wide live capture `Auto` now
/// resolves to the discriminator, which is both the cheaper and the
/// better choice down there. `--demod pll` caps the decimation instead
/// of being silently overruled.
/// Decode rate and demodulator chosen together, the way `run_live`
/// resolves them: decimate first, then ask about the rate that leaves.
///
/// Asking `use_pll` about the *capture* rate answers a question the
/// pipeline never poses. A narrow deviation is exactly what lets the
/// DDC decimate hard, and decimating is what removes the PLL's
/// advantage, so the two cannot be decided apart:
///
/// ```text
///   work rate    PLL gain over the discriminator (200-500 kHz dev)
///     25.00        +9.9 dB
///     15.36        +4.2 dB
///      8.33        +1.6 dB
///      6.25        +1.8 dB
/// ```
///
/// The loop's noise advantage comes from its bandwidth being narrow
/// *relative to the sample rate*, and `PllFmDemod` clamps `wn` to
/// 0.5 rad/sample — so at a low rate the loop is forced wide and the
/// advantage goes. At 25 MSPS input a 500 kHz deviation decimates by 4
/// to 6.25 MSPS, where the PLL is worth +1.8 dB and costs a
/// non-flat video response.
///
/// So `Auto` resolves to the discriminator in every configuration the
/// pipeline can reach, and that is the right answer rather than an
/// oversight. `--demod pll` is how to ask for it anyway; it holds the
/// decode rate up instead of decimating, which is the only way the
/// loop gets a rate it can use.
#[cfg(test)]
fn select_decode(capture_rate_hz: u32, fm_deviation: f32, demod: DemodKind) -> (u32, bool) {
    let cutoff = (fm_deviation + LUMA_HEADROOM_HZ).max(fm_deviation);
    let mut decim = decode_decimation(capture_rate_hz, cutoff);
    if matches!(demod, DemodKind::Pll) {
        let cap = (capture_rate_hz / PLL_AUTO_MIN_SAMPLE_RATE_HZ).max(1) as usize;
        decim = decim.min(cap);
    }
    let work_rate = capture_rate_hz / decim as u32;
    (work_rate, demod.use_pll(work_rate, fm_deviation))
}

/// Loop bandwidth the PLL is built with.
///
/// Also the deviation it can track: a loop cannot follow an excursion
/// faster than it can move. `PllFmDemod` clamps `wn` to 0.5 rad/sample,
/// so the *effective* bandwidth is `min(this, fs / 4π)`.
const PLL_LOOP_BW_HZ: f32 = 1.0e6;

const PLL_AUTO_MIN_SAMPLE_RATE_HZ: u32 = 25_000_000;

/// Which FM demodulator the decode worker runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
enum DemodKind {
    /// Automatically choose the discriminator or the PLL based on the
    /// decode rate after decimation (see
    /// `PLL_AUTO_MIN_SAMPLE_RATE_HZ` and [`decode_decimation`]) — the
    /// default. On a wide capture that resolves to the discriminator.
    Auto,
    /// Per-sample quadrature discriminator (`fm_demod`) — robust at
    /// every sample rate.
    #[value(name = "disc", alias = "discriminator")]
    Discriminator,
    /// PLL demodulator (`PllFmDemod`, 1 MHz loop): measured +6-17 dB
    /// demod SNR and ~one sigma-step deeper sync survival at ~25 MSPS
    /// decode rates and up, but UNUSABLE below ~20 MSPS (the loop
    /// can't track a 5 MHz deviation there).
    ///
    /// Because of that floor, asking for it stops the worker decimating
    /// past 25 MSPS, and says so when that costs it a stride. That
    /// costs the CPU the decimation would have saved, so on a
    /// 61.44 MSPS capture it may not keep up. On a capture already
    /// below 25 MSPS there is nothing to hold back and nothing to
    /// warn about — the loop simply runs where it was asked to, which
    /// below ~20 MSPS is where it cannot track.
    Pll,
}

impl DemodKind {
    /// Resolve to a concrete choice for the given decode rate in Hz —
    /// the rate *after* decimation, not the capture rate. `true`
    /// selects the PLL.
    /// Whether to demodulate with the PLL at this decode rate and
    /// deviation.
    ///
    /// `Auto` used to decide on the rate alone, and that cannot be
    /// right: what the loop has to survive is the *deviation*, and a
    /// loop too slow for it does not merely perform worse, it stops
    /// tracking. Measured at 25 MSPS against the discriminator, 1 MHz
    /// modulation, output SNR:
    ///
    /// ```text
    ///   deviation    PLL advantage
    ///     100 kHz        +12.1 dB
    ///       1 MHz        +12.1 dB
    ///       5 MHz    -24.9 .. -9.3 dB
    /// ```
    ///
    /// The benefit is real where the loop can follow — 12 dB is worth
    /// having — and it inverts completely at the 5 MHz deviation analog
    /// FPV actually uses, where the output is mostly tracking error.
    /// The same setting also costs video bandwidth: against the
    /// discriminator's flat response, the shipped 1 MHz loop reads
    /// +2.25 dB at 1 MHz and **-6.49 dB at 4.2 MHz**, so the top of the
    /// luma band is quietly lost.
    ///
    /// So the test is whether the loop can track. Its bandwidth is
    /// clamped at `wn <= 0.5 rad/sample`, i.e. `fs / 4π`, which at
    /// 25 MSPS is ~2 MHz — tracking 5 MHz would need a *decode* rate
    /// above 60 MSPS, and the decode path decimates below that by
    /// design.
    ///
    /// A narrow deviation does not rescue it either, and that is the
    /// part which is easy to get wrong: asking this function about the
    /// *capture* rate suggests the PLL is reachable, but the pipeline
    /// decimates first, and a narrow deviation is precisely what allows
    /// hard decimation. See [`select_decode`] — through the real
    /// ordering this resolves to the discriminator everywhere.
    ///
    /// `--demod pll` still forces it, and holds the decode rate up so
    /// the loop gets a rate it can use.
    fn use_pll(self, decode_rate_hz: u32, fm_deviation_hz: f32) -> bool {
        match self {
            DemodKind::Discriminator => false,
            DemodKind::Pll => true,
            DemodKind::Auto => {
                if decode_rate_hz < PLL_AUTO_MIN_SAMPLE_RATE_HZ {
                    return false;
                }
                // What the loop actually runs at, after its stability
                // clamp.
                let clamped_bw =
                    (decode_rate_hz as f32 / (4.0 * std::f32::consts::PI)).min(PLL_LOOP_BW_HZ);
                fm_deviation_hz > 0.0 && fm_deviation_hz <= clamped_bw
            }
        }
    }
}

// `--standard` stays a hand-written `value_parser` rather than
// `clap::ValueEnum`: `SignalType` is a foreign, `#[non_exhaustive]`
// type from `orecchiette_fpv_drone_analog_rs`, so the orphan rule
// blocks implementing `ValueEnum` for it here — unlike `DemodKind`,
// which is a local type and derives it directly above.
fn parse_standard(s: &str) -> Result<SignalType, String> {
    match s.to_lowercase().as_str() {
        "pal" => Ok(SignalType::AnalogVideoPal),
        "ntsc" => Ok(SignalType::AnalogVideoNtsc),
        _ => Err(format!(
            "unknown standard '{}'; expected 'pal' or 'ntsc'",
            s
        )),
    }
}

/// How many capture packets per hop get fed to the detector before the
/// rest of the dwell is drained.
///
/// `detect_from_iq_integrated` accumulates magnitude spectra per
/// frequency, and a single 65,536-sample packet is below the length at
/// which the sweep detects anything: with one packet per hop the first
/// sweep finds nothing and detection lands only on the second visit,
/// once the integrator holds two batches. Consecutive packets from one
/// hop show where it turns over (`--` = no detection, else confidence):
///
/// ```text
///   25.00 MSPS  packets 1..8:  --  0.80 0.80 0.80 0.80 0.80  --  0.80
///   61.44 MSPS  packets 1..8:  --  0.80 0.80 0.80 0.80 0.80 0.80 0.80
/// ```
///
/// Two gives first-visit detection, and on a *clean* signal beyond two
/// measures no better: confidence plateaus at 0.80 and never promotes.
/// The table above is exactly that case, which is why it reads as a
/// plateau.
///
/// It does not hold under noise. Resetting the integrator between trials
/// and varying capture position, synthetic centred NTSC gives:
///
/// ```text
///                                  within 2 packets   within 8
///   25 MSPS,    component σ 1.0          7/8             8/8
///   25 MSPS,    component σ 1.5          2/8             8/8
///   61.44 MSPS, component σ 1.0          5/8             8/8
///   61.44 MSPS, component σ 1.5          4/8             4/8
/// ```
///
/// So two packets is the right *fast pass* and the wrong *verdict*: at
/// σ 1.5 it finds a quarter of what eight finds. Hops that show energy
/// So two packets is the right fast pass at a wide capture rate and the
/// wrong verdict at a narrow one, where a packet is four times the
/// signal for a third of the cost — see [`detect_packets_per_hop`].
///
/// The cost is CPU, not time on air: two packets is ~5.2 ms of signal at
/// 25 MSPS but ~18 ms of detection work, so a hop becomes CPU-bound
/// rather than dwell-bound.
const DETECT_PACKETS_PER_HOP: usize = 2;

/// The lock check's running state, factored out so the rule is directly
/// unit-testable rather than buried in the reader closure.
///
/// Three outcomes per check, and the distinction that matters is
/// between the middle two:
///
/// - **Held**: our carrier is there and fields are coming out.
/// - **Provisional**: our carrier is there and no field has come out
///   since the last check. Acquisition looks like this; so does a
///   horizontal-sync train with no vertical sync, which scores 0.8
///   forever and reconstructs nothing. Bounded rather than trusted.
/// - **Lost**: nothing, or only hits that are not ours.
#[derive(Default)]
struct LockState {
    empty_checks: u32,
    provisional_checks: u32,
}

impl LockState {
    /// Fold one check in. `true` means release the channel.
    fn observe(&mut self, on_carrier: bool, decoding: bool) -> bool {
        match (on_carrier, decoding) {
            (true, true) => {
                self.empty_checks = 0;
                self.provisional_checks = 0;
                false
            }
            (true, false) => {
                // Our carrier is there. That answers the "is it still
                // ours" question regardless of whether a field came out,
                // so the miss counter resets — leaving it standing made
                // miss -> recovery -> miss release the channel on two
                // misses that were never consecutive, which is the one
                // thing `LOCK_EMPTY_CHECKS` is counting.
                self.empty_checks = 0;
                self.provisional_checks += 1;
                self.provisional_checks >= LOCK_PROVISIONAL_CHECKS
            }
            _ => {
                self.empty_checks += 1;
                self.empty_checks >= LOCK_EMPTY_CHECKS
            }
        }
    }
}

/// Consecutive checks finding nothing of ours before the channel is
/// released.
///
/// Strictly consecutive: any check that sees our carrier clears this,
/// including one that saw no decoded field. That case is counted by
/// [`LOCK_PROVISIONAL_CHECKS`] instead — "the carrier is gone" and "the
/// carrier is here but undecodable" are different failures and are
/// timed separately.
///
/// Two, not four: the lock integrator holds ~4 checks of history, so an
/// empty result already means several consecutive looks found nothing.
/// Stacking a 4-check threshold on top made a dead channel linger for
/// up to 8 checks.
const LOCK_EMPTY_CHECKS: u32 = 2;

/// How far a continuing detection may sit from the channel we tuned to
/// and still count as *this* signal.
///
/// The lock check ran on whatever the capture held, so a transmitter
/// elsewhere in the 49 MHz span kept the tuned channel's lock alive
/// indefinitely. Sized like [`CHANNEL_SNAP_TOLERANCE_MHZ`]: past this,
/// a hit is more likely a different channel than an off-tune of ours.
const LOCK_CARRIER_TOLERANCE_HZ: f64 = 10e6;

/// Lock checks a carrier may go on being detected without a single
/// field coming out before we let it go.
///
/// A detection is not decodable video. A horizontal-sync pulse train
/// with no vertical sync scores 0.8 on the sweep detector and
/// reconstructs no frame at all, so "there is a hit" held the viewer on
/// a picture it could never draw. Fields reconstructing is the
/// confirmation; this is how long acquisition is allowed to take before
/// the absence of one is treated as an answer.
///
/// Checks run about twice a second, so eight is ~4 s — long against
/// standard classification plus a first field, short against staring at
/// an undecodable carrier.
const LOCK_PROVISIONAL_CHECKS: u32 = 8;

/// Packets fed to the detector on a hop at a *narrow* capture rate.
///
/// A packet is a fixed 65,536 samples, so what it is worth depends
/// entirely on the rate: 1.07 ms of signal at 61.44 MSPS against 4.27 ms
/// at 15.36, and it costs 6.41 ms of detector against 2.49. Narrow
/// captures are the case where more packets are both more useful and
/// cheaper, and six of them is 14.9 ms of CPU inside a 25 ms dwell —
/// where six at 61.44 MSPS would be 38.5 ms and miss the hop entirely.
///
/// Six because integration needs the room. Measured against a real A1
/// transmitter attenuated toward its cliff, a 15.36 MSPS capture that
/// never confirms in two packets first detects on the fourth; the
/// evidence is not present earlier, it is *built* by accumulating
/// spectra. Two packets cannot see it however cleverly they are gated.
const DETECT_PACKETS_NARROW: usize = 6;

/// Above this rate a packet is too little signal and too much detector
/// to spend [`DETECT_PACKETS_NARROW`] of them on.
const NARROW_CAPTURE_MAX_RATE_HZ: f64 = 30_720_000.0;

/// What one sweep may spend per hop.
///
/// Bundled because they are one decision: packets cost detector time,
/// and the dwell has to be long enough to pay for them.
#[derive(Clone, Copy)]
struct SweepBudget {
    dwell: Duration,
    packets_per_hop: usize,
}

/// One sweep in this many is a *sensitive* sweep.
///
/// The fast pass reads two packets at 61.44 MSPS, which is 2.1 ms of
/// signal, and some signals simply are not detectable in that: a noisy
/// NTSC fixture at that rate first detects on packet **3**, so no
/// per-hop decision taken at packet 2 can reach it — the evidence does
/// not exist yet, it is built by accumulating spectra.
///
/// An earlier attempt gated extra packets on a probe's energy margin.
/// That could not work: measured against a real transmitter attenuated
/// toward its cliff, the margin falls at the same rate as detectability,
/// so the gate shut exactly where integration would have paid. Nor does
/// sub-threshold confidence help — on that ladder it is only ever 0.00,
/// 0.60 or 0.80, never inside the band a gate could use.
///
/// So nothing is predicted. Most sweeps stay fast, and every fourth one
/// spends the sensitive budget on every hop. A signal that needs
/// integration is found within four sweeps instead of never, and the
/// average sweep costs about 15% more rather than 60%.
#[cfg(feature = "aaronia")]
const AARONIA_SENSITIVE_EVERY: u32 = 4;

/// Dwell for a sensitive sweep: long enough to pay for
/// [`DETECT_PACKETS_NARROW`] passes at the scan's own rate, where the
/// detector costs ~6.4 ms a packet.
#[cfg(feature = "aaronia")]
const AARONIA_SENSITIVE_DWELL: Duration = Duration::from_millis(60);

/// How many packets one hop feeds the detector at `sample_rate`.
fn detect_packets_per_hop(sample_rate_hz: f64) -> usize {
    if sample_rate_hz > 0.0 && sample_rate_hz <= NARROW_CAPTURE_MAX_RATE_HZ {
        DETECT_PACKETS_NARROW
    } else {
        DETECT_PACKETS_PER_HOP
    }
}

impl SweepBudget {
    /// The budget for sweep number `n` at `sample_rate_hz`.
    ///
    /// Aaronia-only: the cadence is argued from that backend's retune
    /// and detector costs, and says nothing about a synthesiser that
    /// tunes in under a millisecond.
    ///
    /// A narrow capture already reads the sensitive number of packets
    /// every sweep — there a packet is four times the signal for a
    /// third of the cost — so its budget never changes and the cadence
    /// costs it nothing.
    #[cfg(feature = "aaronia")]
    fn for_sweep(n: u32, sample_rate_hz: f64) -> Self {
        let fast = detect_packets_per_hop(sample_rate_hz);
        let sensitive = n % AARONIA_SENSITIVE_EVERY == AARONIA_SENSITIVE_EVERY - 1
            && fast < DETECT_PACKETS_NARROW;
        if sensitive {
            Self {
                dwell: AARONIA_SENSITIVE_DWELL,
                packets_per_hop: DETECT_PACKETS_NARROW,
            }
        } else {
            Self {
                dwell: AARONIA_SCAN_DWELL,
                packets_per_hop: fast,
            }
        }
    }

    /// A flat budget for a backend with its own tuning economics.
    fn flat(dwell: Duration, packets_per_hop: usize) -> Self {
        Self {
            dwell,
            packets_per_hop,
        }
    }
}

/// The rate to capture a *locked* channel at, once the sweep has found
/// one.
///
/// Scanning wants width: 61.44 MSPS covers 49.2 MHz of spectrum per
/// tune, which is the whole point of hopping so few times. Decoding one
/// channel needs only the channel, and the RTSA HTTP link carries
/// 245 MB/s to deliver a signal that fits in a quarter of it.
///
/// This picks the **lowest supported rate whose usable span still holds
/// the channel**, which is not the same as the rate the decoder would
/// have decimated to. Two things make the difference matter:
///
/// - `decode_decimation` is a floor division, not a power of two. At
///   6 MHz deviation it answers 3, and 61.44/3 is 20.48 MSPS — not a
///   rate the hardware runs. Snapping that to the nearest supported
///   rate lands on 15.36 MSPS, *below* the 17.1 MSPS its own ±8 MHz
///   cutoff requires, and the capture aliases onto itself. An earlier
///   version of this shipped that bug.
/// - The hardware's usable span is narrower than its sample rate
///   ([`USABLE_SPAN_FRACTION`]). Capturing at rate `r` is therefore
///   *not* the same as capturing wide and decimating to `r` in
///   software: the device's own filter shapes the band, so the channel
///   has to fit the usable part, not merely Nyquist.
///
/// So the requirement is `USABLE_SPAN_FRACTION * rate >= 2 * cutoff`,
/// checked against each supported rate from the bottom up. At the
/// default 5 MHz deviation that gives 30.72 MSPS — half the network and
/// half the DDC rather than a quarter, but a rate the signal actually
/// fits inside.
///
/// Two cases keep the scan rate:
///
/// - `--demod pll`. The loop cannot track a 5 MHz deviation much below
///   20 MSPS, so `run_live` caps decimation for it, and the cap is not
///   generally a rate the hardware runs. Forcing PLL already costs the
///   CPU decimation would have saved; this does not also hand it a rate
///   it cannot use.
/// - An explicit `--sample-rate`. That is the operator naming a rate,
///   not a default to optimise.
///
/// The cutoff is the widest `run_live` can choose — `bandwidth_hz / 2`
/// is clamped to `fm_deviation + LUMA_HEADROOM_HZ` — so the rate is
/// safe for whatever bandwidth the detector reports, which is not known
/// until after the lock.
#[cfg(feature = "aaronia")]
fn locked_capture_rate_hz(scan_rate_hz: f64, fm_deviation: f32, demod: DemodKind) -> f64 {
    if matches!(demod, DemodKind::Pll) || !scan_rate_hz.is_finite() || scan_rate_hz <= 0.0 {
        return scan_rate_hz;
    }
    let widest_cutoff = (fm_deviation + LUMA_HEADROOM_HZ) as f64;
    if !widest_cutoff.is_finite() || widest_cutoff <= 0.0 {
        return scan_rate_hz;
    }
    // Spectrum the channel occupies, expressed as the sample rate that
    // holds it inside the part of the capture the hardware delivers
    // flat.
    let needed_rate = 2.0 * widest_cutoff / USABLE_SPAN_FRACTION;

    // Walk down the supported grid (`scan_rate / 2^n`, which is what
    // `nearest_iq_sample_rate` quantises to) and keep the lowest rate
    // that still clears the requirement.
    let mut chosen = scan_rate_hz;
    let mut candidate = scan_rate_hz / 2.0;
    while candidate >= needed_rate {
        let supported = sdr_aaronia_rs::nearest_iq_sample_rate(candidate);
        if (supported - candidate).abs() > 1.0 || supported < needed_rate {
            break;
        }
        chosen = supported;
        candidate /= 2.0;
    }
    chosen
}

/// Which FPV bands the auto-scan covers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
enum ScanBands {
    /// The 5.8 GHz band only (5.645–5.945 GHz) — A/B/E/F/R. The default.
    #[value(name = "5.8", alias = "58")]
    Band58,
    /// Every band in the channel table, 1.080–5.945 GHz: 5.8 GHz plus the
    /// 5.3 GHz L/D bands and 1.2 GHz. Costs more tunes per sweep, so the
    /// revisit interval on any one channel goes up proportionally.
    All,
}

impl ScanBands {
    /// Short human-readable name for startup logging.
    fn label(self) -> &'static str {
        match self {
            ScanBands::Band58 => "5.8 GHz",
            ScanBands::All => "all bands",
        }
    }

    /// Channel centre frequencies this setting covers, in Hz.
    fn channel_freqs_hz(self) -> Vec<f64> {
        orecchiette_fpv_drone_analog_rs::bands::get_all_channels()
            .iter()
            .map(|c| c.frequency_hz as f64)
            .filter(|f| match self {
                ScanBands::Band58 => (5_645e6..=5_945e6).contains(f),
                ScanBands::All => true,
            })
            .collect()
    }

    /// The same channels with their names and band letters, for the
    /// band panel.
    fn channels(self) -> Vec<band_scan::Channel> {
        orecchiette_fpv_drone_analog_rs::bands::get_all_channels()
            .into_iter()
            .filter(|c| match self {
                ScanBands::Band58 => (5_645e6..=5_945e6).contains(&(c.frequency_hz as f64)),
                ScanBands::All => true,
            })
            .map(|c| {
                let band = band_scan::band_letter(c.band);
                band_scan::Channel {
                    name: format!("{}{}", band, c.channel),
                    band,
                    hz: c.frequency_hz as f64,
                }
            })
            .collect()
    }
}

/// Fraction of the sample rate across which the detector still places a
/// carrier accurately, so a channel centred anywhere inside it is both
/// found and reported at the right frequency.
///
/// Measured against the sliding-DDC sweep rather than assumed from the
/// anti-alias filter. Feeding the viewer's real scan path (65,536-sample
/// packets through `detect_from_iq_integrated`) a clean synthetic NTSC
/// carrier at increasing offsets from the tuned centre, at 61.44 MSPS:
///
/// ```text
///   offset  +0    +5    +10   +15   +20   +25   +30 MHz
///   error  -0.7  -0.7  -0.7  -0.7  -0.7  -0.7  -55.7
/// ```
///
/// The constant −0.7 MHz is probe-grid quantisation, absorbed downstream
/// by `CHANNEL_SNAP_TOLERANCE_MHZ`. Localisation holds all the way to
/// ±25 MHz (0.81 × the ±30.72 MHz capture half-width) and only collapses
/// at the very edge. Detection *rate* starts thinning around the same
/// point — at 25 MSPS a carrier at 0.8 × half-width scored 7/12 sweeps
/// against 10–12/12 further in. 0.8 keeps planning inside the region
/// where both hold.
const USABLE_SPAN_FRACTION: f64 = 0.8;

/// Plan the fewest tune centres that put every channel in `channels_hz`
/// inside some capture window of width `span_hz`.
///
/// Greedy interval cover, the same shape as `orecchiette`'s
/// `plan_tune_centers`: walk the sorted channels, anchor on the leftmost
/// one not yet covered, absorb every channel within `span_hz` of it, and
/// emit the **midpoint of the group actually absorbed** rather than the
/// anchor, which holds the absorbed channels as far from the window
/// edges — where localisation degrades — as the group allows. Anchoring
/// left and taking as much as fits is optimal in the number of windows
/// for covering points on a line with a fixed width.
///
/// The coverage test is on channel *centres*, not whole channel widths.
/// `orecchiette` subtracts a 20 MHz allowance from the reach because it
/// needs a whole DJI channel inside one window to demodulate it. Here the
/// window only has to be good enough to *detect and localise* a carrier,
/// which the measurements behind [`USABLE_SPAN_FRACTION`] show needs the
/// carrier centre inside the span and nothing more. Charging every narrow
/// analog channel a 20 MHz guard would roughly halve the reach at
/// 25 MSPS for no gain.
fn plan_tune_centers(channels_hz: &[f64], span_hz: f64) -> Vec<f64> {
    let mut sorted: Vec<f64> = channels_hz
        .iter()
        .copied()
        .filter(|f| f.is_finite())
        .collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    sorted.dedup();

    let reach = span_hz.max(0.0);
    let mut centers = Vec::new();
    let mut i = 0;
    while i < sorted.len() {
        let first = sorted[i];
        let mut last = first;
        let mut j = i;
        while j < sorted.len() && sorted[j] - first <= reach {
            last = sorted[j];
            j += 1;
        }
        // `j > i` always holds: the anchor satisfies the window test
        // (distance 0), so the inner loop advances at least once and the
        // outer loop cannot spin. This guards only a NaN-poisoned
        // comparison ordering escaping the filter above.
        if j == i {
            j = i + 1;
        }
        centers.push((first + last) * 0.5);
        i = j;
    }
    centers
}

/// Build the hop list for the auto-scan loop, covering every channel in
/// `bands` with as few tunes as the SDR's instantaneous bandwidth allows.
///
/// Planning against the channel list rather than stepping uniformly
/// spends tunes only where channels are:
///
/// ```text
///                        5.8 GHz    all bands
///                    (40 channels) (137 channels)
///     20.00 MSPS            16          101
///     25.00 MSPS            11           93
///     40.00 MSPS             8           56
///     61.44 MSPS             6           48
/// ```
///
/// Revisit interval is `tunes × (retune + dwell)`, so those tune counts
/// are directly how long a signal can hide between looks.
///
/// Ordered low→high so synthesisers retune monotonically, which minimises
/// PLL re-lock time on most parts.
fn build_scan_hops(sample_rate: f64, bands: ScanBands) -> Vec<f64> {
    let channels: Vec<f64> = bands.channel_freqs_hz();
    plan_tune_centers(&channels, sample_rate * USABLE_SPAN_FRACTION)
}

/// Resolve --channel to a centre frequency in Hz.
fn resolve_channel(ch: &str) -> anyhow::Result<f64> {
    // Try channel name first
    if let Some(freq) = lookup_channel_by_name(ch) {
        return Ok(freq as f64);
    }
    // Try raw frequency
    if let Ok(freq) = ch.parse::<f64>()
        && freq > 1e6
    {
        return Ok(freq);
    }
    anyhow::bail!(
        "unrecognised channel '{}'; expected A1–A8, B1–B8, E1–E8, F1–F8, R1–R8, L1–L8, D1–D8, or a frequency in Hz",
        ch
    );
}

// ── USRP subcommand ────────────────────────────────────────────────

#[derive(ClapArgs, Debug)]
struct UsrpArgs {
    #[command(flatten)]
    live: LiveArgs,
    /// UHD device args (e.g. "type=b200").
    #[arg(long, default_value = "")]
    args: String,
    /// RX gain in dB.
    #[arg(long, default_value_t = 40.0)]
    gain: f64,
    /// RX antenna port.
    #[arg(long, default_value = "RX2")]
    antenna: String,
}

// ── HackRF subcommand ──────────────────────────────────────────────

#[derive(ClapArgs, Debug)]
struct HackrfArgs {
    #[command(flatten)]
    live: LiveArgs,
    /// LNA (IF) gain in dB, 0–40 in 8 dB steps.
    #[arg(long, default_value_t = 16)]
    lna_gain: u16,
    /// VGA (baseband) gain in dB, 0–62 in 2 dB steps.
    #[arg(long, default_value_t = 20)]
    vga_gain: u16,
    /// Enable the front-end +14 dB RF amplifier (off by default; it
    /// overloads easily on strong ambient traffic).
    #[arg(long, default_value_t = false)]
    amp: bool,
    /// Enable the bias-tee (antenna-port DC power) for active antennas.
    #[arg(long, default_value_t = false)]
    bias_tee: bool,
}

// ── Aaronia subcommands ────────────────────────────────────────────

#[cfg(feature = "aaronia")]
#[derive(Subcommand, Debug)]
enum AaroniaCmd {
    /// Stream from an RTSA HTTP server block.
    Http(AaroniaHttpArgs),
    /// Stream via the native AARTSAAPI SDK.
    Sdk(AaroniaSdkArgs),
}

#[cfg(feature = "aaronia")]
#[derive(ClapArgs, Debug)]
struct AaroniaHttpArgs {
    /// Base URL of the RTSA HTTP server.
    #[arg(value_name = "URL")]
    url: String,
    #[command(flatten)]
    live: LiveArgs,
    /// Reference level (dBm).
    #[arg(long, default_value_t = -25.0)]
    ref_level: f64,
    /// IQ wire format for the HTTP stream. f16 (default) halves the
    /// network rate against f32 — 61.44 MSPS is ~246 MB/s in f16 versus
    /// ~491 in f32, and the fatter stream is prone to stalling ("no
    /// samples arrived within the read deadline"). Video tolerates the
    /// precision loss: an 11-bit significand leaves ~66 dB of SNR
    /// headroom, far beyond an analog FPV link.
    #[arg(long, value_enum, ignore_case = true, default_value_t = AaroniaStreamFormat::F16)]
    stream_format: AaroniaStreamFormat,
}

/// IQ wire format for the RTSA HTTP stream (`--stream-format`).
#[cfg(feature = "aaronia")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
enum AaroniaStreamFormat {
    /// IEEE half precision, 4 bytes per IQ pair.
    F16,
    /// Full single precision, 8 bytes per IQ pair.
    F32,
    /// 16-bit integers with a scale factor, 4 bytes per IQ pair.
    Int16,
}

#[cfg(feature = "aaronia")]
impl AaroniaStreamFormat {
    fn to_driver(self) -> sdr_aaronia_rs::http_streaming::StreamFormat {
        use sdr_aaronia_rs::http_streaming::StreamFormat;
        match self {
            AaroniaStreamFormat::F16 => StreamFormat::Float16,
            AaroniaStreamFormat::F32 => StreamFormat::Float32,
            AaroniaStreamFormat::Int16 => StreamFormat::Int16,
        }
    }
}

#[cfg(feature = "aaronia")]
#[derive(ClapArgs, Debug)]
struct AaroniaSdkArgs {
    #[command(flatten)]
    live: LiveArgs,
    /// Device serial number.
    #[arg(long)]
    serial: Option<String>,
    /// Reference level (dBm).
    #[arg(long, default_value_t = -25.0)]
    ref_level: f64,
}

// ── Constants ──────────────────────────────────────────────────────

/// DDC cutoff is the FM deviation plus enough chroma headroom for PAL.
const LUMA_HEADROOM_HZ: f32 = 2_000_000.0;

/// Taps the decode down-converter uses when it decimates.
///
/// The 63-tap default is not sharp enough to decimate a wide capture.
/// Decimating by N folds everything above the new Nyquist back onto the
/// signal, and the frequency landing on the passband edge is
/// `work_rate - cutoff` — 8.36 MHz out, for a 61.44 MSPS capture at 4x.
/// That is where a neighbouring VTX sits, and 63 taps attenuate it by
/// only 24 dB. 127 taps give 75 dB. The longer filter still costs less
/// than today because the convolution runs only on the samples the
/// stride keeps: 127 taps on a quarter of them is half the work of 63
/// taps on all of them. Pinned by the analog crate's
/// `a_decimated_down_conversion_still_decodes`, which puts a
/// transmitter 20 dB up at that offset and watches sync quality fall to
/// 0.01 on the default filter.
const DECODE_FIR_TAPS: usize = 127;

/// Width of the down-converter's transition band, as a fraction of the
/// input rate times the tap count: `transition_hz ≈ K · fs / taps`.
/// Fitted to the crate's Blackman design between its −6 dB and −50 dB
/// points.
const FIR_TRANSITION_K: f32 = 2.33;

/// How far the decode path may decimate a capture and stay alias-free.
///
/// The video occupies `2 · cutoff` of the capture — about 14 MHz for a
/// 5 MHz-deviation channel — so a 61.44 MSPS capture spends three
/// quarters of every stage on spectrum the down-converter's filter
/// already emptied. Decimating there lets the demodulator, deemphasis
/// and frame reconstructor all run at the lower rate: 3.93 against 0.61
/// CPU-seconds per second of signal on one core of an Apple M4
/// (`examples/profile_decode.rs`). Above 1.0 the worker drops chunks.
///
/// The limit is aliasing: energy at `work_rate - cutoff` folds onto the
/// passband edge, so that frequency has to clear the filter's stopband
/// edge, one transition band above the cutoff.
///
/// It cannot be a fixed target rate. File playback defaults to a 17 MHz
/// deviation, so a "always decimate to 15.36 MSPS" rule would fold a
/// 19 MHz-wide channel onto itself.
fn decode_decimation(sample_rate: u32, ddc_cutoff_hz: f32) -> usize {
    if sample_rate == 0 || !ddc_cutoff_hz.is_finite() || ddc_cutoff_hz <= 0.0 {
        return 1;
    }
    let transition = FIR_TRANSITION_K * sample_rate as f32 / DECODE_FIR_TAPS as f32;
    let min_work_rate = 2.0 * ddc_cutoff_hz + transition;
    if min_work_rate <= 0.0 {
        return 1;
    }
    ((sample_rate as f32 / min_work_rate).floor() as usize).max(1)
}

/// Default centre of the 5.8 GHz FPV band for wideband scanning.
/// (Only the Aaronia backend scans a single fixed span; USRP/HackRF
/// hop through `build_scan_hops` instead.)
#[cfg(feature = "aaronia")]
const SCAN_CENTER_HZ: f64 = 5_800_000_000.0;

/// Dwell per hop for the Aaronia sweep, on top of the driver's own
/// 20 ms post-retune drain (`RETUNE_SETTLE` in `sdr-aaronia-rs`, taken
/// before the dwell clock starts — a hop costs settle + dwell).
///
/// A retune must be complete before any packet is attributed to the new
/// centre, or the sweep reports a real signal at a frequency it was
/// never received on. Since `sdr-aaronia-rs` v0.11.0 the driver stamps
/// each `IqPacket` with the frequency the RTSA packet header reports,
/// and drops any buffer whose capture frequency does not match the
/// commanded channel — a stale packet is rejected rather than
/// mislabelled. The drain is what keeps that rejection cheap; it is no
/// longer what makes it correct.
///
/// Measured against a Spectran V6 ECO over RTSA HTTP at 61.44 MSPS, by
/// alternating between 869 MHz and 3500 MHz (a 23 dB power step) and
/// timing the config PUT against the arrival of samples that actually
/// carry the new centre, 24 hops:
///
/// ```text
///   config PUT round-trip        median  4.8 ms   p95  6.0   max  6.1
///   signal carries new centre    median 23.5 ms   p95 26.8   max 38.9
///   two usable packets in hand   median 12.7 ms   p95 27.4   max 30.9
/// ```
///
/// The header frequency and the signal transition agree to within ~1 ms,
/// so the settle is genuinely that short — the previous 150-300 ms
/// figure was not measured this way.
///
/// 25 ms is the floor a *quiet* hop pays. The fast pass reads
/// [`DETECT_PACKETS_PER_HOP`] packets — 2.1 ms of signal at 61.44 MSPS
/// — and everything after that was drained and thrown away, so the
/// dwell was mostly paying for nothing. Measured against a live A1 VTX,
/// dropping 100 ms to 25 ms takes a hop from 122.3 ms to 49.3 ms median
/// (a 48-tune sweep, 5.9 s to ~2.4 s) with detection unchanged at 5/5.
///
/// A hop that shows energy is not held to this floor — see
/// [`AARONIA_SCAN_DWELL_MAX`]. Cutting the floor without that would
/// trade away weak-signal sensitivity that was never measured; the two
/// belong together.
#[cfg(feature = "aaronia")]
const AARONIA_SCAN_DWELL: Duration = Duration::from_millis(25);

/// Default Aaronia span when no --sample-rate is given (max complex span for 92.16 MHz clock).
#[cfg(feature = "aaronia")]
const AARONIA_DEFAULT_RATE_HZ: f64 = 61_440_000.0;

/// The one frame the display will show next.
///
/// A bounded channel cannot express "latest wins": the producer can
/// only add to the back, so when it is full the choice is to block (the
/// display paces the decoder, and dropped IQ follows) or to discard the
/// frame just produced — which keeps the *stale* ones queued. Sixty
/// frames produced during a stall left frame 1 as the newest available.
///
/// A slot the producer overwrites has neither problem. It also hands
/// back whatever it displaced, so the buffer is reused rather than
/// allocated, and the queue depth cannot add latency because there is
/// no queue.
type FrameSlot = Arc<std::sync::Mutex<Option<Vec<u32>>>>;

/// Lock a [`FrameSlot`], ignoring poisoning.
///
/// A panicking producer or consumer says nothing about whether the
/// pixels in the slot are usable, and refusing to show video because a
/// previous frame's thread died is worse than showing it.
fn lock_slot(slot: &FrameSlot) -> std::sync::MutexGuard<'_, Option<Vec<u32>>> {
    slot.lock().unwrap_or_else(|e| e.into_inner())
}

/// One capture chunk on its way to the decoder, and whether the signal
/// reaching it runs on from the last one.
///
/// The decoder carries state across chunk boundaries — a FIR delay
/// line, a carried IQ sample, a field history — all of which assume the
/// samples either side are adjacent. Two things break that, and neither
/// used to reach the decoder: a source overrun (the flag was read once
/// during standard classification and then dropped), and the reader
/// discarding a chunk because the worker was behind.
///
/// Carrying it with the chunk rather than beside it means the decoder
/// cannot act on the wrong one when it falls behind.
struct IqChunk {
    samples: Arc<orecchiette_sdr_source_rs::PooledIqBuffer>,
    /// The stream broke immediately before this chunk.
    discontinuous: bool,
}

// ── No-op dwell advice (single-channel, no hopping) ────────────────

struct NoOpDwell;
impl DwellAdvice for NoOpDwell {
    fn latest_signal_at(&self, _: u64) -> Option<Instant> {
        None
    }
}

// ── Main ───────────────────────────────────────────────────────────

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let cli = Cli::parse();
    match cli.source {
        SourceCmd::File(f) => run_file(f),
        SourceCmd::Usrp(u) => run_live_usrp(u),
        SourceCmd::Hackrf(h) => run_live_hackrf(h),
        #[cfg(feature = "aaronia")]
        SourceCmd::Aaronia(a) => run_live_aaronia(a),
    }
}

// ═══════════════════════════════════════════════════════════════════
//  FILE MODE — unchanged from before, now behind `file` subcommand
// ═══════════════════════════════════════════════════════════════════

/// Turn a detector classification into the concrete PAL-or-NTSC choice
/// a `FrameReconstructor` needs.
///
/// `detect_sync_pulses` / `detect_from_iq` can legitimately return
/// [`SignalType::AnalogVideoUnknown`] — "this is analog video but the
/// FFT couldn't resolve the 109 Hz PAL/NTSC line-rate gap" (common on
/// the first wideband chunk). The library's docs explicitly warn
/// callers not to act on a PAL-vs-NTSC tag in that state; treating it
/// as NTSC (what `== AnalogVideoPal` comparisons silently did) builds
/// a reconstructor with the wrong line rate and geometry for a PAL
/// signal → rolling, torn video with no hint why. Instead, measure the
/// median sync-tip interval directly on the demodulated baseband via
/// `detect_video_standard`, and only fall back to `default` when even
/// that is inconclusive.
fn concretize_standard(
    tagged: SignalType,
    baseband_iq: &[Complex<f32>],
    sample_rate: u32,
    default: SignalType,
) -> SignalType {
    match tagged {
        SignalType::AnalogVideoPal | SignalType::AnalogVideoNtsc => tagged,
        _ => {
            let demod = fm_demod(baseband_iq);
            match detect_video_standard(&demod, sample_rate) {
                s @ (SignalType::AnalogVideoPal | SignalType::AnalogVideoNtsc) => {
                    println!("  → Standard ambiguous; time-domain line-rate check says {s:?}");
                    s
                }
                _ => {
                    println!("  → Standard ambiguous and undecidable; defaulting to {default:?}");
                    default
                }
            }
        }
    }
}

fn run_file(args: FileArgs) -> anyhow::Result<()> {
    // Attempt to load SigMF metadata
    let mut meta_path = args.input.clone();
    meta_path.set_extension("sigmf-meta");

    let mut sample_rate = args.sample_rate.unwrap_or(100_000_000);
    let mut fm_deviation = args.fm_deviation.unwrap_or(17_000_000.0);
    let mut rf_center_freq = 5_800_000_000.0_f64;
    let mut channels_to_spawn = Vec::new();

    if meta_path.exists() {
        println!("Found SigMF metadata: {:?}", meta_path);
        let meta_str = std::fs::read_to_string(&meta_path)?;
        let meta_json: serde_json::Value = serde_json::from_str(&meta_str)?;

        if let Some(global) = meta_json.get("global") {
            if let Some(sr) = global.get("core:sample_rate").and_then(|v| v.as_u64()) {
                sample_rate = sr as u32;
            }
            if let Some(dev) = global.get("fpv:fm_deviation").and_then(|v| v.as_f64()) {
                fm_deviation = dev as f32;
            }
            // The parser below reads interleaved little-endian f32 IQ
            // pairs unconditionally; any other recorded datatype would
            // silently decode as garbage, so reject it loudly instead.
            match global.get("core:datatype").and_then(|v| v.as_str()) {
                Some("cf32_le") | None => {}
                Some(other) => anyhow::bail!(
                    "unsupported SigMF core:datatype '{}'; only cf32_le (interleaved little-endian f32 IQ) is supported",
                    other
                ),
            }
        }

        if let Some(captures) = meta_json.get("captures").and_then(|v| v.as_array())
            && let Some(first_cap) = captures.first()
            && let Some(freq) = first_cap.get("core:frequency").and_then(|v| v.as_f64())
        {
            rf_center_freq = freq;
        }

        if let Some(annotations) = meta_json.get("annotations").and_then(|v| v.as_array()) {
            for ann in annotations {
                let lower = ann
                    .get("core:freq_lower_edge")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0);
                let upper = ann
                    .get("core:freq_upper_edge")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0);
                let label = ann
                    .get("core:label")
                    .and_then(|v| v.as_str())
                    .unwrap_or("Unknown");

                let freq_offset = (lower + upper) / 2.0;
                let label_upper = label.to_uppercase();
                let sig_type = if label_upper.contains("PAL") {
                    Some(SignalType::AnalogVideoPal)
                } else if label_upper.contains("NTSC") {
                    Some(SignalType::AnalogVideoNtsc)
                } else if label_upper.contains("ANALOG") || label_upper.contains("VIDEO") {
                    None
                } else {
                    continue;
                };

                channels_to_spawn.push((sig_type, freq_offset as f32, label.to_string()));
            }
        }
    }

    if channels_to_spawn.is_empty() {
        channels_to_spawn.push((None, -25_000_000.0, "Channel A".to_string()));
        channels_to_spawn.push((None, 25_000_000.0, "Channel B".to_string()));
    }

    println!(
        "Configuration loaded: {} MSPS, {} MHz Deviation",
        sample_rate as f32 / 1e6,
        fm_deviation / 1e6
    );

    let mut data_path = args.input.clone();
    if data_path.extension().and_then(|s| s.to_str()) == Some("sigmf-meta") {
        data_path.set_extension("sigmf-data");
    }

    // Validate the resolved DSP parameters before they feed buffer-size
    // and filter math: a zero sample rate makes `chunk_size` 0 (an
    // empty read that silently "succeeds" on an empty slice), and a
    // non-positive deviation collapses the DDC passband.
    if sample_rate == 0 {
        anyhow::bail!("sample rate resolved to 0 Hz; pass a valid --sample-rate");
    }
    if fm_deviation <= 0.0 {
        anyhow::bail!(
            "fm deviation resolved to {} Hz; pass a positive --fm-deviation",
            fm_deviation
        );
    }

    let mut file = File::open(&data_path)?;
    let chunk_size = sample_rate as usize / 10;
    let mut buf = vec![0u8; chunk_size * 8];

    file.read_exact(&mut buf)?;

    let first_iq: Vec<Complex<f32>> = buf
        .as_chunks::<8>()
        .0
        .iter()
        .map(|chunk| {
            let re = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            let im = f32::from_le_bytes([chunk[4], chunk[5], chunk[6], chunk[7]]);
            Complex::new(re, im)
        })
        .collect();

    let detector = AnalogFpvDetector::default();
    let mut resolved_channels: Vec<(SignalType, f32, f32)> = Vec::new();
    for (maybe_type, freq_offset, label) in &channels_to_spawn {
        let sig_type = if let Some(t) = maybe_type {
            println!(
                "Channel '{}' at {:.0} MHz: explicitly labeled {:?}",
                label,
                freq_offset / 1e6,
                t
            );
            *t
        } else {
            println!(
                "Channel '{}' at {:.0} MHz: probing signal...",
                label,
                freq_offset / 1e6
            );
            let mut probe_ddc =
                StreamingDDC::new(*freq_offset, sample_rate, fm_deviation + LUMA_HEADROOM_HZ);
            let probe_iq = probe_ddc.process(&first_iq);
            let (detected_type, confidence) = detector.detect_sync_pulses(&probe_iq, sample_rate);
            if matches!(
                detected_type,
                SignalType::AnalogVideoPal | SignalType::AnalogVideoNtsc
            ) {
                println!(
                    "  → Auto-detected: {:?} (confidence {:.0}%)",
                    detected_type,
                    confidence * 100.0
                );
                detected_type
            } else if detected_type == SignalType::Unknown {
                println!("  → Could not detect standard, defaulting to NTSC");
                SignalType::AnalogVideoNtsc
            } else {
                concretize_standard(
                    detected_type,
                    &probe_iq,
                    sample_rate,
                    SignalType::AnalogVideoNtsc,
                )
            }
        };
        resolved_channels.push((
            sig_type,
            *freq_offset,
            (fm_deviation + LUMA_HEADROOM_HZ) * 2.0,
        ));
    }

    let exit_reason = Arc::new(std::sync::atomic::AtomicU8::new(0));
    let tune_freq = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let frames_decoded = Arc::new(std::sync::atomic::AtomicU64::new(0));

    let _ = run_viewer_pipeline(
        resolved_channels,
        sample_rate,
        fm_deviation,
        args.deemphasis_tau,
        args.demod,
        args.denoise,
        args.denoise_model.clone(),
        rf_center_freq,
        args.debug,
        args.temporal_window,
        None,
        exit_reason,
        tune_freq,
        frames_decoded.clone(),
        move |channel_txs, _exit_flag| {
            let _ = file.seek(SeekFrom::Start(0));
            let mut active_txs = channel_txs;
            // Looping playback wraps to the start on EOF, and the last
            // sample of the file is not adjacent to the first. Without
            // this the decoder splices the seam and takes the phase
            // difference across it as a frequency — the same spike a
            // dropped chunk produces, once per loop.
            let mut wrapped = false;
            loop {
                if active_txs.is_empty() {
                    return;
                }
                match file.read(&mut buf) {
                    Ok(0) => {
                        let _ = file.seek(SeekFrom::Start(0));
                        wrapped = true;
                    }
                    Ok(bytes_read) => {
                        let samples_read = bytes_read / 8;
                        // Parse via from_le_bytes rather than
                        // bytemuck::cast_slice: cast_slice panics by
                        // contract if the Vec<u8> isn't 4-byte aligned,
                        // which the allocator happens to guarantee today
                        // but nothing requires.
                        let iq_vec: Vec<Complex<f32>> = buf[..samples_read * 8]
                            .as_chunks::<8>()
                            .0
                            .iter()
                            .map(|c| {
                                let re = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                                let im = f32::from_le_bytes([c[4], c[5], c[6], c[7]]);
                                Complex::new(re, im)
                            })
                            .collect();
                        let pooled =
                            orecchiette_sdr_source_rs::PooledIqBuffer::new_unpooled(iq_vec);
                        let arc_chunk = Arc::new(pooled);
                        let breaks = std::mem::take(&mut wrapped);
                        active_txs.retain(|tx| {
                            tx.send(IqChunk {
                                samples: Arc::clone(&arc_chunk),
                                discontinuous: breaks,
                            })
                            .is_ok()
                        });
                    }
                    Err(_) => break,
                }
            }
        },
    )?;
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════
//  LIVE USRP MODE
// ═══════════════════════════════════════════════════════════════════

pub enum RunLiveResult {
    UserExit,
    SignalLost,
    TooManyOverruns,
    /// User pressed 'S' — skip this frequency and resume scanning.
    SkipFrequency,
    /// User pressed 'N' — find the next signal (resume scanning without blacklisting).
    NextChannel,
    /// User pressed 'C' + two-char channel name — tune to a specific channel.
    TuneToChannel(f64),
}

/// Helper: convert a minifb Key to an ASCII char (A-Z, 0-9).
fn key_to_char(k: Key) -> Option<char> {
    match k {
        Key::A => Some('A'),
        Key::B => Some('B'),
        Key::C => Some('C'),
        Key::D => Some('D'),
        Key::E => Some('E'),
        Key::F => Some('F'),
        Key::G => Some('G'),
        Key::H => Some('H'),
        Key::I => Some('I'),
        Key::J => Some('J'),
        Key::K => Some('K'),
        Key::L => Some('L'),
        Key::M => Some('M'),
        Key::N => Some('N'),
        Key::O => Some('O'),
        Key::P => Some('P'),
        Key::Q => Some('Q'),
        Key::R => Some('R'),
        Key::S => Some('S'),
        Key::T => Some('T'),
        Key::U => Some('U'),
        Key::V => Some('V'),
        Key::W => Some('W'),
        Key::X => Some('X'),
        Key::Y => Some('Y'),
        Key::Z => Some('Z'),
        Key::Key0 | Key::NumPad0 => Some('0'),
        Key::Key1 | Key::NumPad1 => Some('1'),
        Key::Key2 | Key::NumPad2 => Some('2'),
        Key::Key3 | Key::NumPad3 => Some('3'),
        Key::Key4 | Key::NumPad4 => Some('4'),
        Key::Key5 | Key::NumPad5 => Some('5'),
        Key::Key6 | Key::NumPad6 => Some('6'),
        Key::Key7 | Key::NumPad7 => Some('7'),
        Key::Key8 | Key::NumPad8 => Some('8'),
        Key::Key9 | Key::NumPad9 => Some('9'),
        _ => None,
    }
}

/// State machine for the C -> two-char channel input.
#[derive(Debug, Clone)]
enum ChannelInputState {
    /// Not active.
    Idle,
    /// Waiting for first character (the band letter).
    WaitingFirst,
    /// Got the first character, waiting for second (the channel number).
    WaitingSecond(char),
}

fn run_live_usrp(args: UsrpArgs) -> anyhow::Result<()> {
    use orecchiette_sdr_usrp_rs::UsrpSource;

    let initial_sample_rate = if let Some(sr) = args.live.sample_rate {
        println!("Using user-specified sample rate: {:.2} MSPS", sr / 1e6);
        sr
    } else {
        println!("No sample rate specified. Defaulting to 25.00 MSPS.");
        25_000_000.0
    };

    let mut current_sample_rate = initial_sample_rate;
    let auto_scan = args.live.channel.is_none();
    let mut current_mode = if auto_scan {
        ViewerMode::Scan
    } else {
        ViewerMode::SingleChannel
    };

    // Frequencies temporarily blacklisted by the user pressing 'S'.
    // Cleared only when the user exits entirely.
    let mut skipped_freqs: std::collections::HashSet<u64> = std::collections::HashSet::new();

    let mut sweep = SweepState::new(args.live.scan_bands);

    let mut explicit_freq = if let Some(ref ch) = args.live.channel {
        Some(resolve_channel(ch)?)
    } else {
        None
    };

    let fm_deviation = args.live.fm_deviation;

    loop {
        let center_freq = match current_mode {
            ViewerMode::SingleChannel => {
                // Invariant: SingleChannel mode is only entered with a
                // resolved frequency. Surface a clean error instead of
                // panicking if a future state-machine edit ever breaks
                // that.
                let freq = explicit_freq.ok_or_else(|| {
                    anyhow::anyhow!("SingleChannel mode entered with no frequency set")
                })?;
                let ch_name = get_fpv_channel_name(freq / 1e6)
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| format!("{:.2} MHz", freq / 1e6));
                println!("Single-channel mode: {} ({:.3} MHz)", ch_name, freq / 1e6);
                freq
            }
            ViewerMode::Scan => {
                let hop_freqs = build_scan_hops(current_sample_rate, args.live.scan_bands);
                sweep.band.set_plan(
                    hop_freqs.clone(),
                    current_sample_rate * USABLE_SPAN_FRACTION,
                );
                sweep.band.set_skipped(&skipped_freqs);
                if sweep.ui.is_none() {
                    sweep.ui = Some(ScanWindow::open(sweep.band.full_height())?);
                }
                println!(
                    "Wideband scan requested [{}]. {} channels covered by {} tunes ({:.1} MHz usable per tune)...",
                    args.live.scan_bands.label(),
                    args.live.scan_bands.channel_freqs_hz().len(),
                    hop_freqs.len(),
                    current_sample_rate * USABLE_SPAN_FRACTION / 1e6
                );

                let config = SourceConfig {
                    sample_rate_hz: current_sample_rate,
                    channels_hz: hop_freqs.clone(),
                    // Long enough to actually deliver
                    // `DETECT_PACKETS_PER_HOP` captures plus retune
                    // settling. Two 65,536-sample packets is 5.2 ms of
                    // signal at 25 MSPS and 6.6 ms at 20 MSPS, so 20 ms
                    // leaves comfortable margin at every rate the scan
                    // runs at. Two detection passes are ~18 ms of CPU,
                    // so the hop is bound by that rather than by the
                    // dwell.
                    //
                    // `dwell_max == dwell_min` leaves the source crate's
                    // adaptive dwell off: a longer look neither promotes
                    // confidence on a strong signal (it plateaus at 0.80
                    // from the second packet) nor recovers a weak one
                    // (the sigma cliff sits between 1.0 and 1.5 and does
                    // not move with 12x the dwell).
                    dwell_min: Duration::from_millis(20),
                    dwell_max: Duration::from_millis(20),
                    dwell_extension: Duration::ZERO,
                };
                let advice = Arc::new(NoOpDwell) as Arc<dyn DwellAdvice>;
                let source = Box::new(UsrpSource {
                    args: args.args.clone(),
                    gain_db: args.gain,
                    antenna: args.antenna.clone(),
                });

                let handle = source.start(config, advice.clone())?;
                let detector = viewer_detector();
                let mut best_hit: Option<(f64, f32, SignalType)> = None;
                let mut seen_freqs = std::collections::HashSet::new();
                let mut sweeps_completed = 0;
                let mut last_progress_msg = Instant::now();
                let mut last_processed_freq = 0;
                let mut packets_this_hop = 0usize;
                let multi_hop = hop_freqs.len() > 1;
                let mut probes: Vec<ProbeEnergy> = Vec::new();

                // `None` when the operator closed the scan window.
                let found_freq: Option<f64> = 'scan_loop: loop {
                    let packet = match handle.receiver.recv_timeout(Duration::from_millis(1000)) {
                        Ok(packet) => packet,
                        // `recv_timeout` returns Err immediately when the
                        // sender is dropped (capture thread died — e.g. a
                        // USB reset), not only after the timeout. Retrying
                        // a disconnected channel busy-spins at 100% CPU
                        // flooding stdout, with no exit path.
                        Err(e) if e.is_disconnected() => {
                            anyhow::bail!(
                                "SDR capture thread died during scan (channel disconnected)"
                            );
                        }
                        Err(_) => {
                            println!("SDR timed out during scan. Retrying...");
                            continue;
                        }
                    };
                    {
                        let center = packet.center_frequency_hz as u64;

                        // Pump the scan window on every packet, drained
                        // ones included, so it stays responsive.
                        if let Some(ui) = sweep.ui.as_mut()
                            && !ui.update(&sweep.band)
                        {
                            (handle.stop)();
                            break 'scan_loop None;
                        }

                        // Feed the sweep `DETECT_PACKETS_PER_HOP` packets
                        // per hop, then drain the rest of the dwell so the
                        // SDR's buffer doesn't overflow.
                        //
                        // A change of centre frequency is the only signal
                        // `IqPacket` carries that a new hop began, so the
                        // budget resets on that. `plan_tune_centers`
                        // dedups and emits strictly ascending centres
                        // (pinned by `tune_planner_emits_ascending_hops`),
                        // so two consecutive hops never share one.
                        //
                        // With a single tune the source never retunes, so
                        // there is no boundary to observe and no hop to
                        // drain for; every packet is processed.
                        if multi_hop && center == last_processed_freq {
                            if packets_this_hop >= DETECT_PACKETS_PER_HOP {
                                continue;
                            }
                        } else if center != last_processed_freq {
                            last_processed_freq = center;
                            packets_this_hop = 0;
                        }
                        packets_this_hop += 1;

                        seen_freqs.insert(center);

                        let results = detector.detect_from_iq_integrated_with_probes(
                            &packet.samples,
                            center,
                            current_sample_rate as u32,
                            &mut sweep.integrator,
                            &mut probes,
                        );
                        // With one tune there is no retune to mark a new
                        // look, so every packet replaces the last.
                        sweep.band.record_hop(
                            center as f64,
                            packets_this_hop == 1 || !multi_hop,
                            &probes,
                            &results,
                        );
                        for res in results {
                            let ch_name = get_fpv_channel_name(res.frequency_hz as f64 / 1e6)
                                .unwrap_or("Unknown");
                            println!(
                                "  → Found {:?} at {:.3} MHz (channel {}, conf {:.0}%, rssi {:.1} dBm)",
                                res.signal_type,
                                res.frequency_hz as f64 / 1e6,
                                ch_name,
                                res.confidence * 100.0,
                                res.rssi_dbm
                            );
                            // A hit is selectable unless the user
                            // blacklisted everything it could tune to.
                            // Off-table bands (3.3 GHz, 6-7 GHz) have no
                            // channel-table candidates at all; those tune
                            // to the detection frequency itself (the
                            // `candidates.is_empty()` snap below), so
                            // they are gated on that frequency's own skip
                            // key rather than on having a candidate.
                            let candidates = get_candidate_fpv_channels(res.frequency_hz as f64);
                            let selectable = if candidates.is_empty() {
                                // Off-table hits tune to (and get
                                // blacklisted at) their raw detection
                                // frequency, which jitters by up to a
                                // probe step (~2.5 MHz) between sweeps —
                                // an exact u64 match would never
                                // re-recognise a skipped signal. Match
                                // within the same tolerance the channel
                                // snap uses.
                                !skipped_freqs.iter().any(|&s| {
                                    (s as f64 - res.frequency_hz as f64).abs()
                                        <= CHANNEL_SNAP_TOLERANCE_MHZ * 1e6
                                })
                            } else {
                                candidates
                                    .iter()
                                    .any(|&c| !skipped_freqs.contains(&(c.round() as u64)))
                            };
                            let better = best_hit
                                .map(|(_, best_rssi, _)| res.rssi_dbm > best_rssi)
                                .unwrap_or(true);
                            if selectable && better {
                                best_hit =
                                    Some((res.frequency_hz as f64, res.rssi_dbm, res.signal_type));
                            }
                        }
                        // Sweep is complete once every hop has been visited
                        // *and* the current one has had its full packet
                        // budget. Without the budget half of that test this
                        // fires on the first packet of the last hop, which
                        // both cuts that hop short of the two passes it
                        // needs to detect anything and clears `seen_freqs`
                        // mid-hop, sliding the sweep boundary one hop
                        // earlier on every subsequent pass.
                        if seen_freqs.len() >= hop_freqs.len()
                            && packets_this_hop >= DETECT_PACKETS_PER_HOP
                        {
                            sweep.band.end_sweep();
                            if let Some((freq, _rssi, _sig_type)) = best_hit {
                                // Found something, break out of the infinite scan loop
                                (handle.stop)();
                                std::thread::sleep(Duration::from_millis(200));

                                let mut candidates = get_candidate_fpv_channels(freq);
                                candidates
                                    .retain(|&c| !skipped_freqs.contains(&(c.round() as u64)));
                                if candidates.is_empty() {
                                    let snapped_freq = snap_to_nearest_fpv_channel(freq);
                                    let ch_name = get_fpv_channel_name(snapped_freq / 1e6)
                                        .unwrap_or("Unknown");
                                    println!(
                                        "Sweep complete. Auto-tuning to exact channel: {} ({:.3} MHz) [raw hit at {:.3} MHz]",
                                        ch_name,
                                        snapped_freq / 1e6,
                                        freq / 1e6
                                    );
                                    break 'scan_loop Some(snapped_freq);
                                }

                                println!(
                                    "Coarse hit at {:.3} MHz. Fine-tuning across {} candidate channels...",
                                    freq / 1e6,
                                    candidates.len()
                                );

                                let ft_config = SourceConfig {
                                    sample_rate_hz: current_sample_rate,
                                    channels_hz: candidates.clone(),
                                    // Fine-tune needs slightly more settle time
                                    // but still only one chunk per candidate.
                                    dwell_min: Duration::from_millis(15),
                                    dwell_max: Duration::from_millis(15),
                                    dwell_extension: Duration::ZERO,
                                };
                                let ft_source = Box::new(UsrpSource {
                                    args: args.args.clone(),
                                    gain_db: args.gain,
                                    antenna: args.antenna.clone(),
                                });
                                let ft_handle = ft_source.start(ft_config, advice.clone())?;

                                let mut ft_best_hit: Option<(f64, f32, f32)> = None;
                                let mut ft_seen = std::collections::HashSet::new();
                                let mut ft_last_freq = 0;

                                loop {
                                    let packet = match ft_handle
                                        .receiver
                                        .recv_timeout(Duration::from_millis(1000))
                                    {
                                        Ok(packet) => packet,
                                        // Same immediate-Err-on-disconnect
                                        // semantics as the coarse loop above.
                                        Err(e) if e.is_disconnected() => {
                                            anyhow::bail!(
                                                "SDR capture thread died during fine-tune (channel disconnected)"
                                            );
                                        }
                                        Err(_) => {
                                            println!("SDR timed out during fine-tune. Retrying...");
                                            continue;
                                        }
                                    };
                                    {
                                        let center = packet.center_frequency_hz as u64;

                                        // Drain duplicates to stay real-time
                                        if center == ft_last_freq {
                                            continue;
                                        }
                                        ft_last_freq = center;

                                        ft_seen.insert(center);

                                        let (_sig_type, conf) = detector.detect_sync_pulses(
                                            &packet.samples,
                                            current_sample_rate as u32,
                                        );

                                        let mut rssi = -100.0;
                                        if let Some(res) = detector
                                            .detect_from_iq(
                                                &packet.samples,
                                                center,
                                                current_sample_rate as u32,
                                            )
                                            .first()
                                        {
                                            rssi = res.rssi_dbm;
                                        }

                                        println!(
                                            "  → Testing {:.3} MHz ({}): conf {:.0}%, rssi {:.1} dBm",
                                            center as f64 / 1e6,
                                            get_fpv_channel_name(center as f64 / 1e6)
                                                .unwrap_or("Unknown"),
                                            conf * 100.0,
                                            rssi
                                        );

                                        if let Some((_, best_conf, best_rssi)) = ft_best_hit {
                                            // Use confidence first, RSSI as a tie-breaker or fallback if confidences are similar
                                            if conf > best_conf + 0.05
                                                || (conf > best_conf - 0.05 && rssi > best_rssi)
                                            {
                                                ft_best_hit = Some((center as f64, conf, rssi));
                                            }
                                        } else {
                                            ft_best_hit = Some((center as f64, conf, rssi));
                                        }

                                        if ft_seen.len() >= candidates.len() {
                                            (ft_handle.stop)();
                                            std::thread::sleep(Duration::from_millis(200));

                                            if let Some((best_freq, best_conf, _)) = ft_best_hit
                                                && best_conf > 0.1
                                            {
                                                println!(
                                                    "Fine-tuning complete. Selected exact channel: {} ({:.3} MHz)",
                                                    get_fpv_channel_name(best_freq / 1e6)
                                                        .unwrap_or("Unknown"),
                                                    best_freq / 1e6
                                                );
                                                break 'scan_loop Some(best_freq);
                                            }

                                            println!(
                                                "Fine-tuning didn't find clear sync pulses. Falling back to simple snap."
                                            );
                                            let snapped_freq = snap_to_nearest_fpv_channel(freq);
                                            let ch_name = get_fpv_channel_name(snapped_freq / 1e6)
                                                .unwrap_or("Unknown");
                                            println!(
                                                "Auto-tuning to exact channel: {} ({:.3} MHz)",
                                                ch_name,
                                                snapped_freq / 1e6
                                            );
                                            break 'scan_loop Some(snapped_freq);
                                        }
                                    }
                                }
                            } else {
                                sweeps_completed += 1;
                                // Time-gated, not every Nth sweep. A sweep
                                // is `tunes × dwell`, which ranges from
                                // ~120 ms for six tunes to ~2 s for the
                                // whole table — so a fixed sweep count
                                // prints either far too often or barely at
                                // all depending on --scan-bands and the
                                // sample rate. It also bounds the
                                // degenerate single-tune case, where a
                                // "sweep" is one hop and completes on
                                // every packet.
                                if last_progress_msg.elapsed() >= Duration::from_secs(5) {
                                    println!(
                                        "Still sweeping ({sweeps_completed} passes)... no signals found yet."
                                    );
                                    last_progress_msg = Instant::now();
                                }
                                // Reset for the next sweep.
                                seen_freqs.clear();
                            }
                        }
                    }
                };
                let Some(found_freq) = found_freq else {
                    // The operator closed the scan window: done.
                    break;
                };
                explicit_freq = Some(found_freq);
                found_freq
            }
        };

        let source = Box::new(UsrpSource {
            args: args.args.clone(),
            gain_db: args.gain,
            antenna: args.antenna.clone(),
        });

        // The picture window takes over from the scan window.
        sweep.ui = None;
        match run_live(
            source,
            current_sample_rate,
            fm_deviation,
            args.live.deemphasis_tau,
            args.live.demod,
            args.live.denoise,
            args.live.denoise_model.clone(),
            center_freq,
            args.live.standard,
            None,
            args.live.debug,
            args.live.temporal_window,
            Some(&sweep.band),
            OverrunPolicy::StepDown,
            Duration::ZERO,
        )? {
            RunLiveResult::UserExit => {
                break;
            }
            RunLiveResult::SignalLost => {
                println!("Signal lost. Resuming scan...");
                if auto_scan {
                    current_mode = ViewerMode::Scan;
                    explicit_freq = None;
                } else {
                    println!("Cannot resume scan in explicit channel mode. Exiting.");
                    break;
                }
            }
            RunLiveResult::SkipFrequency => {
                let freq_key = center_freq.round() as u64;
                skipped_freqs.insert(freq_key);
                println!(
                    "Skipped {:.3} MHz ({}). {} frequencies blacklisted. Resuming scan...",
                    center_freq / 1e6,
                    get_fpv_channel_name(center_freq / 1e6).unwrap_or("Unknown"),
                    skipped_freqs.len()
                );
                if auto_scan {
                    current_mode = ViewerMode::Scan;
                    explicit_freq = None;
                } else {
                    println!("Cannot resume scan in explicit channel mode. Exiting.");
                    break;
                }
            }
            RunLiveResult::NextChannel => {
                println!("Finding next channel...");
                if auto_scan {
                    current_mode = ViewerMode::Scan;
                    explicit_freq = None;
                } else {
                    println!("Cannot scan for next channel in explicit channel mode. Exiting.");
                    break;
                }
            }
            RunLiveResult::TuneToChannel(freq) => {
                let ch_name = get_fpv_channel_name(freq / 1e6)
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| format!("{:.2} MHz", freq / 1e6));
                println!("Tuning to channel {} ({:.3} MHz)...", ch_name, freq / 1e6);
                explicit_freq = Some(freq);
                current_mode = ViewerMode::SingleChannel;
            }
            RunLiveResult::TooManyOverruns => {
                println!("Hardware buffer overrun limit reached.");
                if current_sample_rate > 16_000_000.0 {
                    current_sample_rate -= 5_000_000.0;
                    println!(
                        "Stepping down SDR sample rate to {:.2} MSPS...",
                        current_sample_rate / 1e6
                    );
                    if auto_scan {
                        current_mode = ViewerMode::Scan;
                        explicit_freq = None;
                    }
                } else {
                    println!("Sample rate is already near minimum. Cannot step down further.");
                    if auto_scan {
                        current_mode = ViewerMode::Scan;
                        explicit_freq = None;
                    } else {
                        break;
                    }
                }
            }
        }
    }
    // Use process::exit to skip UHD's broken C++ static destructors.
    // libuhd keeps a global `std::map<unsigned long, usrp_ptr>` that
    // double-frees during __cxa_finalize if the Rust side already dropped
    // the Usrp handle.  The capture thread is already stopped cleanly by
    // handle_stop() in run_live, so this is safe.
    std::process::exit(0);
}

// ═══════════════════════════════════════════════════════════════════
//  LIVE HACKRF MODE
// ═══════════════════════════════════════════════════════════════════

fn run_live_hackrf(args: HackrfArgs) -> anyhow::Result<()> {
    use orecchiette_sdr_hackrf_rs::{HACKRF_MAX_SAMPLE_RATE_HZ, HackRfSource};

    // HackRF One is USB 2.0: default to its ~20 MSPS ceiling (just enough
    // for analog FPV's ~20 MHz FM) and clamp any larger request to it.
    let requested = args.live.sample_rate.unwrap_or(HACKRF_MAX_SAMPLE_RATE_HZ);
    let sample_rate = requested.min(HACKRF_MAX_SAMPLE_RATE_HZ);
    if requested > HACKRF_MAX_SAMPLE_RATE_HZ {
        println!(
            "Requested {:.2} MSPS exceeds the HackRF's {:.0} MSPS USB-2.0 ceiling; using {:.0} MSPS.",
            requested / 1e6,
            HACKRF_MAX_SAMPLE_RATE_HZ / 1e6,
            sample_rate / 1e6
        );
    } else {
        println!("HackRF sample rate: {:.2} MSPS.", sample_rate / 1e6);
    }

    let fm_deviation = args.live.fm_deviation;
    let auto_scan = args.live.channel.is_none();
    let mut skipped_freqs: std::collections::HashSet<u64> = std::collections::HashSet::new();
    // Cross-sweep spectral integration (see the USRP path's note).
    let mut sweep = SweepState::new(args.live.scan_bands);
    let mut explicit_freq = match args.live.channel {
        Some(ref ch) => Some(resolve_channel(ch)?),
        None => None,
    };

    // What the sweep classified the chosen signal as. Carried into the
    // live decode so a concrete PAL/NTSC verdict from the scan is not
    // re-derived (and possibly lost) by the single-channel re-detect:
    // the sweep integrates several looks per hop, while the re-detect
    // gets one centre and a fallback default — it measured NTSC on air
    // and then decoded it as default-PAL geometry.
    let mut scan_standard: Option<SignalType> = None;
    loop {
        let center_freq = match explicit_freq {
            Some(f) => f,
            None => match scan_band_for_channel(
                Box::new(HackRfSource {
                    lna_gain: args.lna_gain,
                    vga_gain: args.vga_gain,
                    amp_enable: args.amp,
                    bias_tee: args.bias_tee,
                }),
                "HackRF",
                sample_rate,
                args.live.scan_bands,
                // See the USRP scan's dwell for why 20 ms and why
                // adaptive dwell stays off. HackRF retunes its
                // synthesiser in well under a millisecond, and its
                // 20 MSPS ceiling makes packets the longest of any
                // backend at 3.3 ms — 20 ms lands the per-hop budget
                // with margin.
                SweepBudget::flat(
                    Duration::from_millis(20),
                    detect_packets_per_hop(sample_rate),
                ),
                &skipped_freqs,
                &mut sweep,
            )? {
                // Used directly as this iteration's centre; in auto-scan
                // mode the RunLiveResult handling below decides whether to
                // re-scan, so we don't need to stash it in `explicit_freq`.
                ScanOutcome::Found(f, sig) => {
                    scan_standard = match sig {
                        SignalType::AnalogVideoPal | SignalType::AnalogVideoNtsc => Some(sig),
                        _ => None,
                    };
                    f
                }
                ScanOutcome::Empty => {
                    println!("No analog FPV signal found in the scan band. Retrying...");
                    continue;
                }
                ScanOutcome::Quit => break,
            },
        };

        let ch_name = get_fpv_channel_name(center_freq / 1e6)
            .map(|s| s.to_string())
            .unwrap_or_else(|| format!("{:.2} MHz", center_freq / 1e6));
        println!("HackRF tuned: {} ({:.3} MHz)", ch_name, center_freq / 1e6);

        let source = Box::new(HackRfSource {
            lna_gain: args.lna_gain,
            vga_gain: args.vga_gain,
            amp_enable: args.amp,
            bias_tee: args.bias_tee,
        });

        // The picture window takes over from the scan window.
        sweep.ui = None;
        match run_live(
            source,
            sample_rate,
            fm_deviation,
            args.live.deemphasis_tau,
            args.live.demod,
            args.live.denoise,
            args.live.denoise_model.clone(),
            center_freq,
            args.live.standard,
            scan_standard,
            args.live.debug,
            args.live.temporal_window,
            Some(&sweep.band),
            OverrunPolicy::StepDown,
            Duration::ZERO,
        )? {
            RunLiveResult::UserExit => break,
            RunLiveResult::SignalLost | RunLiveResult::NextChannel => {
                if auto_scan {
                    explicit_freq = None;
                } else {
                    println!("Signal lost (explicit-channel mode). Exiting.");
                    break;
                }
            }
            RunLiveResult::SkipFrequency => {
                skipped_freqs.insert(center_freq.round() as u64);
                if auto_scan {
                    explicit_freq = None;
                } else {
                    break;
                }
            }
            RunLiveResult::TuneToChannel(freq) => {
                explicit_freq = Some(freq);
                scan_standard = None;
            }
            // HackRF doesn't surface hardware-overrun metadata, so this
            // variant isn't expected from its backend; handle it like a
            // signal loss to stay exhaustive and safe.
            RunLiveResult::TooManyOverruns => {
                if auto_scan {
                    explicit_freq = None;
                } else {
                    break;
                }
            }
        }
    }
    Ok(())
}

/// What a backend keeps between sweeps: the cross-sweep integrator (a
/// weak signal accumulates sensitivity sweep over sweep, ~+5 dB after
/// four visits to a hop; buckets self-reset if the sample rate steps
/// down), the band panel's model, and the scan window, which is open
/// only while sweeping.
struct SweepState {
    integrator: SpectralIntegrator,
    band: BandScan,
    ui: Option<ScanWindow>,
}

impl SweepState {
    fn new(bands: ScanBands) -> Self {
        SweepState {
            integrator: SpectralIntegrator::new(4),
            band: BandScan::new(bands.channels()),
            ui: None,
        }
    }
}

/// What a backend wants done when the source flags an overrun.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OverrunPolicy {
    /// The host cannot keep up with the hardware: give the channel up so
    /// the caller can step the sample rate down (USRP over USB).
    StepDown,
    /// A gap in a network stream drops a few samples and glitches the
    /// picture; it says nothing about the channel. Count it, keep going.
    /// The RTSA HTTP server also leaves timestamp gaps around every
    /// retune, which under `StepDown` ended a fresh lock within 200 ms.
    #[cfg_attr(not(feature = "aaronia"), allow(dead_code))]
    Tolerate,
}

/// How long the RTSA HTTP server keeps delivering samples from the
/// previous centre after acknowledging a retune, over and above the
/// driver's own 20 ms drain. Packets inside this window belong to the
/// old channel, whatever their label says.
///
/// The 150–300 ms this once cited was never measured. On the hop path a
/// retune is carried by the signal within 38.9 ms worst case over 24
/// hops at 61.44 MSPS — but this guard is not that path. It covers
/// entering live decode, cold stream startup included, which that
/// measurement did not touch. It stays at 400 ms until startup is
/// measured on its own.
#[cfg(feature = "aaronia")]
const AARONIA_HTTP_TUNE_SETTLE: Duration = Duration::from_millis(400);

/// Length of contiguous signal the standard is decided on. PAL and NTSC
/// line rates are 109 Hz apart, so the FFT separates them only past
/// ~9.2 ms of record; one 65,536-sample packet is 1–4 ms at the live
/// rates, and on that the library can only answer "analog video,
/// standard unknown" — measured on a clean 20 dB NTSC link, which then
/// decoded as PAL. 25 ms puts the two rates 2.6 bins apart.
const STANDARD_DETECT_RECORD: Duration = Duration::from_millis(25);
/// Give up waiting for a clean record after this long and decide on
/// whatever was gathered.
const STANDARD_DETECT_PATIENCE: Duration = Duration::from_millis(1500);

/// Confidence floor for the sweep and the lock check. Both ask whether
/// analog video is present, not which standard it is: the standard is
/// settled afterwards on a long record (`STANDARD_DETECT_RECORD`). Their
/// 1–4 ms packets cannot resolve PAL from NTSC, and the library reports
/// that honestly as `AnalogVideoUnknown` at 0.6 — a hit that has passed
/// the harmonic and cepstral gates — which the detector's default 0.7
/// floor discarded. Measured on a 20 dB NTSC link at 15.36 MSPS: the
/// sweep reported an empty band on every pass, and a decoder with
/// perfect vertical sync "lost" its signal on every check.
const VIEWER_MIN_CONFIDENCE: f32 = 0.55;

/// The detector every live path uses, with the floor above.
fn viewer_detector() -> AnalogFpvDetector {
    let mut d = AnalogFpvDetector::default();
    d.min_confidence = VIEWER_MIN_CONFIDENCE;
    d
}

/// What a coarse sweep came back with.
enum ScanOutcome {
    /// A channel to tune to, and what the sweep classified it as.
    Found(f64, SignalType),
    /// The band was quiet this pass.
    Empty,
    /// The operator closed the scan window or pressed `Q`.
    Quit,
}

/// Single-stage coarse scan shared by the hopping backends: sweep the
/// planned tune centres, run the wideband detector on
/// `DETECT_PACKETS_PER_HOP` chunks per hop, and snap the strongest
/// (non-blacklisted) hit to the nearest FPV channel. Returns `None` if
/// the band is empty.
///
/// This is intentionally simpler than the USRP path's two-stage
/// coarse-then-fine-tune sweep — the single coarse snap is enough to
/// land on a channel. The `source` decides what actually retunes:
/// HackRF hops its synthesiser, while the Aaronia HTTP backend issues a
/// capture-config update to the RTSA server per hop, whose latency is
/// network-dependent — the per-hop packet budget doesn't care how long
/// a retune took, only that fresh packets eventually arrive at the new
/// centre.
fn scan_band_for_channel(
    source: Box<dyn SdrSource>,
    backend_label: &str,
    sample_rate: f64,
    scan_bands: ScanBands,
    budget: SweepBudget,
    skipped_freqs: &std::collections::HashSet<u64>,
    sweep: &mut SweepState,
) -> anyhow::Result<ScanOutcome> {
    let dwell = budget.dwell;
    let hop_freqs = build_scan_hops(sample_rate, scan_bands);
    sweep
        .band
        .set_plan(hop_freqs.clone(), sample_rate * USABLE_SPAN_FRACTION);
    sweep.band.set_skipped(skipped_freqs);
    if sweep.ui.is_none() {
        sweep.ui = Some(ScanWindow::open(sweep.band.full_height())?);
    }
    println!(
        "{backend_label} wideband scan [{}]: {} channels covered by {} tunes ({:.1} MHz usable per tune)...",
        scan_bands.label(),
        scan_bands.channel_freqs_hz().len(),
        hop_freqs.len(),
        sample_rate * USABLE_SPAN_FRACTION / 1e6
    );
    let config = SourceConfig {
        sample_rate_hz: sample_rate,
        channels_hz: hop_freqs.clone(),
        // Caller-chosen: the dwell must cover the backend's real retune
        // latency, or packets from one centre arrive tagged with the
        // next hop's frequency and every detection is reported at the
        // wrong absolute frequency. See the call sites.
        dwell_min: dwell,
        dwell_max: dwell,
        dwell_extension: Duration::ZERO,
    };
    let advice = Arc::new(NoOpDwell) as Arc<dyn DwellAdvice>;
    let handle = source.start(config, advice)?;
    let detector = viewer_detector();

    // (raw freq Hz, rssi dBm, what the sweep classified it as)
    let mut best_hit: Option<(f64, f32, SignalType)> = None;
    let mut seen = std::collections::HashSet::new();
    let mut last_freq = 0u64;
    let mut packets_this_hop = 0usize;
    // What a packet is worth here decides how many to spend.
    let packets_per_hop = budget.packets_per_hop;
    let mut probes: Vec<ProbeEnergy> = Vec::new();
    let outcome = |hit: Option<(f64, f32, SignalType)>| match hit {
        Some((f, _, sig)) => ScanOutcome::Found(snap_to_nearest_fpv_channel(f), sig),
        None => ScanOutcome::Empty,
    };

    loop {
        match handle.receiver.recv_timeout(Duration::from_millis(2000)) {
            Ok(packet) => {
                let center = packet.center_frequency_hz as u64;
                // Pump the scan window on every packet, drained ones
                // included, so it stays responsive through a long dwell.
                if let Some(ui) = sweep.ui.as_mut()
                    && !ui.update(&sweep.band)
                {
                    (handle.stop)();
                    return Ok(ScanOutcome::Quit);
                }
                // `DETECT_PACKETS_PER_HOP` detection passes per hop, then
                // drain the rest of the dwell. Unlike the USRP loop this
                // one needs no single-tune escape: it *returns* once the
                // sweep is complete rather than clearing `seen` and going
                // round again, so a one-tune plan finishes on its first
                // packet and can never sit draining.
                if center == last_freq {
                    if packets_this_hop >= packets_per_hop {
                        continue;
                    }
                } else {
                    last_freq = center;
                    packets_this_hop = 0;
                }
                packets_this_hop += 1;
                seen.insert(center);

                let results = detector.detect_from_iq_integrated_with_probes(
                    &packet.samples,
                    center,
                    sample_rate as u32,
                    &mut sweep.integrator,
                    &mut probes,
                );
                sweep
                    .band
                    .record_hop(center as f64, packets_this_hop == 1, &probes, &results);

                for res in results {
                    let snapped = snap_to_nearest_fpv_channel(res.frequency_hz as f64);
                    if skipped_freqs.contains(&(snapped.round() as u64)) {
                        continue;
                    }
                    let better = best_hit.map(|(_, r, _)| res.rssi_dbm > r).unwrap_or(true);
                    if better {
                        best_hit = Some((res.frequency_hz as f64, res.rssi_dbm, res.signal_type));
                        println!(
                            "  → {:?} at {:.3} MHz ({}), rssi {:.1} dBm",
                            res.signal_type,
                            res.frequency_hz as f64 / 1e6,
                            get_fpv_channel_name(res.frequency_hz as f64 / 1e6)
                                .unwrap_or("Unknown"),
                            res.rssi_dbm
                        );
                    }
                }

                // As in the USRP loop: the last hop must get its full
                // packet budget before the sweep counts as done. Returning
                // on its first packet left the highest frequency in every
                // HackRF scan with a single detection pass — below the
                // length at which the sweep finds anything — so that
                // channel could never be reported.
                if seen.len() >= hop_freqs.len() && packets_this_hop >= packets_per_hop {
                    sweep.band.end_sweep();
                    (handle.stop)();
                    std::thread::sleep(Duration::from_millis(150));
                    return Ok(outcome(best_hit));
                }
            }
            Err(_) => {
                // SDR went quiet — return whatever we found this sweep.
                (handle.stop)();
                return Ok(outcome(best_hit));
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
//  LIVE AARONIA MODE
// ═══════════════════════════════════════════════════════════════════

#[cfg(feature = "aaronia")]
fn run_live_aaronia(cmd: AaroniaCmd) -> anyhow::Result<()> {
    use sdr_aaronia_rs::{SpectranBackend, SpectranSdrSource};

    // Backend params split from LiveArgs so a fresh source can be built
    // for every (re)tune: `run_live` consumes its source, and every
    // non-quit RunLiveResult needs a new one to honour the S/N/C keys
    // and signal loss.
    enum Backend {
        Http {
            url: String,
            ref_level: f64,
            stream_format: sdr_aaronia_rs::http_streaming::StreamFormat,
        },
        Sdk {
            serial: Option<String>,
            ref_level: f64,
        },
    }
    let (backend, live, label) = match cmd {
        AaroniaCmd::Http(h) => {
            let label = format!("Aaronia HTTP ({})", h.url);
            (
                Backend::Http {
                    url: h.url,
                    ref_level: h.ref_level,
                    stream_format: h.stream_format.to_driver(),
                },
                h.live,
                label,
            )
        }
        AaroniaCmd::Sdk(s) => (
            Backend::Sdk {
                serial: s.serial,
                ref_level: s.ref_level,
            },
            s.live,
            "Aaronia SDK".to_string(),
        ),
    };
    let make_source = |center_freq: f64| -> Box<dyn SdrSource> {
        match &backend {
            Backend::Http {
                url,
                ref_level,
                stream_format,
            } => Box::new(SpectranSdrSource {
                backend: SpectranBackend::Http(url.clone()),
                center_frequency_hz: center_freq,
                reference_level_dbm: *ref_level,
                block_size: 65_536,
                stream_format: Some(*stream_format),
            }),
            Backend::Sdk { serial, ref_level } => Box::new(SpectranSdrSource {
                backend: SpectranBackend::Sdk {
                    serial: serial.clone(),
                },
                center_frequency_hz: center_freq,
                reference_level_dbm: *ref_level,
                block_size: 65_536,
                stream_format: None,
            }),
        }
    };

    // `--sample-rate` is a rate, so it maps to the nearest rate the
    // hardware runs (61.44 MHz over powers of two) — not through the
    // driver's bandwidth-to-rate helper, which reads its argument as
    // usable spectrum and would answer 15.36 MSPS with a 30.72 MSPS
    // stream.
    let requested_rate = live.sample_rate.unwrap_or(AARONIA_DEFAULT_RATE_HZ);
    let sample_rate = sdr_aaronia_rs::nearest_iq_sample_rate(requested_rate);
    if (sample_rate - requested_rate).abs() > 1.0 {
        tracing::warn!(
            requested_hz = requested_rate,
            using_hz = sample_rate,
            "requested sample rate is not one the Aaronia hardware runs; using the nearest"
        );
    }
    let fm_deviation = live.fm_deviation;
    let auto_scan = live.channel.is_none();
    let mut explicit_freq = live
        .channel
        .as_ref()
        .map(|ch| resolve_channel(ch))
        .transpose()?;

    let mut skipped_freqs: std::collections::HashSet<u64> = std::collections::HashSet::new();
    let mut sweep = SweepState::new(live.scan_bands);
    // See the HackRF caller for why the sweep's PAL/NTSC verdict rides
    // along instead of being re-derived at the tuned centre.
    let mut scan_standard: Option<SignalType> = None;
    // Which sweep this is, for the sensitive-sweep cadence.
    let mut sweep_n: u32 = 0;

    loop {
        let center_freq = match explicit_freq {
            Some(f) => f,
            // The sweeping source is built at the band centre, but that
            // choice is cosmetic — the hop list in the scan's
            // SourceConfig retunes it before the first packet arrives.
            None => match scan_band_for_channel(
                make_source(SCAN_CENTER_HZ),
                &label,
                sample_rate,
                live.scan_bands,
                {
                    // Most sweeps are fast; every fourth spends the
                    // sensitive budget on every hop, so a signal that
                    // only appears after several packets of integration
                    // is found within four sweeps rather than never.
                    let b = SweepBudget::for_sweep(sweep_n, sample_rate);
                    sweep_n = sweep_n.wrapping_add(1);
                    b
                },
                &skipped_freqs,
                &mut sweep,
            )? {
                ScanOutcome::Found(f, sig) => {
                    scan_standard = match sig {
                        SignalType::AnalogVideoPal | SignalType::AnalogVideoNtsc => Some(sig),
                        _ => None,
                    };
                    f
                }
                ScanOutcome::Empty => {
                    println!("No analog FPV signal found in the scan band. Retrying...");
                    continue;
                }
                ScanOutcome::Quit => break,
            },
        };

        let ch_name = get_fpv_channel_name(center_freq / 1e6)
            .map(|s| s.to_string())
            .unwrap_or_else(|| format!("{:.2} MHz", center_freq / 1e6));
        // The sweep is over; hold the channel at the rate the decoder
        // would have decimated to rather than paying for spectrum the
        // DDC is about to discard. Scanning keeps `sample_rate`, so
        // losing the signal resumes the wide sweep unchanged.
        let live_rate = if live.sample_rate.is_some() {
            sample_rate
        } else {
            locked_capture_rate_hz(sample_rate, fm_deviation, live.demod)
        };
        if live_rate < sample_rate {
            println!(
                "{}: {} ({:.3} MHz) at {:.2} MSPS (scanned at {:.2})",
                label,
                ch_name,
                center_freq / 1e6,
                live_rate / 1e6,
                sample_rate / 1e6
            );
        } else {
            println!(
                "{}: {} ({:.3} MHz) at {:.2} MSPS",
                label,
                ch_name,
                center_freq / 1e6,
                live_rate / 1e6
            );
        }
        // The picture window takes over from the scan window.
        sweep.ui = None;
        match run_live(
            make_source(center_freq),
            live_rate,
            fm_deviation,
            live.deemphasis_tau,
            live.demod,
            live.denoise,
            live.denoise_model.clone(),
            center_freq,
            live.standard,
            scan_standard,
            live.debug,
            live.temporal_window,
            Some(&sweep.band),
            OverrunPolicy::Tolerate,
            AARONIA_HTTP_TUNE_SETTLE,
        )? {
            RunLiveResult::UserExit => break,
            r @ (RunLiveResult::SignalLost
            | RunLiveResult::NextChannel
            | RunLiveResult::TooManyOverruns) => {
                let why = match r {
                    RunLiveResult::SignalLost => "Signal lost",
                    RunLiveResult::NextChannel => "Next channel requested",
                    _ => "Too many stream overruns",
                };
                if auto_scan {
                    println!("{why}. Resuming scan...");
                    explicit_freq = None;
                } else {
                    println!("{why} (explicit-channel mode). Exiting.");
                    break;
                }
            }
            RunLiveResult::SkipFrequency => {
                skipped_freqs.insert(center_freq.round() as u64);
                if auto_scan {
                    explicit_freq = None;
                } else {
                    break;
                }
            }
            RunLiveResult::TuneToChannel(freq) => {
                explicit_freq = Some(freq);
                scan_standard = None;
            }
        }
    }
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════
//  SHARED LIVE PIPELINE
// ═══════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Copy)]
enum ViewerMode {
    /// Tune to a single channel; one window.
    SingleChannel,
    /// Wideband scan; detect and open windows for all active signals.
    Scan,
}

// Mirrors run_viewer_pipeline's wide signature — every knob the CLI
// surfaces lands here before being forwarded. Splitting would mean
// boxing a config struct, which is more ceremony than the call sites
// justify today.
#[allow(clippy::too_many_arguments)]
fn run_live(
    source: Box<dyn SdrSource>,
    sample_rate: f64,
    fm_deviation: f32,
    deemphasis_tau: f32,
    demod_kind: DemodKind,
    denoise: bool,
    denoise_model: String,
    center_freq: f64,
    forced_standard: Option<SignalType>,
    scan_hint: Option<SignalType>,
    debug: bool,
    temporal_window: usize,
    band: Option<&BandScan>,
    overrun_policy: OverrunPolicy,
    settle_after_tune: Duration,
) -> anyhow::Result<RunLiveResult> {
    let advice = Arc::new(NoOpDwell) as Arc<dyn DwellAdvice>;
    let config = SourceConfig {
        sample_rate_hz: sample_rate,
        channels_hz: vec![center_freq],
        dwell_min: Duration::from_secs(3600), // stay forever
        dwell_max: Duration::from_secs(3600),
        dwell_extension: Duration::ZERO,
    };

    let handle = source.start(config, advice)?;
    let sample_rate_u32 = sample_rate as u32;

    // Receive the first chunk for auto-detection
    println!("Waiting for first IQ chunk from SDR...");
    let mut first_packet = handle
        .receiver
        .recv()
        .map_err(|_| anyhow::anyhow!("SDR source closed before delivering any samples"))?;
    // A source that keeps sending the previous centre's samples after the
    // retune is acknowledged would have the standard auto-detected on the
    // wrong channel (measured: a clean NTSC link read "ambiguous" and
    // decoded as PAL). Discard until the backend's settle has passed and
    // start from the freshest packet.
    if !settle_after_tune.is_zero() {
        let t0 = Instant::now();
        while t0.elapsed() < settle_after_tune {
            let left = settle_after_tune.saturating_sub(t0.elapsed());
            match handle.receiver.recv_timeout(left) {
                Ok(p) => first_packet = p,
                Err(_) => break,
            }
        }
    }
    let actual_sample_rate = first_packet.sample_rate_hz as u32;
    let actual_center = first_packet.center_frequency_hz;
    if actual_sample_rate != sample_rate_u32 {
        println!(
            "Note: SDR actual sample rate {:.2} MSPS differs from requested {:.2} MSPS",
            actual_sample_rate as f64 / 1e6,
            sample_rate / 1e6
        );
    }
    let sample_rate_u32 = actual_sample_rate;

    println!(
        "Receiving: {:.2} MSPS at {:.3} MHz ({} samples in first chunk)",
        sample_rate_u32 as f64 / 1e6,
        actual_center / 1e6,
        first_packet.samples.len()
    );

    // Resolve the single channel at DC. (Wideband multi-signal scan
    // lives in the pre-tune sweeps — scan_band_for_channel and the USRP
    // scan loop — so by the time run_live starts, the centre is chosen.)
    let resolved_channels = {
        // Single channel at DC
        let sig_type = if let Some(st) = forced_standard {
            println!("Forced standard: {:?}", st);
            st
        } else {
            println!("Auto-detecting video standard...");
            let detector = AnalogFpvDetector::default();
            // Decide on one contiguous record long enough for the FFT to
            // tell the line rates apart (see `STANDARD_DETECT_RECORD`).
            // A packet flagged overrun follows a gap in the stream, so
            // the record restarts from it rather than splicing across
            // the hole. The packets consumed here are not fed to the
            // decoder; the pipeline starts from the last one.
            let want = (sample_rate_u32 as f64 * STANDARD_DETECT_RECORD.as_secs_f64()) as usize;
            let mut record: Vec<Complex<f32>> =
                Vec::with_capacity(want + first_packet.samples.len());
            record.extend_from_slice(&first_packet.samples);
            let t0 = Instant::now();
            while record.len() < want && t0.elapsed() < STANDARD_DETECT_PATIENCE {
                let left = STANDARD_DETECT_PATIENCE.saturating_sub(t0.elapsed());
                match handle.receiver.recv_timeout(left) {
                    Ok(p) => {
                        if p.overrun {
                            record.clear();
                        }
                        record.extend_from_slice(&p.samples);
                        first_packet = p;
                    }
                    Err(_) => break,
                }
            }
            let (detected, confidence) = detector.detect_sync_pulses(&record, sample_rate_u32);
            if debug {
                eprintln!(
                    "[DEBUG] standard decided on {:.1} ms of signal: {detected:?} {confidence:.2}",
                    record.len() as f64 / sample_rate_u32 as f64 * 1e3
                );
            }
            match detected {
                SignalType::AnalogVideoPal | SignalType::AnalogVideoNtsc => {
                    println!(
                        "  → Detected: {:?} ({:.0}% confidence)",
                        detected,
                        confidence * 100.0
                    );
                    detected
                }
                // Nothing found at the tuned centre. Prefer what the
                // sweep measured over a blind default: the sweep
                // integrates several looks and classified this very
                // signal, while this re-detect gets one centre and a
                // few chunks. It is a *fallback*, not an override — a
                // confident re-detect above still wins, so a sweep
                // verdict formed from mislabelled samples cannot
                // outrank direct evidence at the tuned frequency.
                SignalType::Unknown => match scan_hint {
                    Some(hint) => {
                        println!(
                            "  → No standard detected here; using the sweep's verdict: {hint:?}"
                        );
                        hint
                    }
                    None => {
                        println!("  → No standard detected, defaulting to PAL");
                        SignalType::AnalogVideoPal
                    }
                },
                // AnalogVideoUnknown (and any future variant):
                // definitely analog video, but the classifier
                // couldn't tell PAL from NTSC — resolve it in the
                // time domain rather than silently guessing.
                other => {
                    // Band-limit first: the capture can be tens of
                    // MHz wide, and demodulating the full span
                    // buries the video baseband in out-of-band
                    // noise, all but guaranteeing an inconclusive
                    // measurement. Same probe filter Scan mode
                    // uses, at 0 Hz offset (signal is at DC).
                    let mut probe_ddc =
                        StreamingDDC::new(0.0, sample_rate_u32, fm_deviation + LUMA_HEADROOM_HZ);
                    let probe_iq = probe_ddc.process(&record);
                    // Same fallback order as above: time-domain
                    // measurement first, then the sweep's verdict,
                    // then PAL.
                    concretize_standard(
                        other,
                        &probe_iq,
                        sample_rate_u32,
                        scan_hint.unwrap_or(SignalType::AnalogVideoPal),
                    )
                }
            }
        };
        // Signal is at DC (SDR tuned directly); size the decode
        // bandwidth so the worker's DDC cutoff lands at
        // `fm_deviation + LUMA_HEADROOM_HZ`. Passing bare
        // `fm_deviation * 2.0` made the worker's
        // `.min(dev + headroom).max(dev)` collapse to exactly
        // `fm_deviation`, so the documented sideband headroom was
        // unreachable on the main live decode path.
        vec![(sig_type, 0.0f32, (fm_deviation + LUMA_HEADROOM_HZ) * 2.0)]
    };

    let rf_center_freq = actual_center;

    // Feed the first chunk into the pipeline, then continue from the receiver
    let first_samples = first_packet.samples;
    let receiver = handle.receiver.clone();
    let handle_stop = handle.stop; // Take ownership of the stop closure

    let exit_reason = Arc::new(std::sync::atomic::AtomicU8::new(0));
    let tune_freq = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let frames_decoded = Arc::new(std::sync::atomic::AtomicU64::new(0));

    let result = run_viewer_pipeline(
        resolved_channels,
        sample_rate_u32,
        fm_deviation,
        deemphasis_tau,
        demod_kind,
        denoise,
        denoise_model,
        rf_center_freq,
        debug,
        temporal_window,
        band.cloned(),
        exit_reason.clone(),
        tune_freq,
        frames_decoded.clone(),
        move |channel_txs, exit_flag| {
            // Each consumer carries its own "the stream broke before
            // your next chunk" flag: a chunk dropped because one
            // window's worker fell behind is a hole for that decoder
            // and not for any other. Held across iterations, because
            // the chunk that would have carried the flag can itself be
            // the one that gets dropped.
            let mut active_txs: Vec<(_, bool)> =
                channel_txs.into_iter().map(|tx| (tx, false)).collect();
            // The lock check must integrate for the same reason the scan
            // does: one 65,536-sample packet is ~1-3 ms of signal, below
            // the length at which a single-shot detection finds anything
            // at all. Checked single-shot, a healthy -49 dBm decode read
            // confidence ~0 on every check, four "misses" accumulated in
            // ~2 s, and the viewer declared signal lost, rescanned, and
            // relocked in a loop — 44 times in four minutes on a live
            // Aaronia link before crashing.
            let mut lock_integ = SpectralIntegrator::new(4);
            let detector = viewer_detector();
            // Check cadence in wall time, not packets: a fixed 400-packet
            // interval was 0.43 s at 61.44 MSPS but 1.3 s at HackRF's
            // 20 MSPS, and the integrator's 4-batch memory plus the miss
            // threshold multiply that into the time a dead channel holds
            // the screen. ~2 Hz everywhere keeps loss detection near 3 s
            // on every backend.
            let check_every = ((sample_rate_u32 as f64 * 0.5) / first_samples.len().max(1) as f64)
                .round()
                .max(1.0) as usize;

            let mut packets = 0;
            let mut lock = LockState::default();
            let mut frames_at_last_check: u64 = 0;
            // The channel this run is holding. Detections elsewhere in
            // the capture are somebody else's signal.
            let tuned_carrier_hz = rf_center_freq;
            let mut dropped_chunks = 0u64;

            let mut overrun_count = 0;
            let mut last_overrun_clear = Instant::now();

            // Send first chunk. The stream has just started, so
            // nothing precedes it to be continuous with — the decoder's
            // carried state is empty at this point anyway.
            {
                let arc_chunk = Arc::new(first_samples);
                active_txs.retain(|(tx, _)| {
                    tx.send(IqChunk {
                        samples: Arc::clone(&arc_chunk),
                        discontinuous: false,
                    })
                    .is_ok()
                });
            }

            // Continue from SDR
            while !active_txs.is_empty() {
                // Bail promptly when the UI signals exit (Q/S/N/channel
                // change) instead of waiting for the worker→UI channels
                // to disconnect — minimises retune/rescan latency.
                if exit_flag.load(std::sync::atomic::Ordering::Relaxed) != 0 {
                    break;
                }
                match receiver.recv() {
                    Ok(packet) => {
                        packets += 1;

                        // Check overruns
                        if packet.overrun {
                            overrun_count += 1;
                            if debug {
                                eprintln!(
                                    "[DEBUG] overrun flagged on packet {packets} ({overrun_count} in the last minute)"
                                );
                            }
                            if overrun_policy == OverrunPolicy::StepDown && overrun_count >= 2 {
                                exit_flag.store(2, std::sync::atomic::Ordering::Relaxed);
                                break;
                            }
                        }
                        if last_overrun_clear.elapsed() >= Duration::from_secs(60) {
                            overrun_count = 0;
                            last_overrun_clear = Instant::now();
                        }

                        // Check signal lock periodically (approx twice a second)
                        if packets % check_every == 0 {
                            // The same sweep path that found the signal
                            // decides whether it is still there — not
                            // `detect_sync_pulses` on the raw full-rate
                            // packet, which sees ~16 video lines per
                            // 65,536-sample chunk at 61.44 MSPS and reads
                            // near-zero confidence on a healthy decode.
                            // Measured on a live -49 dBm Aaronia link:
                            // single-shot flapped lock→lost 44 times in
                            // four minutes, and integrating that same
                            // full-rate check still flapped (13 cycles in
                            // 30 s); the probe-based sweep detector holds.
                            let hits = detector.detect_from_iq_integrated(
                                &packet.samples,
                                packet.center_frequency_hz as u64,
                                sample_rate_u32,
                                &mut lock_integ,
                            );
                            // Two misses, not four: the integrator holds
                            // ~4 checks of history, so an empty result
                            // already means several consecutive looks
                            // found nothing. Stacking a 4-check threshold
                            // on top made a dead channel linger for up to
                            // 8 checks.
                            if debug {
                                let best = hits
                                    .iter()
                                    .max_by(|a, b| a.confidence.total_cmp(&b.confidence));
                                eprintln!(
                                    "[DEBUG] lock check at packet {packets}: {} hit(s){}",
                                    hits.len(),
                                    best.map(|h| format!(
                                        ", best {:?} conf {:.2} at {:.3} MHz rssi {:.1}",
                                        h.signal_type,
                                        h.confidence,
                                        h.frequency_hz as f64 / 1e6,
                                        h.rssi_dbm
                                    ))
                                    .unwrap_or_default()
                                );
                            }
                            // A hit is provisional. Two things promote
                            // it to "still locked": it has to be on the
                            // carrier we tuned to, and the decoder has
                            // to have produced a field since the last
                            // check. Neither was required before, so a
                            // transmitter elsewhere in the span kept
                            // this channel alive, and a carrier with
                            // horizontal sync but no vertical sync —
                            // 0.8 confidence, no frame ever — held the
                            // viewer indefinitely.
                            let on_carrier = hits.iter().any(|h| {
                                (h.frequency_hz as f64 - tuned_carrier_hz).abs()
                                    <= LOCK_CARRIER_TOLERANCE_HZ
                            });
                            let produced =
                                frames_decoded.load(std::sync::atomic::Ordering::Relaxed);
                            let decoding = produced > frames_at_last_check;
                            frames_at_last_check = produced;

                            if lock.observe(on_carrier, decoding) {
                                exit_flag.store(1, std::sync::atomic::Ordering::Relaxed);
                                break;
                            }
                        }

                        // A packet flagged overrun follows a gap in
                        // the stream: its first sample is not adjacent
                        // to the last one the decoder saw.
                        // A packet flagged overrun follows a gap in the
                        // stream: its first sample is not adjacent to
                        // the last one any decoder saw.
                        let overran = packet.overrun;
                        let arc_chunk = Arc::new(packet.samples);
                        active_txs.retain_mut(|(tx, pending)| {
                            let breaks = *pending || overran;
                            match tx.try_send(IqChunk {
                                samples: Arc::clone(&arc_chunk),
                                discontinuous: breaks,
                            }) {
                                Ok(_) => {
                                    // Handed over — this consumer is
                                    // told, so stop telling it.
                                    *pending = false;
                                    true
                                }
                                // Worker behind: drop this chunk but keep
                                // the channel. Dropping raw IQ breaks DDC
                                // continuity, so remember that this
                                // consumer's stream now has a hole in it
                                // and count the cost for debug.
                                Err(mpsc::TrySendError::Full(_)) => {
                                    *pending = true;
                                    dropped_chunks += 1;
                                    true
                                }
                                Err(mpsc::TrySendError::Disconnected(_)) => false, // UI closed
                            }
                        });
                        if debug && packets % check_every == 0 && dropped_chunks > 0 {
                            eprintln!(
                                "[DEBUG] dropped {} IQ chunks so far (decode worker behind)",
                                dropped_chunks
                            );
                        }
                    }
                    Err(_) => break, // SDR disconnected
                }
            }
        },
    );

    // Shut down the USRP capture thread cleanly before dropping the handle.
    // Without this, the capture thread races with UHD's global usrp_ptr map
    // destructor during process exit, causing a double-free (SIGABRT).
    (handle_stop)();
    std::thread::sleep(Duration::from_millis(100));

    result
}

// ═══════════════════════════════════════════════════════════════════
//  SHARED VIEWER PIPELINE — works for both file and live SDR
// ═══════════════════════════════════════════════════════════════════

// Touching every parameter on every call site of this fn would be more
// disruptive than splitting it into separate helpers — the shape of the
// args here mirrors the shape of the live/file dispatch and the
// signature changes whenever a new pipeline knob arrives.
#[allow(clippy::too_many_arguments)]
fn run_viewer_pipeline<F>(
    resolved_channels: Vec<(SignalType, f32, f32)>,
    sample_rate: u32,
    fm_deviation: f32,
    deemphasis_tau: f32,
    demod_kind: DemodKind,
    denoise: bool,
    denoise_model: String,
    rf_center_freq: f64,
    debug: bool,
    temporal_window: usize,
    band: Option<BandScan>,
    exit_reason: Arc<std::sync::atomic::AtomicU8>,
    tune_freq: Arc<std::sync::atomic::AtomicU64>,
    // Fields the decoder has actually reconstructed. The reader thread
    // watches this to tell a detection apart from decodable video.
    frames_decoded: Arc<std::sync::atomic::AtomicU64>,
    reader_fn: F,
) -> anyhow::Result<RunLiveResult>
where
    F: FnOnce(Vec<mpsc::SyncSender<IqChunk>>, Arc<std::sync::atomic::AtomicU8>) + Send + 'static,
{
    let mut channel_txs = Vec::new();
    let mut frame_rxs = Vec::new();
    let mut frame_slots: Vec<FrameSlot> = Vec::new();
    let mut recycle_txs = Vec::new();
    let mut windows = Vec::new();
    let mut display_buffers = Vec::new();

    // Live denoiser toggle, shared UI thread → every decode worker. The
    // reconstructors live in per-channel threads, so the `D` hotkey flips
    // this flag and each worker picks it up on its next field rather than
    // the UI reaching into another thread's state.
    let denoise_on = Arc::new(std::sync::atomic::AtomicBool::new(denoise));

    for (sig_type, freq_offset, bandwidth_hz) in resolved_channels {
        // Both are only read by the `neural-vsr` code paths; without that
        // feature the worker has no denoiser to toggle.
        #[cfg_attr(not(feature = "neural-vsr"), allow(unused_variables))]
        let denoise_on = Arc::clone(&denoise_on);
        #[cfg_attr(not(feature = "neural-vsr"), allow(unused_variables))]
        let denoise_model = denoise_model.clone();
        let (iq_tx, iq_rx) = mpsc::sync_channel::<IqChunk>(10);
        // The channel is now only a wakeup, and the thing that still
        // reports the UI going away: dropping the receiver is how the
        // UI signals shutdown, and a mutex has no equivalent.
        let (frame_tx, frame_rx) = mpsc::sync_channel::<()>(1);
        let frame_slot: FrameSlot = Arc::new(std::sync::Mutex::new(None));
        let producer_slot = Arc::clone(&frame_slot);
        let worker_frames = Arc::clone(&frames_decoded);
        // Frame-buffer recycle pool: the UI hands spent frame buffers
        // back here so the decode worker can refill them in place
        // (`reconstruct_frame_into`) instead of allocating + zeroing a
        // fresh ~1.4 MB Vec every field. Both ends use non-blocking
        // try_*; on miss the worker just allocates and on a full return
        // channel the UI drops the buffer — so there's no deadlock path.
        let (recycle_tx, recycle_rx) = mpsc::sync_channel::<Vec<u32>>(3);

        let is_pal = sig_type == SignalType::AnalogVideoPal;

        // Everything after the down-converter runs at the working rate,
        // not the capture rate. Decided here rather than in the worker
        // because the reconstructor is built here — its geometry sizes
        // the window below.
        //
        // Floor the cutoff at `fm_deviation`: in scan mode
        // `bandwidth_hz` comes from the detector, and an implausibly
        // small detected bandwidth would otherwise collapse the FIR
        // passband and render a black/garbage frame with no clue why.
        let ddc_cutoff = (bandwidth_hz / 2.0)
            .min(fm_deviation + LUMA_HEADROOM_HZ)
            .max(fm_deviation);
        let mut decim = decode_decimation(sample_rate, ddc_cutoff);
        // The PLL cannot track a 5 MHz deviation much below 20 MSPS —
        // its loop-bandwidth stability clamp caps ωₙ at 0.5 rad/sample,
        // i.e. fs/4π (analog crate DESIGN.md §11). Asking for it
        // explicitly therefore caps how far we may decimate, at the
        // cost of the CPU the decimation would have saved.
        if matches!(demod_kind, DemodKind::Pll) {
            let cap = (sample_rate / PLL_AUTO_MIN_SAMPLE_RATE_HZ).max(1) as usize;
            if decim > cap {
                println!(
                    "  → --demod pll holds the decode rate at {:.2} MSPS (the loop cannot \
                     track this deviation below ~20 MSPS); it may not keep up. Drop the \
                     flag to decode at {:.2} MSPS.",
                    (sample_rate / cap as u32) as f64 / 1e6,
                    (sample_rate / decim as u32) as f64 / 1e6
                );
                decim = cap;
            }
        }
        let work_rate = sample_rate / decim as u32;
        let use_pll = demod_kind.use_pll(work_rate, fm_deviation);
        if decim > 1 {
            println!(
                "  → Decoding at {:.2} MSPS ({:.2} MSPS capture / {})",
                work_rate as f64 / 1e6,
                sample_rate as f64 / 1e6,
                decim
            );
        }
        println!(
            "  → Demodulator: {} ({:.2} MSPS)",
            if use_pll { "PLL" } else { "discriminator" },
            work_rate as f64 / 1e6
        );

        #[allow(unused_mut)]
        let mut reconstructor = FrameReconstructor::new(work_rate, is_pal, fm_deviation, debug)
            .with_temporal_window(temporal_window);
        // Load the denoiser once per channel (opening an ONNX session is
        // expensive, and the `D` hotkey must toggle instantly). It is
        // parked in `stashed_restorer` while disabled and moved back into
        // the reconstructor when enabled, so no reload happens on toggle.
        #[cfg(feature = "neural-vsr")]
        let mut stashed_restorer = {
            reconstructor = reconstructor.with_neural_restorer(&denoise_model, true);
            if reconstructor.neural_restorer.is_none() {
                eprintln!(
                    "Denoiser unavailable: could not load {denoise_model} — continuing without it."
                );
            }
            // Start parked unless the operator asked for it up front.
            if denoise_on.load(std::sync::atomic::Ordering::Relaxed) {
                None
            } else {
                reconstructor.neural_restorer.take()
            }
        };
        let width = reconstructor.width;
        let height = reconstructor.height;

        let type_name = if is_pal { "PAL" } else { "NTSC" };
        let absolute_freq_mhz = (rf_center_freq + freq_offset as f64) / 1_000_000.0;
        let channel_name = get_fpv_channel_name(absolute_freq_mhz);
        let window_title = if let Some(ch) = channel_name {
            format!("{} · Channel {}", type_name, ch)
        } else {
            format!("{} · {:.2} MHz", type_name, absolute_freq_mhz)
        };
        // The band panel gets its own rows under the picture rather than
        // being painted over the bottom of it. The window is created
        // tall enough for both; the buffer is sized for the largest
        // panel so cycling `B` never reallocates mid-frame.
        let panel_h_now = band
            .as_ref()
            .map(|b| b.below_height(PanelMode::load()))
            .unwrap_or(0);
        let panel_h_max = band
            .as_ref()
            .map(|b| b.below_height(PanelMode::Full))
            .unwrap_or(0);
        println!(
            "  → Window: {} ({}×{}{})",
            window_title,
            width,
            height + panel_h_now,
            if panel_h_now > 0 {
                format!(", {height} picture + {panel_h_now} band")
            } else {
                String::new()
            }
        );
        let window = Window::new(
            &window_title,
            width,
            height + panel_h_now,
            WindowOptions {
                // The composite changes height when `B` cycles the
                // panel and minifb cannot resize a window, so let it
                // letterbox rather than distort the picture.
                resize: true,
                scale_mode: minifb::ScaleMode::AspectRatioStretch,
                ..WindowOptions::default()
            },
        )?;

        windows.push((window, width, height, is_pal, absolute_freq_mhz));
        display_buffers.push(vec![0u32; width * (height + panel_h_max)]);
        channel_txs.push(iq_tx);
        frame_rxs.push(frame_rx);
        frame_slots.push(frame_slot);
        recycle_txs.push(recycle_tx);

        // Per-channel snapshot encoder thread. PNG encoding is tens of
        // ms; running it inline in the decode loop stalls decoding,
        // which drops IQ chunks (breaking DDC continuity) right when the
        // snapshot is taken — so the saved image would misrepresent the
        // decode. The decode loop hands (path, rgb, w, h) here instead.
        let (snap_tx, snap_rx) = mpsc::channel::<(String, Vec<u8>, u32, u32)>();
        thread::spawn(move || {
            while let Ok((path, rgb, w, h)) = snap_rx.recv() {
                match image::save_buffer(&path, &rgb, w, h, image::ColorType::Rgb8) {
                    Ok(()) => eprintln!("[DEBUG] Saved {}", path),
                    Err(e) => eprintln!("Failed to save {}: {}", path, e),
                }
            }
        });

        thread::spawn(move || {
            // The down-converter is the one stage that stays at the
            // capture rate: it reads every input sample and emits one
            // per stride. A decimating cut needs the longer filter (see
            // `DECODE_FIR_TAPS`); at stride 1 nothing folds, so keep the
            // default and leave that path byte-identical.
            let mut ddc = if decim > 1 {
                StreamingDDC::with_taps(freq_offset, sample_rate, ddc_cutoff, DECODE_FIR_TAPS)
            } else {
                StreamingDDC::new(freq_offset, sample_rate, ddc_cutoff)
            };
            let mut shifted_iq: Vec<Complex<f32>> = Vec::new();
            // Last DDC output sample of the previous chunk. `fm_demod`
            // is stateless and returns n−1 samples for n inputs, so
            // without this carry the phase transition across every
            // chunk boundary is silently dropped (~15 ppm timebase
            // slip). Seeding each chunk with the previous chunk's final
            // sample makes the demod stream gapless.
            let mut iq_carry: Option<Complex<f32>> = None;
            // Reused per-chunk demod scratch (fm_demod_into) — a fresh
            // ~256 KB Vec per chunk was pure allocator churn.
            let mut demod_scratch: Vec<f32> = Vec::new();
            // Stream-side deemphasis, applied exactly once per sample,
            // before anything enters the persistent demod buffer — the
            // placement the library documents (a stateful filter inside
            // the reconstructor would re-filter the re-read tail).
            let mut deemph = (deemphasis_tau.is_finite() && deemphasis_tau > 0.0)
                .then(|| Deemphasis::new(work_rate, deemphasis_tau));
            // PLL demodulator (--demod pll, or auto's >= 25 MSPS
            // crossover): threshold extension — see DemodKind's doc
            // for the measured numbers and the low-rate caveat. 1 MHz
            // loop bandwidth per the weak_signal_sweep tuning.
            // 1 MHz loop bandwidth is an absolute figure and does not
            // scale with the rate; `use_pll` was resolved against the
            // working rate in the setup loop.
            let mut pll =
                use_pll.then(|| PllFmDemod::new(work_rate, PLL_LOOP_BW_HZ, fm_deviation * 1.2));
            let mut demod_buffer: Vec<f32> = Vec::new();
            // A field plus its blanking, the same bound the
            // reconstructor's own fallback path uses before it will
            // commit to a field. ~13 ms at 15.36 MSPS, well under a
            // field period, so gating on it costs no latency.
            let min_samples_per_field = {
                let line_rate = if is_pal { 15_625.0f32 } else { 15_734.0 };
                let lines = if is_pal { 288 } else { 240 } + 22;
                ((work_rate as f32 / line_rate) * lines as f32) as usize
            };
            // ~33 ms consumed before compacting, ~83 ms of live region
            // before the valve fires, ~16 ms kept when it does.
            let compact_after = work_rate as usize / 30;
            let live_region_cap = work_rate as usize / 12;
            let keep_on_skip = work_rate as usize / 60;
            // Cursor into `demod_buffer`: live (unconsumed) samples are
            // `demod_buffer[demod_start..]`. Advancing a cursor instead
            // of `drain(0..consumed)` per field avoids an O(n) prefix
            // memmove every frame; we compact in one shot once the
            // consumed prefix grows large.
            let mut demod_start = 0usize;
            // Reused frame buffer; refilled in place by
            // `reconstruct_frame_into` and swapped for a recycled/fresh
            // one each time a completed field is shipped to the UI.
            let mut frame_buf: Vec<u32> = vec![0u32; width * height];
            let mut frame_count = 0u64;
            // One-shot guard for the PAL demod-range probe. Gating on
            // `frame_count == 0` instead floods the console: that stays
            // true for every chunk when the field never locks.
            let mut pal_debug_done = false;

            while let Ok(iq_chunk) = iq_rx.recv() {
                if iq_chunk.discontinuous {
                    // The stream broke before this chunk, so every
                    // piece of state that spans a chunk boundary is now
                    // describing a moment that no longer connects to
                    // this one.
                    //
                    // The carry sample is the sharpest of them: the
                    // discriminator takes the phase difference between
                    // it and the first new sample, and across a gap
                    // that difference is not a frequency — it is
                    // whatever the signal did while nobody was looking,
                    // rendered as one enormous spike. That reads
                    // downstream as a sync edge, which is why a dropped
                    // chunk could cost lock rather than a glitch.
                    iq_carry = None;
                    ddc.reset();
                    if let Some(p) = pll.as_mut() {
                        p.reset();
                    }
                    if let Some(d) = deemph.as_mut() {
                        d.reset();
                    }
                    // Drop the partially decoded field: it now has the
                    // far side of a gap spliced onto it.
                    demod_buffer.clear();
                    demod_start = 0;
                    // Temporal denoise, dropout repair and the time-base
                    // reference all assume consecutive views of one
                    // scene. Holding the last good picture while sync
                    // reacquires is better than blending across the
                    // hole.
                    reconstructor.forget_history();
                    #[cfg(feature = "neural-vsr")]
                    {
                        reconstructor.hidden_state = None;
                    }
                }
                shifted_iq.clear();
                // The carry sample only serves the stateless
                // discriminator; the PLL's loop state already spans
                // chunk boundaries, and a duplicated sample would be a
                // phase glitch to it.
                if pll.is_none()
                    && let Some(prev) = iq_carry
                {
                    shifted_iq.push(prev);
                }
                ddc.process_into_decimated(&iq_chunk.samples, &mut shifted_iq, decim);
                iq_carry = shifted_iq.last().copied();
                match pll.as_mut() {
                    Some(p) => p.process_into(&shifted_iq, &mut demod_scratch),
                    None => fm_demod_into(&shifted_iq, &mut demod_scratch),
                }
                if let Some(d) = deemph.as_mut() {
                    d.process_in_place(&mut demod_scratch);
                }

                // Live denoiser toggle + link-quality conditioning. The
                // CNR is measured on the DDC-filtered channel, not the
                // raw wideband chunk, so it reflects THIS signal's link
                // rather than the whole capture's noise floor.
                #[cfg(feature = "neural-vsr")]
                {
                    let want = denoise_on.load(std::sync::atomic::Ordering::Relaxed);
                    let have = reconstructor.neural_restorer.is_some();
                    if want && !have {
                        reconstructor.neural_restorer = stashed_restorer.take();
                        // Drop temporal context accumulated before the
                        // gap — blending across a disabled period would
                        // mix in stale fields.
                        reconstructor.hidden_state = None;
                    } else if !want && have {
                        stashed_restorer = reconstructor.neural_restorer.take();
                    }
                    if reconstructor.neural_restorer.is_some()
                        && let Some(cnr) = estimate_cnr_db(&shifted_iq)
                    {
                        reconstructor.set_neural_noise_level(cnr);
                    }
                }

                if debug && is_pal && !pal_debug_done && !demod_scratch.is_empty() {
                    let min = demod_scratch.iter().cloned().fold(f32::INFINITY, f32::min);
                    let max = demod_scratch
                        .iter()
                        .cloned()
                        .fold(f32::NEG_INFINITY, f32::max);
                    let mean = demod_scratch.iter().sum::<f32>() / demod_scratch.len() as f32;
                    eprintln!(
                        "[PAL DEBUG] first demod chunk: min={:.4} max={:.4} mean={:.4}",
                        min, max, mean
                    );
                    pal_debug_done = true;
                }

                demod_buffer.extend_from_slice(&demod_scratch);

                // None ends the loop (no full field buffered yet);
                // `frame_buf` is left untouched on the None paths (they
                // return before writing it), so it's safe to keep for the
                // next chunk.
                //
                // Only ask once a field could actually be in there.
                // `reconstruct_frame_into` runs three full-slice passes
                // and two full-length allocations *before* the length
                // check that would reject a short buffer, so calling it
                // on every chunk means most calls do all that work and
                // return None. It costs more the more chunks a second
                // brings: measured, reconstruction ran at 0.34 CPU-s/s
                // fed 65,536-sample chunks and 0.75 fed the
                // 16,384-sample chunks a 4x stride produces.
                while demod_buffer.len() - demod_start >= min_samples_per_field {
                    let Some(consumed) = reconstructor
                        .reconstruct_frame_into(&demod_buffer[demod_start..], &mut frame_buf)
                    else {
                        break;
                    };
                    frame_count += 1;

                    // Rate-limited debug telemetry: print per-frame
                    // for the first 3 frames (which also get saved
                    // as PNGs for visual inspection), then every
                    // 30 frames thereafter (≈ once every 500 ms at
                    // NTSC's 60-field rate). The 3-frame auto-exit
                    // was dropped after the chroma-PLL wind-up bug
                    // — that failure mode develops over seconds of
                    // continuous capture and was invisible in a
                    // 3-frame snapshot, so debug mode now keeps
                    // running and lets you `Ctrl-C` (or `timeout`)
                    // when you've seen enough.
                    let metrics_due = debug && (frame_count <= 3 || frame_count.is_multiple_of(30));
                    if metrics_due {
                        // Envelope CNR (link-quality meter) + PLL lock
                        // telemetry when applicable. Measured on the
                        // DDC-filtered channel (`shifted_iq`), not the
                        // raw capture — a wideband chunk's envelope
                        // reflects the whole band's floor, not this
                        // channel's link.
                        let cnr = estimate_cnr_db(&shifted_iq)
                            .map(|v| format!("{v:.1} dB"))
                            .unwrap_or_else(|| "n/a".into());
                        let lock = pll
                            .as_ref()
                            .map(|p| format!(" | PLLerr: {:.2} rad", p.phase_error_rms()))
                            .unwrap_or_default();
                        println!("[DEBUG LINK] CNRest: {cnr}{lock}");
                        // New v0.4.37 telemetry fields:
                        //  - SyncQ : per-field MAD-rejection-pass
                        //    rate. 1.00 = perfect; <0.5 = dropout
                        //    repair fires.
                        //  - Y_avg : mean Y amplitude post-notch.
                        //    Sudden drops mark transmitter going
                        //    out of range / antenna blockage.
                        //  - HistD : how full the temporal history
                        //    is (1 → window). Denoise benefit
                        //    scales with √HistD.
                        println!(
                            "[DEBUG METRICS] Frame {} | Std: {:?} | LinePer: {:.2}s | SyncQ: {:.2} | Y_avg: {:+.3} | HistD: {}",
                            frame_count,
                            reconstructor.video_standard(),
                            reconstructor.line_period_samples(),
                            reconstructor.latest_sync_quality(),
                            reconstructor.latest_mean_amplitude(),
                            reconstructor.history_depth(),
                        );
                    }

                    // Snapshot the first 3 frames (startup transient,
                    // before the line-period history + temporal denoise
                    // settle) and frames 30-32 (steady state, ≈ 0.5 s
                    // in) for visual inspection. The cheap RGB conversion
                    // runs here; the PNG encode is handed to the snapshot
                    // thread so it never stalls the decode loop.
                    if debug && (frame_count <= 3 || (30..=32).contains(&frame_count)) {
                        // Frequency in the filename: in Scan mode several
                        // per-channel workers snapshot concurrently, and
                        // an undiscriminated `fpv_frame_{n}.png` had them
                        // racing to overwrite each other's files.
                        let path =
                            format!("fpv_frame_{:.0}MHz_{}.png", absolute_freq_mhz, frame_count);
                        let mut rgb_buf = vec![0u8; width * height * 3];
                        for (i, &pixel) in frame_buf.iter().enumerate() {
                            rgb_buf[i * 3] = ((pixel >> 16) & 0xFF) as u8;
                            rgb_buf[i * 3 + 1] = ((pixel >> 8) & 0xFF) as u8;
                            rgb_buf[i * 3 + 2] = (pixel & 0xFF) as u8;
                        }
                        let _ = snap_tx.send((path, rgb_buf, width as u32, height as u32));
                    }

                    demod_start += consumed;
                    // Ship the just-filled frame to the UI, but never
                    // wait for it.
                    //
                    // A blocking send here made the display the
                    // decoder's pacer: two frames of queue, and a UI
                    // that stalls stalls the decoder, which stops
                    // draining the IQ queue, which is ten packets —
                    // 10.7 ms of signal at 61.44 MSPS — after which the
                    // reader discards live IQ. A dropped *frame* costs
                    // one repeated picture; dropped *IQ* costs sync.
                    //
                    // So the frame goes into a slot the producer
                    // overwrites: the UI always sees the newest one,
                    // and whatever it had not taken yet comes back here
                    // to be reused, which keeps the steady state
                    // allocation-free even when the recycle pool has
                    // run dry.
                    // A field came out. This is the only evidence that
                    // the signal is decodable rather than merely
                    // present, and the lock check upstream needs it.
                    worker_frames.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let displaced =
                        lock_slot(&producer_slot).replace(std::mem::take(&mut frame_buf));
                    frame_buf = displaced
                        .or_else(|| recycle_rx.try_recv().ok())
                        .unwrap_or_else(|| vec![0u32; width * height]);
                    // Wake the UI. `Full` just means it has not looked
                    // since the last frame, which is exactly the case
                    // the slot already handled.
                    match frame_tx.try_send(()) {
                        Ok(()) | Err(mpsc::TrySendError::Full(())) => {}
                        Err(mpsc::TrySendError::Disconnected(())) => return,
                    }
                }

                // Amortised compaction: drop the consumed prefix in a
                // single memmove once it's grown past the threshold,
                // rather than once per field. Sized in time rather than
                // samples, so decimating doesn't quietly buy four times
                // the buffering.
                if demod_start > compact_after {
                    demod_buffer.drain(0..demod_start);
                    demod_start = 0;
                }
                // Safety valve: if reconstruction can't keep up and the
                // live region balloons, skip ahead (corrupts one field's
                // sync, but bounds memory) instead of growing unbounded.
                if demod_buffer.len() - demod_start > live_region_cap {
                    demod_start = demod_buffer.len().saturating_sub(keep_on_skip);
                }
            }
        });
    }

    if channel_txs.is_empty() {
        println!("No analog video signals detected!");
        std::process::exit(0);
    }

    // Spawn reader thread (file or SDR)
    let thread_exit_flag = exit_reason.clone();
    thread::spawn(move || reader_fn(channel_txs, thread_exit_flag));
    // Main UI Loop
    let mut ch_input = ChannelInputState::Idle;
    let mut denoise_toggle_requested = false;
    // The band panel rides under the picture, marking the frequency this
    // window is locked to. `B` cycles its size; the choice is process-wide
    // so it survives a relock.
    let mut band = band;
    if let Some(b) = band.as_mut() {
        b.set_lock(Some(rf_center_freq));
    }
    let mut panel_mode = PanelMode::load();
    let mut panel_toggle_requested = false;
    let mut ch_input_flash_until: Option<std::time::Instant> = None;
    let mut ch_input_flash_msg = String::new();
    while !windows.is_empty() {
        let reason = exit_reason.load(std::sync::atomic::Ordering::Relaxed);
        if reason != 0 {
            windows.clear();
            frame_rxs.clear();
            frame_slots.clear();
            display_buffers.clear();
            break;
        }

        let mut i = 0;
        while i < windows.len() {
            let (window, width, height, is_pal, absolute_freq_mhz) = &mut windows[i];

            // Q / Escape / window close → quit the app. Gated on the
            // channel-input state machine being idle, and edge-triggered
            // (`get_keys_pressed`, which is idempotent within a frame):
            // while typing a channel, Escape must *cancel the input* (the
            // state machine below handles that) and Q is a candidate
            // keystroke, so neither may quit the app. Edge-triggering
            // also stops the Escape that cancelled input from still being
            // held down on the next 5 ms iteration and quitting anyway.
            let pressed = window.get_keys_pressed(minifb::KeyRepeat::No);
            let idle = matches!(ch_input, ChannelInputState::Idle);
            if !window.is_open()
                || (idle && (pressed.contains(&Key::Escape) || pressed.contains(&Key::Q)))
            {
                windows.remove(i);
                frame_rxs.remove(i);
                frame_slots.remove(i);
                recycle_txs.remove(i);
                display_buffers.remove(i);
                continue;
            }

            // S → skip this frequency (blacklist it), resume scanning
            if matches!(ch_input, ChannelInputState::Idle) && window.is_key_down(Key::S) {
                exit_reason.store(3, std::sync::atomic::Ordering::Relaxed);
                windows.clear();
                frame_rxs.clear();
                frame_slots.clear();
                display_buffers.clear();
                break;
            }

            // D → toggle the neural denoiser. Only *record* the request
            // here: this block runs once per open window, so acting on it
            // inline would toggle N times for one keypress — with two
            // windows that is a silent no-op. Applied once after the
            // per-window loop.
            if idle && pressed.contains(&Key::D) {
                denoise_toggle_requested = true;
            }

            // B → cycle the band panel (strip, full, off). Recorded like
            // D and applied once after the per-window loop.
            if idle && band.is_some() && pressed.contains(&Key::B) {
                panel_toggle_requested = true;
            }

            // N → find next channel (resume scanning without blacklisting)
            if matches!(ch_input, ChannelInputState::Idle) && window.is_key_down(Key::N) {
                exit_reason.store(4, std::sync::atomic::Ordering::Relaxed);
                windows.clear();
                frame_rxs.clear();
                frame_slots.clear();
                display_buffers.clear();
                break;
            }

            // C → enter channel input mode
            if matches!(ch_input, ChannelInputState::Idle) {
                let keys = window.get_keys_pressed(minifb::KeyRepeat::No);
                if keys.contains(&Key::C) {
                    ch_input = ChannelInputState::WaitingFirst;
                }
            }

            // Channel input state machine
            match ch_input.clone() {
                ChannelInputState::WaitingFirst => {
                    let keys = window.get_keys_pressed(minifb::KeyRepeat::No);
                    for k in keys {
                        if k == Key::C {
                            continue;
                        } // ignore the C that started us
                        if k == Key::Escape {
                            ch_input = ChannelInputState::Idle;
                            break;
                        }
                        if let Some(c) = key_to_char(k)
                            && c.is_ascii_alphabetic()
                        {
                            ch_input = ChannelInputState::WaitingSecond(c);
                            break;
                        }
                    }
                }
                ChannelInputState::WaitingSecond(first) => {
                    let keys = window.get_keys_pressed(minifb::KeyRepeat::No);
                    for k in keys {
                        if k == Key::Escape {
                            ch_input = ChannelInputState::Idle;
                            break;
                        }
                        if let Some(c) = key_to_char(k) {
                            let ch_name = format!("{}{}", first, c);
                            if let Some(freq) = lookup_channel_by_name(&ch_name) {
                                // Found it! Store the frequency and signal exit.
                                tune_freq.store(freq, std::sync::atomic::Ordering::Relaxed);
                                exit_reason.store(5, std::sync::atomic::Ordering::Relaxed);
                                break;
                            } else {
                                // Invalid channel name, flash error and reset
                                ch_input_flash_msg = format!("Unknown channel: {}", ch_name);
                                ch_input_flash_until =
                                    Some(std::time::Instant::now() + Duration::from_secs(2));
                                ch_input = ChannelInputState::Idle;
                                break;
                            }
                        }
                    }
                }
                ChannelInputState::Idle => {}
            }

            // Drain the wakeup tokens — the frame itself is in the
            // slot, and there is only ever the newest one there.
            while frame_rxs[i].try_recv().is_ok() {}
            let newest = lock_slot(&frame_slots[i]).take();
            if let Some(frame_u32) = newest {
                // The buffer is taller than the picture now — the band
                // panel owns the rows underneath — so the frame fills
                // the picture region rather than the whole buffer.
                let picture_len = *width * *height;
                if frame_u32.len() == picture_len && display_buffers[i].len() >= picture_len {
                    display_buffers[i][..picture_len].copy_from_slice(&frame_u32);
                    // Hand the buffer back to the worker's recycle pool
                    // (drop it if the pool is full — the worker will just
                    // allocate). Done before drawing the overlays, which
                    // operate on `display_buffers[i]`.
                    let _ = recycle_txs[i].try_send(frame_u32);

                    let format_str = if *is_pal { "PAL" } else { "NTSC" };
                    let channel_name = get_fpv_channel_name(*absolute_freq_mhz);
                    // Show the denoiser state so the operator can see what
                    // `D` did without hunting the console.
                    let dn = if denoise_on.load(std::sync::atomic::Ordering::Relaxed) {
                        " DN"
                    } else {
                        ""
                    };
                    let display_text = if let Some(ch) = channel_name {
                        format!("{} · Channel {} [BW]{}", format_str, ch, dn)
                    } else {
                        format!("{} · {:.2} MHz [BW]{}", format_str, absolute_freq_mhz, dn)
                    };

                    draw_text_with_bg(
                        &mut display_buffers[i],
                        *width,
                        *height,
                        10,
                        10,
                        &display_text,
                        0xff00ff00,
                        0xff000000,
                    );

                    // Band panel in its own rows *below* the picture.
                    // Painting it over the bottom of the frame cost those
                    // rows of video, which is the whole reason it moved.
                    // Clamp to the rows the buffer actually has spare
                    // beneath the picture, so a panel mode taller than
                    // what was allocated cannot index past the end.
                    let spare_rows =
                        (display_buffers[i].len() / (*width).max(1)).saturating_sub(*height);
                    let panel_h = band
                        .as_ref()
                        .map(|b| b.below_height(panel_mode))
                        .unwrap_or(0)
                        .min(spare_rows);
                    if panel_h > 0
                        && let Some(b) = band.as_ref()
                    {
                        let picture_len = *width * *height;
                        let panel =
                            &mut display_buffers[i][picture_len..picture_len + *width * panel_h];
                        b.draw_below(panel, *width, panel_h, panel_mode);
                    }

                    // Keybinding hints at the bottom of the picture. They
                    // no longer have to dodge the panel. `B` is only
                    // offered when there is a panel to cycle.
                    let hint_y = (*height).saturating_sub(20);
                    let hint = if band.is_some() {
                        "Q:Quit  S:Skip  N:Next  C:Channel  D:Denoise  B:Band"
                    } else {
                        "Q:Quit  S:Skip  N:Next  C:Channel  D:Denoise"
                    };
                    draw_text_with_bg(
                        &mut display_buffers[i],
                        *width,
                        *height,
                        10,
                        hint_y,
                        hint,
                        0xffaaaaaa,
                        0x80000000,
                    );

                    // Channel input prompt overlay
                    match &ch_input {
                        ChannelInputState::WaitingFirst => {
                            draw_text_with_bg(
                                &mut display_buffers[i],
                                *width,
                                *height,
                                10,
                                30,
                                "CH: __",
                                0xffffff00,
                                0xcc000000,
                            );
                        }
                        ChannelInputState::WaitingSecond(first) => {
                            draw_text_with_bg(
                                &mut display_buffers[i],
                                *width,
                                *height,
                                10,
                                30,
                                &format!("CH: {}_", first),
                                0xffffff00,
                                0xcc000000,
                            );
                        }
                        ChannelInputState::Idle => {}
                    }

                    // Flash message for invalid channel
                    if let Some(until) = ch_input_flash_until {
                        if std::time::Instant::now() < until {
                            draw_text_with_bg(
                                &mut display_buffers[i],
                                *width,
                                *height,
                                10,
                                30,
                                &ch_input_flash_msg,
                                0xffff4444,
                                0xcc000000,
                            );
                        } else {
                            ch_input_flash_until = None;
                        }
                    }

                    let composite_h = *height + panel_h;
                    let _ = window.update_with_buffer(
                        &display_buffers[i][..*width * composite_h],
                        *width,
                        composite_h,
                    );
                } else if debug {
                    eprintln!(
                        "[DEBUG] dropped frame on window {}: size {} != picture {}",
                        i,
                        frame_u32.len(),
                        *width * *height
                    );
                }
            } else {
                window.update();
            }

            i += 1;
        }

        // One keypress → one toggle, regardless of how many windows are
        // open. Workers pick the new state up on their next field; the
        // model stays loaded either way, so the switch is instant.
        if denoise_toggle_requested {
            denoise_toggle_requested = false;
            let now = !denoise_on.load(std::sync::atomic::Ordering::Relaxed);
            denoise_on.store(now, std::sync::atomic::Ordering::Relaxed);
            if cfg!(feature = "neural-vsr") {
                println!("Denoiser {}", if now { "ON" } else { "OFF" });
            } else {
                println!(
                    "Denoiser unavailable: rebuild with `--features neural-vsr` to enable it."
                );
            }
        }

        if panel_toggle_requested {
            panel_toggle_requested = false;
            panel_mode = panel_mode.next();
            panel_mode.store();
            println!("Band panel: {}", panel_mode.label());
        }

        thread::sleep(Duration::from_millis(5));
    }

    let final_reason = exit_reason.load(std::sync::atomic::Ordering::Relaxed);
    match final_reason {
        1 => Ok(RunLiveResult::SignalLost),
        2 => Ok(RunLiveResult::TooManyOverruns),
        3 => Ok(RunLiveResult::SkipFrequency),
        4 => Ok(RunLiveResult::NextChannel),
        5 => {
            let freq = tune_freq.load(std::sync::atomic::Ordering::Relaxed);
            Ok(RunLiveResult::TuneToChannel(freq as f64))
        }
        _ => Ok(RunLiveResult::UserExit),
    }
}

// ═══════════════════════════════════════════════════════════════════
//  TEXT RENDERING
// ═══════════════════════════════════════════════════════════════════

fn get_fpv_channel_name(freq_mhz: f64) -> Option<&'static str> {
    let freq_round = freq_mhz.round() as i32;
    match freq_round {
        // --- 5.8 GHz Band A (Boscam A) ---
        5865 => Some("A1"),
        5845 => Some("A2"),
        5825 => Some("A3"),
        5805 => Some("A4"),
        5785 => Some("A5"),
        5765 => Some("A6"),
        5745 => Some("A7"),
        5725 => Some("A8"),

        // --- 5.8 GHz Band B (Boscam B) ---
        5733 => Some("B1"),
        5752 => Some("B2"),
        5771 => Some("B3"),
        5790 => Some("B4"),
        5809 => Some("B5"),
        5828 => Some("B6"),
        5847 => Some("B7"),
        5866 => Some("B8"),

        // --- 5.8 GHz Band E (Boscam E) ---
        5705 => Some("E1"),
        5685 => Some("E2"),
        5665 => Some("E3"),
        5645 => Some("E4"),
        5885 => Some("E5"),
        5905 => Some("E6"),
        5925 => Some("E7"),
        5945 => Some("E8"),

        // --- 5.8 GHz Band F (Fat Shark / ImmersionRC) ---
        5740 => Some("F1"),
        5760 => Some("F2"),
        5780 => Some("F3"),
        5800 => Some("F4"),
        5820 => Some("F5"),
        5840 => Some("F6"),
        5860 => Some("F7"),

        // --- 5.8 GHz Band R (Raceband) ---
        5658 => Some("R1"),
        5695 => Some("R2"),
        5732 => Some("R3"),
        5769 => Some("R4"),
        5806 => Some("R5"),
        5843 => Some("R6"),
        5880 => Some("R7/F8"),
        5917 => Some("R8"),

        // --- 5.3 GHz Band D (Boscam D / "5.3G", 5362 + 37 MHz grid) ---
        // The 37 MHz-spaced 5362 grid is the Boscam D band. The
        // 48-channel VTX "L" lowband is the 40 MHz-spaced 5333 grid
        // below; the two are 29 MHz apart at their first channel and
        // are easily conflated.
        5362 => Some("D1"),
        5399 => Some("D2"),
        5436 => Some("D3"),
        // 5473 (D4) collides with U8 — see the shared arm below.
        5510 => Some("D5"),
        5547 => Some("D6"),
        5584 => Some("D7"),
        5621 => Some("D8"),

        // --- 5.3 GHz Band L (Lowband, 5333 + 40 MHz grid) ---
        5333 => Some("L1"),
        // 5373 (L2) collides with U4 — see the shared arm below.
        5413 => Some("L3"),
        5453 => Some("L4"),
        5493 => Some("L5"),
        5533 => Some("L6"),
        5573 => Some("L7"),
        5613 => Some("L8"),

        // --- 5.8 GHz Band U (Ultrabando) ---
        5300 => Some("U1"),
        5325 => Some("U2"),
        5348 => Some("U3"),
        5398 => Some("U5"),
        5423 => Some("U6"),
        5448 => Some("U7"),

        // --- Overlapping/Duplicate 5.8 GHz channels ---
        5373 => Some("U4/L2"),
        5473 => Some("U8/D4"),

        // --- 1.2 / 1.3 GHz Video Bands ---
        1080 => Some("1.2G Ch1"),
        1120 => Some("1.2G Ch2"),
        1160 => Some("1.2G Ch3"),
        1200 => Some("1.2G Ch4"),
        1240 => Some("1.2G Ch5"),
        1258 => Some("1.3G Ch9"),
        1280 => Some("1.3G Ch6"),
        1320 => Some("1.3G Ch7"),
        1360 => Some("1.3G Ch8"),

        // --- 2.4 GHz Video Bands ---
        2410 => Some("2.4G Ch6"),
        2414 => Some("2.4G Ch1"),
        2430 => Some("2.4G Ch7"),
        2432 => Some("2.4G Ch2"),
        2450 => Some("2.4G Ch3/Ch8"),
        2468 => Some("2.4G Ch4"),
        2470 => Some("2.4G Ch9"),
        2490 => Some("2.4G Ch5"),
        _ => None,
    }
}

/// Single source-of-truth for the FPV-channel snap & fine-tune
/// candidate set used by `snap_to_nearest_fpv_channel` and
/// `get_candidate_fpv_channels`. Centralising the array avoids the
/// previous DRY violation where the two functions each inlined the
/// same ~70-entry list and could drift out of sync.
///
/// Note: this list is a superset of
/// `orecchiette_fpv_drone_analog_rs::bands::get_all_channels()`. The overlapping
/// bands (A/B/E/F/R, L, and D) use identical anchor frequencies in
/// both places — `bands.rs` models L as the standard 40 MHz-spaced
/// 5333 lowband grid and D as the 37 MHz-spaced 5362 Boscam D grid,
/// matching this table, so `--channel L1`/`D1` and the display labels
/// agree. The "U" (Ultra-low) and 2.4 GHz video channels live only
/// here because `bands.rs` doesn't model them yet (so `--channel U4`
/// isn't resolvable — that's a separate gap, not a divergence). A
/// follow-up unification pass should move this whole list into
/// `bands.rs` and have both paths read it.
const FPV_CHANNELS_MHZ: &[f64] = &[
    // 5.8 GHz bands
    5865.0, 5845.0, 5825.0, 5805.0, 5785.0, 5765.0, 5745.0, 5725.0, // A
    5733.0, 5752.0, 5771.0, 5790.0, 5809.0, 5828.0, 5847.0, 5866.0, // B
    5705.0, 5685.0, 5665.0, 5645.0, 5885.0, 5905.0, 5925.0, 5945.0, // E
    5740.0, 5760.0, 5780.0, 5800.0, 5820.0, 5840.0,
    5860.0, // F (Fatshark; F8 = 5880 in R row)
    5658.0, 5695.0, 5732.0, 5769.0, 5806.0, 5843.0, 5880.0, 5917.0, // R (Raceband)
    5362.0, 5399.0, 5436.0, 5473.0, 5510.0, 5547.0, 5584.0, 5621.0, // D (Boscam D / "5.3G")
    5333.0, 5413.0, 5453.0, 5493.0, 5533.0, 5573.0, 5613.0, // L (Lowband; L2 = 5373 in U row)
    5300.0, 5325.0, 5348.0, 5373.0, 5398.0, 5423.0, 5448.0, // U (Ultra-low)
    // 1.2 / 1.3 GHz long-range analog FPV
    1080.0, 1120.0, 1160.0, 1200.0, 1240.0, 1258.0, 1280.0, 1320.0, 1360.0,
    // 2.4 GHz video channels (some overlap with Wi-Fi but legitimate analog FPV)
    2410.0, 2414.0, 2430.0, 2432.0, 2450.0, 2468.0, 2470.0, 2490.0,
];

/// Max snap distance when collapsing a coarse-search hit to the
/// nearest FPV channel. 15 MHz is one FPV channel's typical spacing
/// — anything further off is more likely a different channel
/// entirely than an off-tune of the same channel.
const CHANNEL_SNAP_TOLERANCE_MHZ: f64 = 15.0;

fn snap_to_nearest_fpv_channel(freq_hz: f64) -> f64 {
    let freq_mhz = freq_hz / 1e6;
    let mut best_mhz = freq_mhz;
    let mut min_diff = CHANNEL_SNAP_TOLERANCE_MHZ;

    for &ch in FPV_CHANNELS_MHZ {
        let diff = (freq_mhz - ch).abs();
        if diff < min_diff {
            min_diff = diff;
            best_mhz = ch;
        }
    }

    best_mhz * 1e6
}

fn get_candidate_fpv_channels(freq_hz: f64) -> Vec<f64> {
    let freq_mhz = freq_hz / 1e6;
    let mut candidates: Vec<f64> = FPV_CHANNELS_MHZ
        .iter()
        .filter(|&&ch| (freq_mhz - ch).abs() <= CHANNEL_SNAP_TOLERANCE_MHZ)
        .map(|&ch| ch * 1e6)
        .collect();

    // Sort by proximity to the coarse hit so the most likely
    // candidates are checked/printed first by the fine-tune loop.
    candidates.sort_by(|a, b| {
        (freq_hz - *a)
            .abs()
            .partial_cmp(&(freq_hz - *b).abs())
            .unwrap()
    });

    candidates
}

fn get_char_bitmap(c: char) -> [u8; 8] {
    match c {
        ' ' => [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
        '.' => [0x00, 0x00, 0x00, 0x00, 0x00, 0x60, 0x60, 0x00],
        ':' => [0x00, 0x18, 0x18, 0x00, 0x18, 0x18, 0x00, 0x00],
        '/' => [0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x00],
        '+' => [0x00, 0x10, 0x10, 0x7c, 0x10, 0x10, 0x00, 0x00],
        '-' => [0x00, 0x00, 0x00, 0x3e, 0x00, 0x00, 0x00, 0x00],
        '·' | '*' => [0x00, 0x00, 0x18, 0x18, 0x00, 0x00, 0x00, 0x00],
        '0' => [0x3c, 0x66, 0x6e, 0x76, 0x66, 0x66, 0x3c, 0x00],
        '1' => [0x18, 0x38, 0x18, 0x18, 0x18, 0x18, 0x7e, 0x00],
        '2' => [0x3c, 0x66, 0x06, 0x0c, 0x30, 0x60, 0x7e, 0x00],
        '3' => [0x3c, 0x66, 0x06, 0x1c, 0x06, 0x66, 0x3c, 0x00],
        '4' => [0x0c, 0x1c, 0x3c, 0x6c, 0x7e, 0x0c, 0x0c, 0x00],
        '5' => [0x7e, 0x60, 0x7c, 0x06, 0x06, 0x66, 0x3c, 0x00],
        '6' => [0x3c, 0x66, 0x60, 0x7c, 0x66, 0x66, 0x3c, 0x00],
        '7' => [0x7e, 0x66, 0x0c, 0x18, 0x30, 0x30, 0x30, 0x00],
        '8' => [0x3c, 0x66, 0x66, 0x3c, 0x66, 0x66, 0x3c, 0x00],
        '9' => [0x3c, 0x66, 0x66, 0x3e, 0x06, 0x66, 0x3c, 0x00],
        'A' | 'a' => [0x18, 0x3c, 0x66, 0x7e, 0x66, 0x66, 0x66, 0x00],
        'B' | 'b' => [0x7c, 0x66, 0x66, 0x7c, 0x66, 0x66, 0x7c, 0x00],
        'C' | 'c' => [0x3c, 0x66, 0x60, 0x60, 0x60, 0x66, 0x3c, 0x00],
        'D' | 'd' => [0x78, 0x6c, 0x66, 0x66, 0x66, 0x6c, 0x78, 0x00],
        'E' | 'e' => [0x7e, 0x60, 0x60, 0x7c, 0x60, 0x60, 0x7e, 0x00],
        'F' | 'f' => [0x7e, 0x60, 0x60, 0x7c, 0x60, 0x60, 0x60, 0x00],
        'G' | 'g' => [0x3c, 0x66, 0x60, 0x6e, 0x66, 0x66, 0x3c, 0x00],
        'H' | 'h' => [0x66, 0x66, 0x66, 0x7e, 0x66, 0x66, 0x66, 0x00],
        'I' | 'i' => [0x7e, 0x18, 0x18, 0x18, 0x18, 0x18, 0x7e, 0x00],
        'J' | 'j' => [0x06, 0x06, 0x06, 0x06, 0x06, 0x66, 0x3c, 0x00],
        'K' | 'k' => [0x66, 0x6c, 0x78, 0x70, 0x78, 0x6c, 0x66, 0x00],
        'L' | 'l' => [0x60, 0x60, 0x60, 0x60, 0x60, 0x60, 0x7e, 0x00],
        'M' | 'm' => [0x63, 0x77, 0x7f, 0x6b, 0x63, 0x63, 0x63, 0x00],
        'N' | 'n' => [0x66, 0x76, 0x7e, 0x7e, 0x6e, 0x66, 0x66, 0x00],
        'O' | 'o' => [0x3c, 0x66, 0x66, 0x66, 0x66, 0x66, 0x3c, 0x00],
        'P' | 'p' => [0x7c, 0x66, 0x66, 0x7c, 0x60, 0x60, 0x60, 0x00],
        'Q' | 'q' => [0x3c, 0x66, 0x66, 0x66, 0x6a, 0x6c, 0x3e, 0x00],
        'R' | 'r' => [0x7c, 0x66, 0x66, 0x7c, 0x78, 0x6c, 0x66, 0x00],
        'S' | 's' => [0x3c, 0x66, 0x60, 0x3c, 0x06, 0x66, 0x3c, 0x00],
        'T' | 't' => [0x7e, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x00],
        'U' | 'u' => [0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x3c, 0x00],
        'V' | 'v' => [0x66, 0x66, 0x66, 0x66, 0x66, 0x3c, 0x18, 0x00],
        'W' | 'w' => [0x63, 0x63, 0x63, 0x6b, 0x7f, 0x77, 0x63, 0x00],
        'X' | 'x' => [0x66, 0x66, 0x3c, 0x18, 0x3c, 0x66, 0x66, 0x00],
        'Y' | 'y' => [0x66, 0x66, 0x66, 0x3c, 0x18, 0x18, 0x18, 0x00],
        'Z' | 'z' => [0x7e, 0x06, 0x0c, 0x18, 0x30, 0x60, 0x7e, 0x00],
        _ => [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
    }
}

#[allow(clippy::too_many_arguments)]
fn draw_rect(
    buffer: &mut [u32],
    width: usize,
    height: usize,
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    color: u32,
) {
    for row in 0..h {
        let row_y = y + row;
        if row_y >= height {
            break;
        }
        for col in 0..w {
            let col_x = x + col;
            if col_x >= width {
                break;
            }
            buffer[row_y * width + col_x] = color;
        }
    }
}

fn draw_string(
    buffer: &mut [u32],
    width: usize,
    height: usize,
    x: usize,
    y: usize,
    text: &str,
    color: u32,
) {
    let mut current_x = x;
    for c in text.chars() {
        let bitmap = get_char_bitmap(c);
        for (row, &val) in bitmap.iter().enumerate().take(8) {
            let row_y = y + row;
            if row_y >= height {
                break;
            }
            let row_val = val;
            for col in 0..8 {
                let col_x = current_x + col;
                if col_x >= width {
                    break;
                }
                if (row_val & (0x80 >> col)) != 0 {
                    buffer[row_y * width + col_x] = color;
                }
            }
        }
        current_x += 8;
    }
}

#[allow(clippy::too_many_arguments)]
fn draw_text_with_bg(
    buffer: &mut [u32],
    width: usize,
    height: usize,
    x: usize,
    y: usize,
    text: &str,
    text_color: u32,
    bg_color: u32,
) {
    let w = text.len() * 8 + 4;
    let h = 12;
    draw_rect(buffer, width, height, x, y, w, h, bg_color);
    draw_string(buffer, width, height, x + 2, y + 2, text, text_color);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every channel must sit inside some planned window. This is the
    /// property the whole planner exists for — a channel nobody tunes
    /// close enough to is a drone the scan cannot see.
    fn assert_covers(channels: &[f64], centers: &[f64], span: f64) {
        for ch in channels {
            let covered = centers.iter().any(|c| (ch - c).abs() <= span / 2.0 + 1.0);
            assert!(
                covered,
                "channel {:.3} MHz not covered by any of {} tunes at {:.1} MHz span",
                ch / 1e6,
                centers.len(),
                span / 1e6
            );
        }
    }

    /// The decode path may only shed bandwidth it has already filtered
    /// away. Energy at `work_rate - cutoff` folds onto the passband
    /// edge, so that frequency has to clear the filter's stopband edge.
    ///
    /// The rule has to be driven by the channel's own cutoff, not a
    /// fixed target rate: file playback defaults to a 17 MHz deviation,
    /// and "always decode at 15.36 MSPS" would fold a 19 MHz-wide
    /// channel onto itself.
    #[test]
    fn decode_decimation_never_folds_a_channel_onto_itself() {
        // A live channel (5 MHz deviation) and a file one (17 MHz).
        for cutoff in [5e6f32 + LUMA_HEADROOM_HZ, 17e6 + LUMA_HEADROOM_HZ] {
            for rate in [
                7_680_000u32,
                15_360_000,
                20_000_000,
                25_000_000,
                30_720_000,
                40_000_000,
                50_000_000,
                61_440_000,
                100_000_000,
            ] {
                let decim = decode_decimation(rate, cutoff);
                assert!(decim >= 1, "{rate} at {cutoff}: factor must be positive");
                if decim == 1 {
                    // Declining to decimate is always safe. Whether the
                    // capture was wide enough for the channel in the
                    // first place is not this function's business — at
                    // 7.68 MSPS a 7 MHz cutoff is already undersampled,
                    // and no stride fixes that.
                    continue;
                }
                let work_rate = (rate / decim as u32) as f32;
                // Nothing above the new Nyquist that could fold into
                // the passband is still inside the filter's transition.
                let stopband_edge =
                    cutoff + FIR_TRANSITION_K * rate as f32 / DECODE_FIR_TAPS as f32;
                assert!(
                    work_rate - cutoff >= stopband_edge,
                    "{:.2} MSPS at a {:.1} MHz cutoff decimated by {decim} to {:.2} MSPS: \
                     energy at {:.2} MHz folds onto the passband edge but the filter is \
                     still {:.2} MHz from its stopband",
                    rate as f64 / 1e6,
                    cutoff as f64 / 1e6,
                    work_rate as f64 / 1e6,
                    (work_rate - cutoff) as f64 / 1e6,
                    stopband_edge as f64 / 1e6
                );
            }
        }

        // The live defaults, spelled out: both wide Aaronia spans land
        // on the same working rate, and every rate a decoder already
        // kept up with is left alone.
        let live = 5e6f32 + LUMA_HEADROOM_HZ;
        assert_eq!(decode_decimation(61_440_000, live), 4);
        assert_eq!(decode_decimation(30_720_000, live), 2);
        for unchanged in [25_000_000u32, 20_000_000, 15_360_000, 7_680_000] {
            assert_eq!(
                decode_decimation(unchanged, live),
                1,
                "{unchanged} has nothing to shed at a 7 MHz cutoff"
            );
        }

        // A 19 MHz-wide file channel needs ~38 MSPS of working rate, so
        // it must not decimate below that.
        let file = 17e6f32 + LUMA_HEADROOM_HZ;
        assert_eq!(decode_decimation(61_440_000, file), 1);
        assert!(100_000_000 / decode_decimation(100_000_000, file) as u32 >= 38_000_000);

        // Degenerate input must not divide by zero or panic.
        assert_eq!(decode_decimation(0, live), 1);
        assert_eq!(decode_decimation(61_440_000, 0.0), 1);
        assert_eq!(decode_decimation(61_440_000, f32::NAN), 1);
    }

    #[test]
    fn tune_planner_covers_every_channel_at_every_plausible_rate() {
        for bands in [ScanBands::Band58, ScanBands::All] {
            let channels = bands.channel_freqs_hz();
            assert!(!channels.is_empty(), "{:?} yielded no channels", bands);
            for rate in [15.36e6f64, 20e6, 25e6, 30.72e6, 40e6, 50e6, 61.44e6] {
                let span = rate * USABLE_SPAN_FRACTION;
                let hops = build_scan_hops(rate, bands);
                assert_covers(&channels, &hops, span);
                assert!(
                    hops.len() <= channels.len(),
                    "{:?} at {:.2} MSPS planned {} tunes for {} channels — never worse than one each",
                    bands,
                    rate / 1e6,
                    hops.len(),
                    channels.len()
                );
            }
        }
    }

    /// A wide capture must actually collapse the table, otherwise the
    /// planner is buying nothing over tuning each channel in turn.
    #[test]
    fn tune_planner_collapses_the_table_when_the_capture_is_wide() {
        let channels = ScanBands::Band58.channel_freqs_hz();
        let hops = build_scan_hops(61.44e6, ScanBands::Band58);
        assert!(
            hops.len() * 4 < channels.len(),
            "61.44 MSPS should cover {} channels in far fewer than {} tunes, got {}",
            channels.len(),
            channels.len() / 4,
            hops.len()
        );
    }

    /// Hops ascend so synthesisers retune monotonically.
    #[test]
    fn tune_planner_emits_ascending_hops() {
        let hops = build_scan_hops(25e6, ScanBands::All);
        assert!(
            hops.windows(2).all(|w| w[0] < w[1]),
            "hop list must be strictly ascending, got {hops:?}"
        );
    }

    /// A span too narrow to group anything must degrade to one tune per
    /// channel rather than silently dropping channels — and must not
    /// spin forever doing it.
    #[test]
    fn tune_planner_degrades_to_one_tune_per_channel_when_span_is_zero() {
        let channels = vec![5_658e6, 5_695e6, 5_732e6];
        let hops = plan_tune_centers(&channels, 0.0);
        assert_eq!(hops, channels);
    }

    #[test]
    fn tune_planner_handles_degenerate_input() {
        assert!(plan_tune_centers(&[], 50e6).is_empty());
        // Duplicates collapse rather than each claiming a tune.
        assert_eq!(
            plan_tune_centers(&[5_800e6, 5_800e6, 5_800e6], 0.0).len(),
            1
        );
        // A non-finite entry must not poison the sort into an infinite loop.
        let hops = plan_tune_centers(&[5_800e6, f64::NAN, 5_810e6], 50e6);
        assert_eq!(hops.len(), 1);
    }

    /// Scanning everything must be a strict superset of scanning 5.8,
    /// and must cost more tunes — that trade is the point of the flag.
    #[test]
    fn all_bands_is_a_superset_of_the_58_band() {
        let band58 = ScanBands::Band58.channel_freqs_hz();
        let all = ScanBands::All.channel_freqs_hz();
        assert!(band58.iter().all(|f| all.contains(f)));
        assert!(all.len() > band58.len());
        assert!(
            build_scan_hops(61.44e6, ScanBands::All).len()
                > build_scan_hops(61.44e6, ScanBands::Band58).len()
        );
    }

    /// A healthy decode must never be released.
    #[test]
    fn a_decoding_carrier_holds_lock_indefinitely() {
        let mut lock = LockState::default();
        for i in 0..1000 {
            assert!(
                !lock.observe(true, true),
                "released a decoding carrier at check {i}"
            );
        }
    }

    /// The review's case: a horizontal-sync train with no vertical sync
    /// scores 0.8 on the sweep detector forever and reconstructs no
    /// frame. "There is a hit" held the viewer on a picture it could
    /// never draw.
    #[test]
    fn a_detected_but_undecodable_carrier_is_eventually_released() {
        let mut lock = LockState::default();
        let mut checks = 0;
        while !lock.observe(true, false) {
            checks += 1;
            assert!(checks < 100, "never released an undecodable carrier");
        }
        assert_eq!(checks + 1, LOCK_PROVISIONAL_CHECKS);
    }

    /// A hit that is not ours must not keep this channel alive — the
    /// lock check ran on the whole capture, so a transmitter elsewhere
    /// in a 49 MHz span did exactly that.
    #[test]
    fn a_hit_elsewhere_does_not_hold_our_channel() {
        let mut lock = LockState::default();
        // Someone else's strong, decodable signal.
        for _ in 0..LOCK_EMPTY_CHECKS - 1 {
            assert!(!lock.observe(false, true));
        }
        assert!(
            lock.observe(false, true),
            "another channel's detection kept ours locked"
        );
    }

    /// The review's sequence: a miss, then the carrier returns without
    /// a field, then another miss. Those two misses are not consecutive
    /// and must not release the channel.
    #[test]
    fn non_consecutive_misses_do_not_release_the_channel() {
        // miss -> carrier back but no field yet -> miss. Two misses,
        // not consecutive. `LOCK_EMPTY_CHECKS` counts consecutive ones,
        // so this must survive; before the fix the second miss released.
        let mut lock = LockState::default();
        assert!(!lock.observe(false, true), "a single miss released");
        assert!(!lock.observe(true, false), "a provisional check released");
        assert!(
            !lock.observe(false, true),
            "two non-consecutive misses released the channel"
        );

        // A decoded field clears both counters, so an isolated miss
        // between good checks can repeat indefinitely without drifting
        // toward release. (Good check first: the sequence above already
        // ended on a miss, and two in a row *should* release.)
        for _ in 0..50 {
            assert!(!lock.observe(true, true));
            assert!(!lock.observe(false, true), "an isolated miss released");
        }
    }

    /// Strictly consecutive misses still release, on the documented
    /// count — the fix must not make the channel un-releasable.
    #[test]
    fn consecutive_misses_still_release() {
        let mut lock = LockState::default();
        for _ in 0..LOCK_EMPTY_CHECKS - 1 {
            assert!(!lock.observe(false, false));
        }
        assert!(lock.observe(false, false));
    }

    /// And an undecodable carrier still runs out its own budget even
    /// though it keeps clearing the miss counter.
    #[test]
    fn a_provisional_carrier_still_times_out() {
        let mut lock = LockState::default();
        for _ in 0..LOCK_PROVISIONAL_CHECKS - 1 {
            assert!(!lock.observe(true, false));
        }
        assert!(lock.observe(true, false));
    }

    /// Acquisition stutter must not accumulate toward release: a check
    /// that sees both clears the provisional count.
    #[test]
    fn a_field_arriving_clears_the_provisional_count() {
        let mut lock = LockState::default();
        for _ in 0..LOCK_PROVISIONAL_CHECKS - 1 {
            assert!(!lock.observe(true, false));
        }
        assert!(!lock.observe(true, true), "a good check should recover");
        // Full budget available again.
        for _ in 0..LOCK_PROVISIONAL_CHECKS - 1 {
            assert!(!lock.observe(true, false));
        }
    }

    /// The auto rule has to be about the deviation, not the rate. At
    /// FPV's 5 MHz the PLL measured 9-25 dB *worse* than the
    /// discriminator, because its loop cannot track that excursion —
    /// and it also costs 6.49 dB at 4.2 MHz luma.
    #[test]
    fn auto_does_not_pick_the_pll_at_an_fpv_deviation() {
        for rate in [25_000_000u32, 30_720_000, 61_440_000] {
            assert!(
                !DemodKind::Auto.use_pll(rate, 5_000_000.0),
                "auto chose the PLL at {rate} Hz for a 5 MHz deviation"
            );
        }
    }

    /// Through the *pipeline's* ordering — decimate, then choose —
    /// `Auto` resolves to the discriminator everywhere.
    ///
    /// This replaces a test that asked `use_pll(25_000_000, 500_000.0)`
    /// directly and concluded the PLL was reachable. The pipeline never
    /// poses that question: a 500 kHz deviation at 25 MSPS decimates by
    /// 4 first, so what `Auto` is actually asked about is 6.25 MSPS,
    /// where the loop is worth +1.8 dB rather than +9.9 and the rate
    /// gate correctly declines.
    #[test]
    fn auto_resolves_to_the_discriminator_through_the_real_ordering() {
        for capture in [15_360_000u32, 25_000_000, 30_720_000, 61_440_000] {
            for dev in [200_000.0f32, 500_000.0, 1_000_000.0, 5_000_000.0] {
                let (work, pll) = select_decode(capture, dev, DemodKind::Auto);
                assert!(
                    !pll,
                    "auto chose the PLL at capture {capture} dev {dev} \
                     (work rate {work})"
                );
            }
        }
    }

    /// `--demod pll` still gets the PLL, and gets it at a rate the loop
    /// can use rather than a decimated one.
    #[test]
    fn forcing_the_pll_holds_the_decode_rate_up() {
        let (work, pll) = select_decode(25_000_000, 500_000.0, DemodKind::Pll);
        assert!(pll, "--demod pll must force it");
        assert!(
            work >= PLL_AUTO_MIN_SAMPLE_RATE_HZ,
            "forced PLL decoded at {work}, below the rate its loop needs"
        );
    }

    /// Below the rate threshold the deviation is irrelevant.
    #[test]
    fn auto_keeps_the_discriminator_below_the_rate_threshold() {
        assert!(!DemodKind::Auto.use_pll(15_360_000, 500_000.0));
    }

    /// The clamp is real: at a low rate the loop cannot reach its
    /// nominal bandwidth, so a deviation it could otherwise track is
    /// still out of reach.
    #[test]
    fn the_loop_bandwidth_clamp_is_respected() {
        // fs / 4pi at 25 MSPS is ~1.99 MHz, so PLL_LOOP_BW_HZ binds.
        assert!(!DemodKind::Auto.use_pll(25_000_000, 1_500_000.0));
    }

    /// Explicit flags keep meaning what they say.
    #[test]
    fn explicit_demod_choices_ignore_the_heuristic() {
        assert!(DemodKind::Pll.use_pll(15_360_000, 5_000_000.0));
        assert!(!DemodKind::Discriminator.use_pll(61_440_000, 100_000.0));
    }

    /// A degenerate deviation must not select the PLL by accident.
    #[test]
    fn a_degenerate_deviation_keeps_the_discriminator() {
        assert!(!DemodKind::Auto.use_pll(25_000_000, 0.0));
        assert!(!DemodKind::Auto.use_pll(25_000_000, -1.0));
        assert!(!DemodKind::Auto.use_pll(25_000_000, f32::NAN));
    }

    /// A packet is a fixed 65,536 samples, so its worth depends on the
    /// rate: 4.27 ms of signal at 15.36 MSPS against 1.07 at 61.44, for
    /// 2.49 ms of detector against 6.41. Narrow captures get the
    /// integration budget because there it is both more useful and
    /// affordable.
    #[test]
    fn a_narrow_capture_gets_the_integration_budget() {
        assert_eq!(detect_packets_per_hop(15_360_000.0), DETECT_PACKETS_NARROW);
        assert_eq!(detect_packets_per_hop(30_720_000.0), DETECT_PACKETS_NARROW);
    }

    /// The review's case: a noisy fixture at 61.44 MSPS first detects
    /// on packet 3, which a two-packet fast pass can never reach. Some
    /// sweep has to spend more, and no per-hop prediction can decide
    /// which — so the cadence does it unconditionally.
    #[cfg(feature = "aaronia")]
    #[test]
    fn a_sensitive_sweep_comes_round_and_reaches_packet_three() {
        let wide = 61_440_000.0;
        let budgets: Vec<usize> = (0..AARONIA_SENSITIVE_EVERY * 2)
            .map(|n| SweepBudget::for_sweep(n, wide).packets_per_hop)
            .collect();
        assert!(
            budgets.iter().any(|&p| p >= 3),
            "no sweep in two full cycles could see packet 3: {budgets:?}"
        );
        assert_eq!(
            budgets
                .iter()
                .filter(|&&p| p == DETECT_PACKETS_NARROW)
                .count(),
            2,
            "expected one sensitive sweep per cycle: {budgets:?}"
        );
    }

    /// A sensitive sweep must be able to pay for its own packets, or
    /// the hop ends mid-budget and the extra passes are never read.
    #[cfg(feature = "aaronia")]
    #[test]
    fn the_sensitive_dwell_pays_for_its_packets() {
        let b = SweepBudget::for_sweep(AARONIA_SENSITIVE_EVERY - 1, 61_440_000.0);
        // 6.41 ms per pass, measured at 61.44 MSPS.
        let cpu_ms = 6.41 * b.packets_per_hop as f64;
        assert!(
            cpu_ms < b.dwell.as_secs_f64() * 1e3,
            "{cpu_ms} ms of detector does not fit a {:?} dwell",
            b.dwell
        );
    }

    /// Most sweeps stay fast — the point is a bounded cost, not a
    /// slower scanner.
    #[cfg(feature = "aaronia")]
    #[test]
    fn most_sweeps_keep_the_fast_budget() {
        let wide = 61_440_000.0;
        let fast = (0..AARONIA_SENSITIVE_EVERY)
            .filter(|&n| SweepBudget::for_sweep(n, wide).packets_per_hop == DETECT_PACKETS_PER_HOP)
            .count();
        assert_eq!(fast as u32, AARONIA_SENSITIVE_EVERY - 1);
    }

    /// A narrow capture already reads the sensitive count every sweep,
    /// so the cadence must not make it slower.
    #[cfg(feature = "aaronia")]
    #[test]
    fn a_narrow_capture_is_unaffected_by_the_cadence() {
        for n in 0..AARONIA_SENSITIVE_EVERY * 2 {
            let b = SweepBudget::for_sweep(n, 15_360_000.0);
            assert_eq!(b.packets_per_hop, DETECT_PACKETS_NARROW);
            assert_eq!(b.dwell, AARONIA_SCAN_DWELL);
        }
    }

    /// Six passes at 61.44 MSPS is 38.5 ms of detector against a 25 ms
    /// dwell — the hop would end before the budget was spent.
    #[test]
    fn a_wide_capture_keeps_the_fast_pass() {
        assert_eq!(detect_packets_per_hop(61_440_000.0), DETECT_PACKETS_PER_HOP);
        assert!(
            detect_packets_per_hop(61_440_000.0) < detect_packets_per_hop(15_360_000.0),
            "a wide capture must spend fewer passes than a narrow one"
        );
    }

    /// A nonsense rate must not silently pick the expensive branch.
    #[test]
    fn a_degenerate_rate_keeps_the_fast_pass() {
        assert_eq!(detect_packets_per_hop(0.0), DETECT_PACKETS_PER_HOP);
        assert_eq!(detect_packets_per_hop(-1.0), DETECT_PACKETS_PER_HOP);
        assert_eq!(detect_packets_per_hop(f64::NAN), DETECT_PACKETS_PER_HOP);
    }

    /// The budget must fit the dwell, or the hop ends mid-budget and
    /// the extra packets are never read.
    #[test]
    fn the_narrow_budget_fits_the_dwell() {
        // 2.49 ms per pass, measured at 15.36 MSPS.
        let cpu_ms = 2.49 * DETECT_PACKETS_NARROW as f64;
        assert!(
            cpu_ms < AARONIA_SCAN_DWELL.as_secs_f64() * 1e3,
            "{cpu_ms} ms of detector does not fit a {:?} dwell",
            AARONIA_SCAN_DWELL
        );
    }

    /// A locked channel drops below the scan rate, but only to a rate
    /// whose usable span still holds the channel.
    #[cfg(feature = "aaronia")]
    #[test]
    fn a_locked_channel_drops_to_a_rate_that_fits_the_channel() {
        // 5 MHz deviation -> 7 MHz cutoff -> needs 17.5 MSPS of usable
        // span, so 30.72 is the lowest supported rate that holds it.
        let r = locked_capture_rate_hz(61_440_000.0, 5_000_000.0, DemodKind::Auto);
        assert_eq!(r, 30_720_000.0);
        assert!(r < 61_440_000.0, "must still be a reduction");
    }

    /// The bug this function shipped with: `decode_decimation` answers
    /// 3 at 6 MHz deviation, 61.44/3 is 20.48 MSPS which the hardware
    /// does not run, and snapping to the nearest supported rate landed
    /// on 15.36 — below the 17.1 MSPS its own cutoff requires. Every
    /// rate this returns must hold its own channel.
    #[cfg(feature = "aaronia")]
    #[test]
    fn every_chosen_rate_satisfies_its_own_bandwidth_requirement() {
        for dev in [3e6f32, 5e6, 6e6, 7e6, 9e6, 12e6, 17e6] {
            let r = locked_capture_rate_hz(61_440_000.0, dev, DemodKind::Auto);
            let cutoff = (dev + LUMA_HEADROOM_HZ) as f64;
            assert!(
                USABLE_SPAN_FRACTION * r >= 2.0 * cutoff,
                "deviation {dev} chose {r}, whose usable span {} cannot hold {} Hz",
                USABLE_SPAN_FRACTION * r,
                2.0 * cutoff
            );
            assert!(r <= 61_440_000.0, "must never exceed the scan rate");
        }
    }

    /// Specifically the two the review named.
    #[cfg(feature = "aaronia")]
    #[test]
    fn six_and_seven_mhz_deviation_no_longer_pick_15_36() {
        assert_eq!(
            locked_capture_rate_hz(61_440_000.0, 6_000_000.0, DemodKind::Auto),
            30_720_000.0
        );
        assert_eq!(
            locked_capture_rate_hz(61_440_000.0, 7_000_000.0, DemodKind::Auto),
            30_720_000.0
        );
    }

    /// Forcing the PLL keeps the scan rate: `run_live` caps decimation
    /// for it, and that cap is not generally a rate the hardware runs.
    #[cfg(feature = "aaronia")]
    #[test]
    fn forcing_the_pll_keeps_the_scan_rate() {
        assert_eq!(
            locked_capture_rate_hz(61_440_000.0, 5_000_000.0, DemodKind::Pll),
            61_440_000.0
        );
    }

    /// A capture already at the narrowest rate that fits has nothing to
    /// give up.
    #[cfg(feature = "aaronia")]
    #[test]
    fn an_already_narrow_capture_is_left_alone() {
        assert_eq!(
            locked_capture_rate_hz(15_360_000.0, 5_000_000.0, DemodKind::Auto),
            15_360_000.0
        );
    }

    /// A wide-deviation link must not be narrowed onto itself.
    #[cfg(feature = "aaronia")]
    #[test]
    fn a_wide_deviation_link_is_not_narrowed() {
        let wide = 17_000_000.0f32;
        let r = locked_capture_rate_hz(61_440_000.0, wide, DemodKind::Auto);
        assert_eq!(r, 61_440_000.0, "nothing below the scan rate can hold it");
    }
}
