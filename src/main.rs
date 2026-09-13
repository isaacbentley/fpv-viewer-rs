mod band_scan;
mod decode_worker;

use band_scan::{BandScan, PanelMode, ScanWindow};
use clap::{Args as ClapArgs, Parser, Subcommand};
use decode_worker::{DecodeWorker, DecodeWorkerConfig, FrameSlot, IqChunk, lock_slot};
use minifb::{Key, Window, WindowOptions};
use num_complex::Complex;
use orecchiette_fpv_drone_analog_rs::acquisition::{
    AcquisitionRecord, LockState, analyze_channel_standard, analyze_tuned_channel,
};
use orecchiette_fpv_drone_analog_rs::bands::{get_fpv_channel_name, snap_to_nearest_fpv_channel};
use orecchiette_fpv_drone_analog_rs::decode::{
    DecodePlan, DecoderConfig, DemodulationMode, LUMA_HEADROOM_HZ, decode_decimation,
};
use orecchiette_fpv_drone_analog_rs::demod::DEFAULT_DEEMPHASIS_TAU_S;
use orecchiette_fpv_drone_analog_rs::detector::{FpvDetector, ProbeEnergy, SpectralIntegrator};
use orecchiette_fpv_drone_analog_rs::lookup_channel_by_name;
use orecchiette_fpv_drone_analog_rs::scanner::{
    CandidatePolicy, DETECT_PACKETS_PER_HOP, FineTuneSelector, ScanProgress, ScanSelector,
    SweepBudget, plan_tune_centers,
};
#[cfg(any(test, feature = "aaronia"))]
use orecchiette_fpv_drone_analog_rs::scanner::{DETECT_PACKETS_NARROW, detect_packets_per_hop};
use orecchiette_fpv_drone_analog_rs::types::SignalType;
use orecchiette_fpv_drone_analog_rs::video::FrameReconstructor;
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
    /// D1–D8, U1–U8, N1–N8, W1–W9, T1–T9, S1–S64) or raw frequency in Hz (e.g. 5865000000).
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

impl From<DemodKind> for DemodulationMode {
    fn from(mode: DemodKind) -> Self {
        match mode {
            DemodKind::Auto => Self::Auto,
            DemodKind::Discriminator => Self::Discriminator,
            DemodKind::Pll => Self::Pll,
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

/// Aaronia's measured tuning costs, supplied to the shared sweep policy.
#[cfg(feature = "aaronia")]
fn aaronia_sweep_budget(n: u32, sample_rate_hz: f64) -> SweepBudget {
    orecchiette_fpv_drone_analog_rs::scanner::SweepPolicy {
        fast: SweepBudget::flat(AARONIA_SCAN_DWELL, DETECT_PACKETS_PER_HOP),
        sensitive: SweepBudget::flat(AARONIA_SENSITIVE_DWELL, DETECT_PACKETS_NARROW),
        sensitive_every: AARONIA_SENSITIVE_EVERY,
        narrow_capture_max_rate_hz:
            orecchiette_fpv_drone_analog_rs::scanner::NARROW_CAPTURE_MAX_RATE_HZ,
    }
    .budget(n, sample_rate_hz)
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
    let Some(needed_rate) = orecchiette_fpv_drone_analog_rs::decode::required_capture_rate(
        fm_deviation,
        USABLE_SPAN_FRACTION,
    ) else {
        return scan_rate_hz;
    };

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

    fn selection(self) -> orecchiette_fpv_drone_analog_rs::bands::BandSelection {
        match self {
            Self::Band58 => orecchiette_fpv_drone_analog_rs::bands::BandSelection::Band58,
            Self::All => orecchiette_fpv_drone_analog_rs::bands::BandSelection::All,
        }
    }
    fn channel_freqs_hz(self) -> Vec<f64> {
        self.selection()
            .channels()
            .map(|c| c.frequency_hz as f64)
            .collect()
    }
    fn channels(self) -> Vec<band_scan::Channel> {
        self.selection()
            .channels()
            .map(|c| band_scan::Channel {
                name: c.name(),
                band: c.band.code(),
                hz: c.frequency_hz as f64,
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

/// Build the hop list for the auto-scan loop, covering every channel in
/// `bands` with as few tunes as the SDR's instantaneous bandwidth allows.
///
/// Planning against the channel list rather than stepping uniformly
/// spends tunes only where channels are:
///
/// ```text
///                        5.8 GHz    all bands
///                    (40 channels) (154 channels)
///     20.00 MSPS            16          108
///     25.00 MSPS            11          100
///     40.00 MSPS             8           60
///     61.44 MSPS             6           51
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

/// Snap an observed packet centre frequency to the nearest expected hop frequency
/// within tolerance (1 MHz). Hardware synthesisers (such as Aaronia over HTTP metadata)
/// can report exact PLL frequencies with small fractional offsets from the commanded centre.
/// Use this only for hop bookkeeping; DSP must use the packet's measured RF centre.
fn snap_to_nearest_hop(freq_hz: f64, hops: &[f64]) -> u64 {
    hops.iter()
        .min_by(|a, b| (freq_hz - **a).abs().total_cmp(&(freq_hz - **b).abs()))
        .filter(|&&nearest| (freq_hz - nearest).abs() <= 1_000_000.0)
        .map(|&nearest| nearest as u64)
        .unwrap_or(freq_hz as u64)
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
        "unrecognised channel '{}'; expected a catalog channel (A/B/E/F/R/L/D/U/N/W/T/S) or a frequency in Hz",
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
            let (standard, confidence) = analyze_channel_standard(
                &first_iq,
                *freq_offset,
                sample_rate,
                fm_deviation,
                SignalType::AnalogVideoNtsc,
            )?;
            println!(
                "  → Standard {:?}: {:?} ({:.0}% classifier confidence)",
                standard.evidence,
                standard.standard,
                confidence * 100.0
            );
            standard.standard
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
    let observed_timing_fields = Arc::new(std::sync::atomic::AtomicU64::new(0));

    let _ = run_viewer_pipeline(
        resolved_channels,
        // File playback tunes nothing, so there is no tuned centre to
        // re-measure against; the channel comes from the file's own
        // stated frequency.
        None,
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
        observed_timing_fields.clone(),
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
    /// Collecting a channel number; Enter accepts an ambiguous prefix such as S6.
    Typing(String),
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
                let detector = orecchiette_fpv_drone_analog_rs::scanner::receiver_detector();
                let mut selector =
                    ScanSelector::new(&skipped_freqs, CandidatePolicy::AnyUnskippedChannel);
                let mut progress = ScanProgress::new(&hop_freqs, DETECT_PACKETS_PER_HOP);
                let mut sweeps_completed = 0;
                let mut last_progress_msg = Instant::now();
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
                        let rf_center = packet.center_frequency_hz as u64;
                        let hop_center =
                            snap_to_nearest_hop(packet.center_frequency_hz, &hop_freqs);

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
                        let Some(hop) = progress.observe(hop_center) else {
                            continue;
                        };

                        let results = detector.detect_from_iq_integrated_with_probes(
                            &packet.samples,
                            rf_center,
                            current_sample_rate as u32,
                            &mut sweep.integrator,
                            &mut probes,
                        );
                        // With one tune there is no retune to mark a new
                        // look, so every packet replaces the last.
                        sweep.band.record_hop(
                            hop_center as f64,
                            rf_center as f64,
                            hop.first_packet || !multi_hop,
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
                            selector.consider(&res);
                        }
                        // Sweep is complete once every hop has been visited
                        // *and* the current one has had its full packet
                        // budget. Without the budget half of that test this
                        // fires on the first packet of the last hop, which
                        // both cuts that hop short of the two passes it
                        // needs to detect anything and clears `seen_freqs`
                        // mid-hop, sliding the sweep boundary one hop
                        // earlier on every subsequent pass.
                        if hop.sweep_complete {
                            sweep.band.end_sweep();
                            if let Some((freq, _rssi, _sig_type)) = selector.best() {
                                // Found something, break out of the infinite scan loop
                                (handle.stop)();
                                std::thread::sleep(Duration::from_millis(200));

                                let candidates = selector.refinement_candidates(freq);
                                if candidates.is_empty() {
                                    let snapped_freq = selector
                                        .fallback_frequency(freq)
                                        .expect("selected hit has an unskipped fallback");
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

                                let mut ft_selector = FineTuneSelector::default();
                                let mut ft_progress = ScanProgress::new(&candidates, 1);

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
                                        let rf_center = packet.center_frequency_hz as u64;
                                        let hop_center = snap_to_nearest_hop(
                                            packet.center_frequency_hz,
                                            &candidates,
                                        );

                                        let Some(ft_hop) = ft_progress.observe(hop_center) else {
                                            continue;
                                        };

                                        let (_sig_type, conf) = detector.detect_sync_pulses(
                                            &packet.samples,
                                            current_sample_rate as u32,
                                        );

                                        let mut rssi = -100.0;
                                        if let Some(res) = detector
                                            .detect_from_iq(
                                                &packet.samples,
                                                rf_center,
                                                current_sample_rate as u32,
                                            )
                                            .first()
                                        {
                                            rssi = res.rssi_dbm;
                                        }

                                        println!(
                                            "  → Testing {:.3} MHz ({}): conf {:.0}%, rssi {:.1} dBm",
                                            hop_center as f64 / 1e6,
                                            get_fpv_channel_name(hop_center as f64 / 1e6)
                                                .unwrap_or("Unknown"),
                                            conf * 100.0,
                                            rssi
                                        );

                                        ft_selector.consider(hop_center as f64, conf, rssi);

                                        if ft_hop.sweep_complete {
                                            (ft_handle.stop)();
                                            std::thread::sleep(Duration::from_millis(200));

                                            if let Some((best_freq, _best_conf, _)) =
                                                ft_selector.best()
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
                                            let snapped_freq = selector
                                                .fallback_frequency(freq)
                                                .expect("selected hit has an unskipped fallback");
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
                                progress.reset_sweep();
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

/// Give up waiting for a clean record after this long and decide on
/// whatever was gathered.
const STANDARD_DETECT_PATIENCE: Duration = Duration::from_millis(1500);

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
    let detector = orecchiette_fpv_drone_analog_rs::scanner::receiver_detector();

    let mut selector = ScanSelector::new(skipped_freqs, CandidatePolicy::NearestChannel);
    let mut progress = ScanProgress::new(&hop_freqs, budget.packets_per_hop);
    let mut probes: Vec<ProbeEnergy> = Vec::new();
    let outcome = |hit: Option<(f64, f32, SignalType)>| match hit {
        Some((f, _, sig)) => ScanOutcome::Found(snap_to_nearest_fpv_channel(f), sig),
        None => ScanOutcome::Empty,
    };

    loop {
        match handle.receiver.recv_timeout(Duration::from_millis(2000)) {
            Ok(packet) => {
                let rf_center = packet.center_frequency_hz as u64;
                let hop_center = snap_to_nearest_hop(packet.center_frequency_hz, &hop_freqs);
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
                let Some(hop) = progress.observe(hop_center) else {
                    continue;
                };

                let results = detector.detect_from_iq_integrated_with_probes(
                    &packet.samples,
                    rf_center,
                    sample_rate as u32,
                    &mut sweep.integrator,
                    &mut probes,
                );
                sweep.band.record_hop(
                    hop_center as f64,
                    rf_center as f64,
                    hop.first_packet,
                    &probes,
                    &results,
                );
                if let Some(ui) = sweep.ui.as_mut()
                    && !ui.update(&sweep.band)
                {
                    (handle.stop)();
                    return Ok(ScanOutcome::Quit);
                }

                for res in results {
                    if selector.consider(&res) {
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
                if hop.sweep_complete {
                    sweep.band.end_sweep();
                    (handle.stop)();
                    std::thread::sleep(Duration::from_millis(150));
                    return Ok(outcome(selector.best()));
                }
            }
            Err(_) => {
                // SDR went quiet — return whatever we found this sweep.
                (handle.stop)();
                return Ok(outcome(selector.best()));
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
                    let b = aaronia_sweep_budget(sweep_n, sample_rate);
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
    // The carrier, re-measured once we are tuned to it.
    //
    // The sweep's estimate named this channel, and at a low probe SNR it
    // is not good enough to: measured live, the sweep put a 5865 MHz
    // signal at 5871.775 and the nearest-channel snap called it B8
    // (5866) rather than A1 (5865), which sit 1 MHz apart. The same
    // signal seen centred in its own capture localizes to within about
    // 0.1 MHz of a consistent value, because the probe is looking
    // straight at it instead of off the side of a sweep bin.
    let mut refined_carrier_hz: Option<f64> = None;
    let resolved_channels = {
        // Single channel at DC
        let sig_type = if let Some(st) = forced_standard {
            println!("Forced standard: {:?}", st);
            st
        } else {
            println!("Auto-detecting video standard...");
            let mut record = AcquisitionRecord::new(sample_rate_u32);
            record.push(&first_packet.samples, false);
            let t0 = Instant::now();
            while !record.is_ready() && t0.elapsed() < STANDARD_DETECT_PATIENCE {
                let left = STANDARD_DETECT_PATIENCE.saturating_sub(t0.elapsed());
                match handle.receiver.recv_timeout(left) {
                    Ok(p) => {
                        if p.sample_rate_hz as u32 != sample_rate_u32
                            || (p.center_frequency_hz - actual_center).abs() > 1.0
                        {
                            (handle.stop)();
                            anyhow::bail!("SDR rate or centre changed during channel acquisition");
                        }
                        record.push(&p.samples, p.overrun);
                        first_packet = p;
                    }
                    Err(_) => break,
                }
            }
            let acquisition = analyze_tuned_channel(
                record.samples(),
                actual_center,
                sample_rate_u32,
                fm_deviation,
                scan_hint
                    .filter(|hint| {
                        matches!(
                            hint,
                            SignalType::AnalogVideoPal | SignalType::AnalogVideoNtsc
                        )
                    })
                    .unwrap_or(SignalType::AnalogVideoPal),
            )?;
            refined_carrier_hz = acquisition.measured_carrier_hz;
            println!(
                "  → Standard {:?}: {:?} ({:.0}% classifier confidence)",
                acquisition.standard.evidence,
                acquisition.standard.standard,
                acquisition.classifier_confidence * 100.0
            );
            if debug {
                eprintln!(
                    "[DEBUG] standard record: {:.1} ms",
                    record.samples().len() as f64 / sample_rate_u32 as f64 * 1e3
                );
            }
            acquisition.standard.standard
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
    let observed_timing_fields = Arc::new(std::sync::atomic::AtomicU64::new(0));

    let result = run_viewer_pipeline(
        resolved_channels,
        refined_carrier_hz,
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
        observed_timing_fields.clone(),
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
            let detector = orecchiette_fpv_drone_analog_rs::scanner::receiver_detector();
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

                            let produced =
                                observed_timing_fields.load(std::sync::atomic::Ordering::Relaxed);
                            let decoding = produced > frames_at_last_check;
                            frames_at_last_check = produced;

                            if lock.observe_detections(&hits, tuned_carrier_hz, decoding) {
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
    // The carrier re-measured at the tuned centre, if it was. Names the
    // channel; see the use site for why it does not re-tune.
    refined_carrier_hz: Option<f64>,
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
    observed_timing_fields: Arc<std::sync::atomic::AtomicU64>,
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
        let worker_observed = Arc::clone(&observed_timing_fields);
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
        let plan = DecodePlan::new(sample_rate, fm_deviation, bandwidth_hz, demod_kind.into())?;
        let decim = plan.decimation();
        let work_rate = plan.work_rate();
        let use_pll = plan.use_pll();
        let unconstrained = decode_decimation(sample_rate, plan.ddc_cutoff_hz());
        if decim < unconstrained {
            println!(
                "  → --demod pll holds the decode rate at {:.2} MSPS instead of {:.2} MSPS",
                work_rate as f64 / 1e6,
                (sample_rate / unconstrained as u32) as f64 / 1e6
            );
        }
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

        let reconstructor = FrameReconstructor::new(work_rate, is_pal, fm_deviation, debug);
        let width = reconstructor.width;
        let height = reconstructor.height;

        let type_name = if is_pal { "PAL" } else { "NTSC" };
        let absolute_freq_mhz = (rf_center_freq + freq_offset as f64) / 1_000_000.0;
        // Name the channel from what the signal measured *here*, not
        // from the frequency the sweep told us to tune to. Those are the
        // same thing only when the sweep's localization was good enough
        // to pick between channels a megahertz apart, and at a low probe
        // SNR it is not: a 5865 MHz signal localized to 5871.775 and was
        // labelled B8 (5866) instead of A1 (5865).
        //
        // The tuning itself is left alone. The refined estimate carries
        // its own bias — localization assumes a particular FM deviation,
        // and against the raw spectrum it reads about 1.3 MHz low — so
        // it is good enough to choose between candidate channels but not
        // to re-centre a capture that is already decoding.
        // What every piece of UI should call this channel. Distinct
        // from the tuned frequency, which the sweep may have snapped to
        // the wrong neighbour — the window title used this while the
        // video overlay and the snapshot filename still used the tuned
        // value, so one run showed "A1" in the title bar and "F7" on
        // the picture.
        let display_mhz = refined_carrier_hz
            .map(|hz| snap_to_nearest_fpv_channel(hz) / 1_000_000.0)
            .unwrap_or(absolute_freq_mhz);
        let channel_name = get_fpv_channel_name(display_mhz);
        if let Some(refined) = refined_carrier_hz
            && (display_mhz - absolute_freq_mhz).abs() > 0.5
        {
            println!(
                "  → Measured carrier {:.3} MHz here: this is {}, not the {} the sweep snapped to",
                refined / 1e6,
                get_fpv_channel_name(display_mhz).unwrap_or("an unlisted channel"),
                get_fpv_channel_name(absolute_freq_mhz).unwrap_or("frequency"),
            );
        }
        let window_title = if let Some(ch) = channel_name {
            format!("{} · Channel {}", type_name, ch)
        } else {
            format!("{} · {:.2} MHz", type_name, display_mhz)
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

        windows.push((window, width, height, is_pal, display_mhz));
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

        let config = DecodeWorkerConfig {
            decoder: DecoderConfig {
                plan,
                frequency_offset_hz: freq_offset,
                standard: orecchiette_fpv_drone_analog_rs::timing::Standard::from_is_pal(is_pal),
                deemphasis_tau_s: deemphasis_tau,
                temporal_window,
                debug,
            },
            display_mhz,
            denoise_model: Some(denoise_model),
        };

        DecodeWorker::spawn(
            config,
            iq_rx,
            producer_slot,
            frame_tx,
            recycle_rx,
            worker_frames,
            worker_observed,
            snap_tx,
            #[cfg(feature = "neural-vsr")]
            denoise_on,
        )?;
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
            let (window, width, height, is_pal, display_mhz) = &mut windows[i];

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
                            ch_input = ChannelInputState::Typing(c.to_string());
                            break;
                        }
                    }
                }
                ChannelInputState::Typing(mut name) => {
                    let keys = window.get_keys_pressed(minifb::KeyRepeat::No);
                    for key in keys {
                        if key == Key::Escape {
                            ch_input = ChannelInputState::Idle;
                            break;
                        }
                        if key == Key::Backspace {
                            name.pop();
                            ch_input = if name.is_empty() {
                                ChannelInputState::WaitingFirst
                            } else {
                                ChannelInputState::Typing(name)
                            };
                            break;
                        }
                        let submit = key == Key::Enter;
                        if !submit {
                            let Some(c) = key_to_char(key) else {
                                continue;
                            };
                            name.push(c);
                        }
                        let exact = lookup_channel_by_name(&name);
                        let has_longer = orecchiette_fpv_drone_analog_rs::bands::channel_catalog()
                            .iter()
                            .any(|channel| {
                                let full = channel.name();
                                full.len() > name.len() && full.starts_with(&name)
                            });
                        if !submit && has_longer {
                            ch_input = ChannelInputState::Typing(name);
                            break;
                        }
                        if let Some(freq) = exact {
                            tune_freq.store(freq, std::sync::atomic::Ordering::Relaxed);
                            exit_reason.store(5, std::sync::atomic::Ordering::Relaxed);
                        } else {
                            ch_input_flash_msg = format!("Unknown channel: {name}");
                            ch_input_flash_until = Some(Instant::now() + Duration::from_secs(2));
                            ch_input = ChannelInputState::Idle;
                        }
                        break;
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
                    let channel_name = get_fpv_channel_name(*display_mhz);
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
                        format!("{} · {:.2} MHz [BW]{}", format_str, display_mhz, dn)
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
                        ChannelInputState::Typing(name) => {
                            draw_text_with_bg(
                                &mut display_buffers[i],
                                *width,
                                *height,
                                10,
                                30,
                                &format!("CH: {name}_  ENTER TO TUNE"),
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

    /// The review's case: a noisy fixture at 61.44 MSPS first detects
    /// on packet 3, which a two-packet fast pass can never reach. Some
    /// sweep has to spend more, and no per-hop prediction can decide
    /// which — so the cadence does it unconditionally.
    #[cfg(feature = "aaronia")]
    #[test]
    fn a_sensitive_sweep_comes_round_and_reaches_packet_three() {
        let wide = 61_440_000.0;
        let budgets: Vec<usize> = (0..AARONIA_SENSITIVE_EVERY * 2)
            .map(|n| aaronia_sweep_budget(n, wide).packets_per_hop)
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
        let b = aaronia_sweep_budget(AARONIA_SENSITIVE_EVERY - 1, 61_440_000.0);
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
            .filter(|&n| aaronia_sweep_budget(n, wide).packets_per_hop == DETECT_PACKETS_PER_HOP)
            .count();
        assert_eq!(fast as u32, AARONIA_SENSITIVE_EVERY - 1);
    }

    /// A narrow capture already reads the sensitive count every sweep,
    /// so the cadence must not make it slower.
    #[cfg(feature = "aaronia")]
    #[test]
    fn a_narrow_capture_is_unaffected_by_the_cadence() {
        for n in 0..AARONIA_SENSITIVE_EVERY * 2 {
            let b = aaronia_sweep_budget(n, 15_360_000.0);
            assert_eq!(b.packets_per_hop, DETECT_PACKETS_NARROW);
            assert_eq!(b.dwell, AARONIA_SCAN_DWELL);
        }
    }

    /// The budget must fit the dwell, or the hop ends mid-budget and
    /// the extra packets are never read.
    #[cfg(feature = "aaronia")]
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

    /// Snapping to the nearest planned hop absorbs hardware PLL fractional offsets.
    #[test]
    fn snap_to_nearest_hop_absorbs_fractional_offset() {
        let hops = vec![1_200_000_000.0, 3_312_500_000.0, 5_800_000_000.0];
        // Typical Aaronia Spectran V6 ECO PLL frequency with -2.6 kHz offset
        let observed = 3_312_497_366.4;
        assert_eq!(snap_to_nearest_hop(observed, &hops), 3_312_500_000);
        // Distant unexpected frequency is kept as-is
        let foreign = 2_400_000_000.0;
        assert_eq!(snap_to_nearest_hop(foreign, &hops), 2_400_000_000);
    }
}
