use clap::{Args as ClapArgs, Parser, Subcommand};
use fpv_drone_analog_rs::ddc::StreamingDDC;
use fpv_drone_analog_rs::demod::fm_demod;
use fpv_drone_analog_rs::detector::{AnalogFpvDetector, FpvDetector};
use fpv_drone_analog_rs::lookup_channel_by_name;
use fpv_drone_analog_rs::types::SignalType;
use fpv_drone_analog_rs::video::FrameReconstructor;
use minifb::{Key, Window, WindowOptions};
use num_complex::Complex;
use sdr_source_rs::{DwellAdvice, SdrSource, SourceConfig};
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
    #[arg(long)]
    debug: bool,
    #[arg(long, short = 'm')]
    monochrome: bool,
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
    /// FPV channel name (A1–A8, B1–B8, E1–E8, F1–F8, R1–R8, L1–L8)
    /// or raw frequency in Hz (e.g. 5865000000).
    ///
    /// Omit to auto-scan: the viewer captures wideband, detects all
    /// active analog FPV signals, and opens a window for each.
    #[arg(long)]
    channel: Option<String>,
    /// Force video standard instead of auto-detecting.
    #[arg(long, value_parser = parse_standard)]
    standard: Option<SignalType>,
    /// Override sample rate (Hz). By default the viewer queries the
    /// SDR for its maximum supported rate.
    #[arg(long)]
    sample_rate: Option<f64>,
    /// FM deviation (Hz). Defaults to 5 MHz for live SDR.
    #[arg(long, default_value_t = 5_000_000.0)]
    fm_deviation: f32,
    /// Enable debug mode (saves 3 frames to PNG, prints metrics, and auto-exits)
    #[arg(long)]
    debug: bool,
    /// Scan mode when auto-scanning (no --channel).
    ///   58  = 5.8 GHz band only (default, fast)
    ///   ua  = Ukraine: 900 MHz + 1.2 GHz + 2.4 GHz + 5.3–5.9 GHz
    #[arg(long, value_parser = parse_scan_mode, default_value = "58")]
    scan_mode: ScanMode,
    /// Number of fields kept in the temporal denoise / dropout-repair
    /// history (Phase A-E). 5 (default) gives ≈ +7 dB SNR on static
    /// scenes with ~83 ms latency; 1 disables temporal processing.
    /// Clamped to the range 1–8 (higher values allocate history the
    /// denoise never reads).
    #[arg(long, default_value_t = 5)]
    temporal_window: usize,
}

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

#[derive(Debug, Clone, Copy)]
enum ScanMode {
    /// Standard 5.8 GHz FPV band only (5.6–5.95 GHz).
    Band58,
    /// Ukraine theatre: 900 MHz, 1.2 GHz, 2.4 GHz, LowBand, and 5.8 GHz.
    Ua,
}

fn parse_scan_mode(s: &str) -> Result<ScanMode, String> {
    match s.to_lowercase().as_str() {
        "58" | "5.8" | "5800" => Ok(ScanMode::Band58),
        "ua" | "ukraine" => Ok(ScanMode::Ua),
        _ => Err(format!(
            "unknown scan mode '{}'; expected '58' (5.8 GHz only) or 'ua' (Ukraine full-spectrum)",
            s
        )),
    }
}

/// Build the hop frequency list for the scan loop based on the
/// selected scan mode and the SDR's instantaneous bandwidth.
///
/// The list is ordered low→high so the USRP retunes monotonically
/// (minimises PLL re-lock time on most synthesisers).
///
/// **UA mode** covers every confirmed analog video TX band observed
/// in the Ukraine theatre (2024-2025), modelled after the Chuyka 3.0
/// detector's coverage plus the newer 6-7 GHz evasion band:
///
/// | Band           | Range (MHz)     | Notes                                              |
/// |----------------|----------------|----------------------------------------------------|
/// | 1.2 GHz        | 1080 – 1360    | Long-range analog FPV, 1240-1300 amateur segment   |
/// | 3.3 GHz        | 2870 – 4080    | Mid-range alternative, Chuyka Band 2               |
/// | 5.3 GHz Low    | 5300 – 5640    | LowBand (L1-L8) FPV channels                       |
/// | 5.8 GHz        | 5645 – 5945    | Standard FPV (A/B/E/F/R bands)                     |
/// | 6-7 GHz        | 6100 – 7300    | New evasion band (PEAK THOR T67 VTX, 6.1-7.2 GHz)  |
fn build_scan_hops(mode: ScanMode, sample_rate: f64) -> Vec<f64> {
    let step = sample_rate * 0.8; // 80% of BW per hop to avoid filter rolloff edges
    let mut hops = Vec::new();

    let mut sweep = |start: f64, end: f64| {
        let mut f = start;
        while f <= end {
            hops.push(f);
            f += step;
        }
    };

    match mode {
        ScanMode::Band58 => {
            // Standard 5.8 GHz FPV: 5.645 – 5.945 GHz
            sweep(5_645_000_000.0, 5_945_000_000.0);
        }
        ScanMode::Ua => {
            // ── 1.2 GHz (long-range analog FPV, very common in UA) ──
            // Standard 1.2 GHz VTX range plus amateur 1240-1300 MHz.
            sweep(1_080_000_000.0, 1_360_000_000.0);

            // ── 3.3 GHz (mid-range FPV video, Chuyka Band 2) ──
            // Covers 2870-4080 MHz per Chuyka 3.0 spec.
            sweep(2_870_000_000.0, 4_080_000_000.0);

            // ── 5.3 GHz LowBand (L-band FPV: 5.333–5.613 GHz) ──
            sweep(5_300_000_000.0, 5_640_000_000.0);

            // ── 5.8 GHz (standard FPV: A/B/E/F/R bands) ──
            sweep(5_645_000_000.0, 5_945_000_000.0);

            // ── 6-7 GHz (new evasion band, PEAK THOR T67 VTX) ──
            // 6.1-7.2 GHz; Ukraine forces shifted here in 2024-2025
            // to bypass Russian EW tuned to ≤6 GHz.
            sweep(6_100_000_000.0, 7_300_000_000.0);
        }
    }

    hops
}

/// Resolve --channel to a centre frequency in Hz.
fn resolve_channel(ch: &str) -> anyhow::Result<f64> {
    // Try channel name first
    if let Some(freq) = lookup_channel_by_name(ch) {
        return Ok(freq as f64);
    }
    // Try raw frequency
    if let Ok(freq) = ch.parse::<f64>() {
        if freq > 1e6 {
            return Ok(freq);
        }
    }
    anyhow::bail!(
        "unrecognised channel '{}'; expected A1–A8, B1–B8, E1–E8, F1–F8, R1–R8, L1–L8, or a frequency in Hz",
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

#[derive(Subcommand, Debug)]
enum AaroniaCmd {
    /// Stream from an RTSA HTTP server block.
    Http(AaroniaHttpArgs),
    /// Stream via the native AARTSAAPI SDK.
    Sdk(AaroniaSdkArgs),
}

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
}

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

/// Default centre of the 5.8 GHz FPV band for wideband scanning.
const SCAN_CENTER_HZ: f64 = 5_800_000_000.0;

/// Default Aaronia span when no --sample-rate is given (V6 nominal RTBW).
const AARONIA_DEFAULT_SPAN: f64 = 92_000_000.0;

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
        }

        if let Some(captures) = meta_json.get("captures").and_then(|v| v.as_array()) {
            if let Some(first_cap) = captures.first() {
                if let Some(freq) = first_cap.get("core:frequency").and_then(|v| v.as_f64()) {
                    rf_center_freq = freq;
                }
            }
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
        .chunks_exact(8)
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
            if detected_type != SignalType::Unknown {
                println!(
                    "  → Auto-detected: {:?} (confidence {:.0}%)",
                    detected_type,
                    confidence * 100.0
                );
                detected_type
            } else {
                println!("  → Could not detect standard, defaulting to NTSC");
                SignalType::AnalogVideoNtsc
            }
        };
        resolved_channels.push((sig_type, *freq_offset, fm_deviation * 2.0));
    }

    let exit_reason = Arc::new(std::sync::atomic::AtomicU8::new(0));
    let tune_freq = Arc::new(std::sync::atomic::AtomicU64::new(0));

    let _ = run_viewer_pipeline(
        resolved_channels,
        sample_rate,
        fm_deviation,
        rf_center_freq,
        args.debug,
        args.temporal_window,
        exit_reason,
        tune_freq,
        move |channel_txs, _exit_flag| {
            let _ = file.seek(SeekFrom::Start(0));
            let mut active_txs = channel_txs;
            loop {
                if active_txs.is_empty() {
                    return;
                }
                match file.read(&mut buf) {
                    Ok(0) => {
                        let _ = file.seek(SeekFrom::Start(0));
                    }
                    Ok(bytes_read) => {
                        let samples_read = bytes_read / 8;
                        let pairs: &[[f32; 2]] = bytemuck::cast_slice(&buf[..samples_read * 8]);
                        let iq_vec: Vec<Complex<f32>> =
                            pairs.iter().map(|&[re, im]| Complex::new(re, im)).collect();
                        let pooled = sdr_source_rs::PooledIqBuffer::new_unpooled(iq_vec);
                        let arc_chunk = Arc::new(pooled);
                        active_txs.retain(|tx| tx.send(Arc::clone(&arc_chunk)).is_ok());
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
    use sdr_usrp_rs::UsrpSource;

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

    let mut explicit_freq = if let Some(ref ch) = args.live.channel {
        Some(resolve_channel(ch)?)
    } else {
        None
    };

    let fm_deviation = args.live.fm_deviation;

    loop {
        let (center_freq, target_mode) = match current_mode {
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
                (freq, ViewerMode::SingleChannel)
            }
            ViewerMode::Scan => {
                let hop_freqs = build_scan_hops(args.live.scan_mode, current_sample_rate);
                let mode_label = match args.live.scan_mode {
                    ScanMode::Band58 => "5.8 GHz",
                    ScanMode::Ua => "UA (1.2 + 3.3 + 5.3-5.9 + 6-7 GHz)",
                };
                println!(
                    "Wideband scan requested [{}]. {} hops at {:.1} MHz step...",
                    mode_label,
                    hop_freqs.len(),
                    current_sample_rate * 0.8 / 1e6
                );

                let config = SourceConfig {
                    sample_rate_hz: current_sample_rate,
                    channels_hz: hop_freqs.clone(),
                    // Only need one 2.6ms chunk per hop for detection;
                    // 10ms allows for retune settling + one full capture.
                    dwell_min: Duration::from_millis(10),
                    dwell_max: Duration::from_millis(10),
                    dwell_extension: Duration::ZERO,
                };
                let advice = Arc::new(NoOpDwell) as Arc<dyn DwellAdvice>;
                let source = Box::new(UsrpSource {
                    args: args.args.clone(),
                    gain_db: args.gain,
                    antenna: args.antenna.clone(),
                });

                let handle = source.start(config, advice.clone())?;
                let detector = AnalogFpvDetector::default();
                let mut best_hit: Option<(f64, f32, SignalType)> = None;
                let mut seen_freqs = std::collections::HashSet::new();
                let mut sweeps_completed = 0;
                let mut last_processed_freq = 0;

                let found_freq = 'scan_loop: loop {
                    if let Ok(packet) = handle.receiver.recv_timeout(Duration::from_millis(1000)) {
                        let center = packet.center_frequency_hz as u64;

                        // We only need to run the expensive wideband sweep once per hop.
                        // Drain any remaining packets from this dwell to prevent the SDR buffer from overflowing.
                        if center == last_processed_freq {
                            continue;
                        }
                        last_processed_freq = center;

                        seen_freqs.insert(center);

                        let results = detector.detect_from_iq(
                            &packet.samples,
                            center,
                            current_sample_rate as u32,
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
                            if let Some((_, best_rssi, _)) = best_hit {
                                if res.rssi_dbm > best_rssi {
                                    let candidates =
                                        get_candidate_fpv_channels(res.frequency_hz as f64);
                                    let has_unskipped = candidates
                                        .iter()
                                        .any(|&c| !skipped_freqs.contains(&(c.round() as u64)));
                                    if has_unskipped {
                                        best_hit = Some((
                                            res.frequency_hz as f64,
                                            res.rssi_dbm,
                                            res.signal_type,
                                        ));
                                    }
                                }
                            } else {
                                let candidates =
                                    get_candidate_fpv_channels(res.frequency_hz as f64);
                                let has_unskipped = candidates
                                    .iter()
                                    .any(|&c| !skipped_freqs.contains(&(c.round() as u64)));
                                if has_unskipped {
                                    best_hit = Some((
                                        res.frequency_hz as f64,
                                        res.rssi_dbm,
                                        res.signal_type,
                                    ));
                                }
                            }
                        }
                        // If we've seen all frequencies in the hop list at least once
                        if seen_freqs.len() >= hop_freqs.len() {
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
                                    break 'scan_loop snapped_freq;
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
                                    if let Ok(packet) =
                                        ft_handle.receiver.recv_timeout(Duration::from_millis(1000))
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

                                            if let Some((best_freq, best_conf, _)) = ft_best_hit {
                                                if best_conf > 0.1 {
                                                    println!(
                                                        "Fine-tuning complete. Selected exact channel: {} ({:.3} MHz)",
                                                        get_fpv_channel_name(best_freq / 1e6)
                                                            .unwrap_or("Unknown"),
                                                        best_freq / 1e6
                                                    );
                                                    break 'scan_loop best_freq;
                                                }
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
                                            break 'scan_loop snapped_freq;
                                        }
                                    } else {
                                        println!("SDR timed out during fine-tune. Retrying...");
                                    }
                                }
                            } else {
                                sweeps_completed += 1;
                                if sweeps_completed % 3 == 0 {
                                    println!("Still sweeping... no signals found yet.");
                                }
                                // Reset for the next sweep
                                seen_freqs.clear();
                            }
                        }
                    } else {
                        // Timeout receiving packet, maybe SDR died or is taking too long
                        println!("SDR timed out during scan. Retrying...");
                    }
                };
                explicit_freq = Some(found_freq);
                (found_freq, ViewerMode::SingleChannel)
            }
        };

        let source = Box::new(UsrpSource {
            args: args.args.clone(),
            gain_db: args.gain,
            antenna: args.antenna.clone(),
        });

        match run_live(
            source,
            current_sample_rate,
            fm_deviation,
            center_freq,
            target_mode,
            args.live.standard,
            args.live.debug,
            args.live.temporal_window,
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
    use sdr_hackrf_rs::{HACKRF_MAX_SAMPLE_RATE_HZ, HackRfSource};

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
    let mut explicit_freq = match args.live.channel {
        Some(ref ch) => Some(resolve_channel(ch)?),
        None => None,
    };

    loop {
        let center_freq = match explicit_freq {
            Some(f) => f,
            None => match hackrf_scan_for_channel(&args, sample_rate, &skipped_freqs)? {
                // Used directly as this iteration's centre; in auto-scan
                // mode the RunLiveResult handling below decides whether to
                // re-scan, so we don't need to stash it in `explicit_freq`.
                Some(f) => f,
                None => {
                    println!("No analog FPV signal found in the scan band. Retrying...");
                    continue;
                }
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

        match run_live(
            source,
            sample_rate,
            fm_deviation,
            center_freq,
            ViewerMode::SingleChannel,
            args.live.standard,
            args.live.debug,
            args.live.temporal_window,
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

/// Single-stage coarse scan for the HackRF: sweep the band, run the
/// wideband detector on one chunk per hop, and snap the strongest
/// (non-blacklisted) hit to the nearest FPV channel. Returns `None` if
/// the band is empty.
///
/// This is intentionally simpler than the USRP path's two-stage
/// coarse-then-fine-tune sweep — the single coarse snap is enough to
/// land on a channel; a HackRF fine-tune stage is a future addition.
fn hackrf_scan_for_channel(
    args: &HackrfArgs,
    sample_rate: f64,
    skipped_freqs: &std::collections::HashSet<u64>,
) -> anyhow::Result<Option<f64>> {
    use sdr_hackrf_rs::HackRfSource;

    let hop_freqs = build_scan_hops(args.live.scan_mode, sample_rate);
    println!(
        "HackRF wideband scan: {} hops at {:.1} MHz step...",
        hop_freqs.len(),
        sample_rate * 0.8 / 1e6
    );
    let config = SourceConfig {
        sample_rate_hz: sample_rate,
        channels_hz: hop_freqs.clone(),
        dwell_min: Duration::from_millis(15),
        dwell_max: Duration::from_millis(15),
        dwell_extension: Duration::ZERO,
    };
    let advice = Arc::new(NoOpDwell) as Arc<dyn DwellAdvice>;
    let source = Box::new(HackRfSource {
        lna_gain: args.lna_gain,
        vga_gain: args.vga_gain,
        amp_enable: args.amp,
        bias_tee: args.bias_tee,
    });
    let handle = source.start(config, advice)?;
    let detector = AnalogFpvDetector::default();

    let mut best_hit: Option<(f64, f32)> = None; // (raw freq Hz, rssi dBm)
    let mut seen = std::collections::HashSet::new();
    let mut last_freq = 0u64;

    loop {
        match handle.receiver.recv_timeout(Duration::from_millis(2000)) {
            Ok(packet) => {
                let center = packet.center_frequency_hz as u64;
                // One detection pass per hop; drain the rest of the dwell.
                if center == last_freq {
                    continue;
                }
                last_freq = center;
                seen.insert(center);

                for res in detector.detect_from_iq(&packet.samples, center, sample_rate as u32) {
                    let snapped = snap_to_nearest_fpv_channel(res.frequency_hz as f64);
                    if skipped_freqs.contains(&(snapped.round() as u64)) {
                        continue;
                    }
                    let better = best_hit.map(|(_, r)| res.rssi_dbm > r).unwrap_or(true);
                    if better {
                        best_hit = Some((res.frequency_hz as f64, res.rssi_dbm));
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

                if seen.len() >= hop_freqs.len() {
                    (handle.stop)();
                    std::thread::sleep(Duration::from_millis(150));
                    return Ok(best_hit.map(|(f, _)| snap_to_nearest_fpv_channel(f)));
                }
            }
            Err(_) => {
                // SDR went quiet — return whatever we found this sweep.
                (handle.stop)();
                return Ok(best_hit.map(|(f, _)| snap_to_nearest_fpv_channel(f)));
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
//  LIVE AARONIA MODE
// ═══════════════════════════════════════════════════════════════════

fn run_live_aaronia(cmd: AaroniaCmd) -> anyhow::Result<()> {
    use sdr_aaronia_rs::{AaroniaBackend, AaroniaSdrSource};

    let (source, live, label) = match cmd {
        AaroniaCmd::Http(h) => {
            let center_freq = h
                .live
                .channel
                .as_ref()
                .map(|ch| resolve_channel(ch))
                .transpose()?
                .unwrap_or(SCAN_CENTER_HZ);
            let source = Box::new(AaroniaSdrSource {
                backend: AaroniaBackend::Http(h.url.clone()),
                center_frequency_hz: center_freq,
                reference_level_dbm: h.ref_level,
                block_size: 65_536,
            });
            (
                source as Box<dyn SdrSource>,
                h.live,
                format!("Aaronia HTTP ({})", h.url),
            )
        }
        AaroniaCmd::Sdk(s) => {
            let center_freq = s
                .live
                .channel
                .as_ref()
                .map(|ch| resolve_channel(ch))
                .transpose()?
                .unwrap_or(SCAN_CENTER_HZ);
            let source = Box::new(AaroniaSdrSource {
                backend: AaroniaBackend::Sdk {
                    serial: s.serial.clone(),
                },
                center_frequency_hz: center_freq,
                reference_level_dbm: s.ref_level,
                block_size: 65_536,
            });
            (
                source as Box<dyn SdrSource>,
                s.live,
                "Aaronia SDK".to_string(),
            )
        }
    };

    let sample_rate = live.sample_rate.unwrap_or(AARONIA_DEFAULT_SPAN);
    let fm_deviation = live.fm_deviation;
    let center_freq = live
        .channel
        .as_ref()
        .map(|ch| resolve_channel(ch))
        .transpose()?
        .unwrap_or(SCAN_CENTER_HZ);
    let mode = if live.channel.is_some() {
        ViewerMode::SingleChannel
    } else {
        ViewerMode::Scan
    };

    println!(
        "{}: {:.2} MSPS at {:.3} MHz",
        label,
        sample_rate / 1e6,
        center_freq / 1e6
    );

    let _ = run_live(
        source,
        sample_rate,
        fm_deviation,
        center_freq,
        mode,
        live.standard,
        live.debug,
        live.temporal_window,
    )?;
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
    center_freq: f64,
    mode: ViewerMode,
    forced_standard: Option<SignalType>,
    debug: bool,
    temporal_window: usize,
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
    let first_packet = handle
        .receiver
        .recv()
        .map_err(|_| anyhow::anyhow!("SDR source closed before delivering any samples"))?;
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

    // Detect channels
    let resolved_channels = match mode {
        ViewerMode::SingleChannel => {
            // Single channel at DC
            let sig_type = if let Some(st) = forced_standard {
                println!("Forced standard: {:?}", st);
                st
            } else {
                println!("Auto-detecting video standard...");
                let detector = AnalogFpvDetector::default();
                let (detected, confidence) =
                    detector.detect_sync_pulses(&first_packet.samples, sample_rate_u32);
                if detected != SignalType::Unknown {
                    println!(
                        "  → Detected: {:?} ({:.0}% confidence)",
                        detected,
                        confidence * 100.0
                    );
                    detected
                } else {
                    println!("  → No standard detected, defaulting to PAL");
                    SignalType::AnalogVideoPal
                }
            };
            vec![(sig_type, 0.0f32, fm_deviation * 2.0)] // signal is at DC (SDR tuned directly)
        }
        ViewerMode::Scan => {
            println!("Scanning for analog FPV signals...");
            let detector = AnalogFpvDetector::default();
            let results = detector.detect_from_iq(
                &first_packet.samples,
                actual_center as u64,
                sample_rate_u32,
            );
            if results.is_empty() {
                println!("No analog FPV signals detected in the capture bandwidth.");
                println!("Try specifying a channel with --channel A1");
                std::process::exit(0);
            }
            let mut channels = Vec::new();
            for res in &results {
                let freq_offset = res.frequency_hz as f64 - actual_center;
                let sig_type = res.signal_type;
                let ch_name =
                    get_fpv_channel_name(res.frequency_hz as f64 / 1e6).unwrap_or("Unknown");
                println!(
                    "  → Found {:?} at {:.3} MHz (channel {}, confidence {:.0}%, offset {:.2} MHz)",
                    sig_type,
                    res.frequency_hz as f64 / 1e6,
                    ch_name,
                    res.confidence * 100.0,
                    freq_offset / 1e6
                );
                channels.push((sig_type, freq_offset as f32, res.bandwidth_hz as f32));
            }
            channels
        }
    };

    let rf_center_freq = actual_center;

    // Feed the first chunk into the pipeline, then continue from the receiver
    let first_samples = first_packet.samples;
    let receiver = handle.receiver.clone();
    let handle_stop = handle.stop; // Take ownership of the stop closure

    let exit_reason = Arc::new(std::sync::atomic::AtomicU8::new(0));
    let tune_freq = Arc::new(std::sync::atomic::AtomicU64::new(0));

    let result = run_viewer_pipeline(
        resolved_channels,
        sample_rate_u32,
        fm_deviation,
        rf_center_freq,
        debug,
        temporal_window,
        exit_reason.clone(),
        tune_freq,
        move |channel_txs, exit_flag| {
            let mut active_txs = channel_txs;
            let detector = AnalogFpvDetector::default();

            let mut packets = 0;
            let mut consecutive_lost = 0;
            let mut dropped_chunks = 0u64;

            let mut overrun_count = 0;
            let mut last_overrun_clear = Instant::now();

            // Send first chunk
            {
                let arc_chunk = Arc::new(first_samples);
                active_txs.retain(|tx| tx.send(Arc::clone(&arc_chunk)).is_ok());
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
                            if overrun_count >= 2 {
                                exit_flag.store(2, std::sync::atomic::Ordering::Relaxed);
                                break;
                            }
                        }
                        if last_overrun_clear.elapsed() >= Duration::from_secs(60) {
                            overrun_count = 0;
                            last_overrun_clear = Instant::now();
                        }

                        // Check signal lock periodically (approx twice a second)
                        if packets % 400 == 0 {
                            let (_, conf) =
                                detector.detect_sync_pulses(&packet.samples, sample_rate_u32);
                            if conf < 0.15 {
                                consecutive_lost += 1;
                                if consecutive_lost >= 4 {
                                    exit_flag.store(1, std::sync::atomic::Ordering::Relaxed);
                                    break;
                                }
                            } else {
                                consecutive_lost = 0;
                            }
                        }

                        let arc_chunk = Arc::new(packet.samples);
                        active_txs.retain(|tx| {
                            match tx.try_send(Arc::clone(&arc_chunk)) {
                                Ok(_) => true,
                                // Worker behind: drop this chunk but keep
                                // the channel. Dropping raw IQ breaks DDC
                                // continuity (a visible glitch), so we
                                // count it and surface the cost in debug.
                                Err(mpsc::TrySendError::Full(_)) => {
                                    dropped_chunks += 1;
                                    true
                                }
                                Err(mpsc::TrySendError::Disconnected(_)) => false, // UI closed
                            }
                        });
                        if debug && packets % 400 == 0 && dropped_chunks > 0 {
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
    rf_center_freq: f64,
    debug: bool,
    temporal_window: usize,
    exit_reason: Arc<std::sync::atomic::AtomicU8>,
    tune_freq: Arc<std::sync::atomic::AtomicU64>,
    reader_fn: F,
) -> anyhow::Result<RunLiveResult>
where
    F: FnOnce(
            Vec<mpsc::SyncSender<Arc<sdr_source_rs::PooledIqBuffer>>>,
            Arc<std::sync::atomic::AtomicU8>,
        ) + Send
        + 'static,
{
    let mut channel_txs = Vec::new();
    let mut frame_rxs = Vec::new();
    let mut recycle_txs = Vec::new();
    let mut windows = Vec::new();
    let mut display_buffers = Vec::new();

    for (sig_type, freq_offset, bandwidth_hz) in resolved_channels {
        let (iq_tx, iq_rx) = mpsc::sync_channel::<Arc<sdr_source_rs::PooledIqBuffer>>(10);
        let (frame_tx, frame_rx) = mpsc::sync_channel::<Vec<u32>>(2);
        // Frame-buffer recycle pool: the UI hands spent frame buffers
        // back here so the decode worker can refill them in place
        // (`reconstruct_frame_into`) instead of allocating + zeroing a
        // fresh ~1.4 MB Vec every field. Both ends use non-blocking
        // try_*; on miss the worker just allocates and on a full return
        // channel the UI drops the buffer — so there's no deadlock path.
        let (recycle_tx, recycle_rx) = mpsc::sync_channel::<Vec<u32>>(3);

        let is_pal = sig_type == SignalType::AnalogVideoPal;

        let mut reconstructor = FrameReconstructor::new(sample_rate, is_pal, fm_deviation, debug)
            .with_temporal_window(temporal_window);
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
        println!("  → Window: {} ({}×{})", window_title, width, height);
        let window = Window::new(&window_title, width, height, WindowOptions::default())?;

        windows.push((window, width, height, is_pal, absolute_freq_mhz));
        display_buffers.push(vec![0u32; width * height]);
        channel_txs.push(iq_tx);
        frame_rxs.push(frame_rx);
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
            // Floor the cutoff at `fm_deviation`: in scan mode
            // `bandwidth_hz` comes from the detector, and an implausibly
            // small detected bandwidth would otherwise collapse the FIR
            // passband and render a black/garbage frame with no clue why.
            let ddc_cutoff = (bandwidth_hz / 2.0)
                .min(fm_deviation + LUMA_HEADROOM_HZ)
                .max(fm_deviation);
            let mut ddc = StreamingDDC::new(freq_offset, sample_rate, ddc_cutoff);
            let mut shifted_iq: Vec<Complex<f32>> = Vec::new();
            let mut demod_buffer: Vec<f32> = Vec::new();
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
            // One-shot guard for the PAL demod-range probe: previously
            // gated on `frame_count == 0`, which prints every chunk when
            // the field never locks (e.g. no signal) — a console flood.
            let mut pal_debug_done = false;

            while let Ok(iq_chunk) = iq_rx.recv() {
                shifted_iq.clear();
                ddc.process_into(&iq_chunk, &mut shifted_iq);
                let mut new_demod = fm_demod(&shifted_iq);

                if debug && is_pal && !pal_debug_done && !new_demod.is_empty() {
                    let min = new_demod.iter().cloned().fold(f32::INFINITY, f32::min);
                    let max = new_demod.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let mean = new_demod.iter().sum::<f32>() / new_demod.len() as f32;
                    eprintln!(
                        "[PAL DEBUG] first demod chunk: min={:.4} max={:.4} mean={:.4}",
                        min, max, mean
                    );
                    pal_debug_done = true;
                }

                demod_buffer.append(&mut new_demod);

                // None ends the loop (no full field buffered yet);
                // `frame_buf` is left untouched on the None paths (they
                // return before writing it), so it's safe to keep for the
                // next chunk.
                while let Some(consumed) = reconstructor
                    .reconstruct_frame_into(&demod_buffer[demod_start..], &mut frame_buf)
                {
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
                    let metrics_due = debug && (frame_count <= 3 || frame_count % 30 == 0);
                    if metrics_due {
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
                        let path = format!("fpv_frame_{}.png", frame_count);
                        let mut rgb_buf = vec![0u8; width * height * 3];
                        for (i, &pixel) in frame_buf.iter().enumerate() {
                            rgb_buf[i * 3] = ((pixel >> 16) & 0xFF) as u8;
                            rgb_buf[i * 3 + 1] = ((pixel >> 8) & 0xFF) as u8;
                            rgb_buf[i * 3 + 2] = (pixel & 0xFF) as u8;
                        }
                        let _ = snap_tx.send((path, rgb_buf, width as u32, height as u32));
                    }

                    demod_start += consumed;
                    // Swap in a recycled (or fresh) buffer and ship the
                    // just-filled one to the UI.
                    let next = recycle_rx
                        .try_recv()
                        .unwrap_or_else(|_| vec![0u32; width * height]);
                    if frame_tx
                        .send(std::mem::replace(&mut frame_buf, next))
                        .is_err()
                    {
                        return;
                    }
                }

                // Amortised compaction: drop the consumed prefix in a
                // single memmove once it's grown past the threshold,
                // rather than once per field.
                if demod_start > 2_000_000 {
                    demod_buffer.drain(0..demod_start);
                    demod_start = 0;
                }
                // Safety valve: if reconstruction can't keep up and the
                // live region balloons, skip ahead (corrupts one field's
                // sync, but bounds memory) instead of growing unbounded.
                if demod_buffer.len() - demod_start > 5_000_000 {
                    demod_start = demod_buffer.len().saturating_sub(1_000_000);
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
    let mut ch_input_flash_until: Option<std::time::Instant> = None;
    let mut ch_input_flash_msg = String::new();
    while !windows.is_empty() {
        let reason = exit_reason.load(std::sync::atomic::Ordering::Relaxed);
        if reason != 0 {
            windows.clear();
            frame_rxs.clear();
            display_buffers.clear();
            break;
        }

        let mut i = 0;
        while i < windows.len() {
            let (window, width, height, is_pal, absolute_freq_mhz) = &mut windows[i];

            if !window.is_open() || window.is_key_down(Key::Escape) || window.is_key_down(Key::Q) {
                // Q or Escape or window close → quit the app
                windows.remove(i);
                frame_rxs.remove(i);
                recycle_txs.remove(i);
                display_buffers.remove(i);
                continue;
            }

            // S → skip this frequency (blacklist it), resume scanning
            if matches!(ch_input, ChannelInputState::Idle) && window.is_key_down(Key::S) {
                exit_reason.store(3, std::sync::atomic::Ordering::Relaxed);
                windows.clear();
                frame_rxs.clear();
                display_buffers.clear();
                break;
            }

            // N → find next channel (resume scanning without blacklisting)
            if matches!(ch_input, ChannelInputState::Idle) && window.is_key_down(Key::N) {
                exit_reason.store(4, std::sync::atomic::Ordering::Relaxed);
                windows.clear();
                frame_rxs.clear();
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
                        if let Some(c) = key_to_char(k) {
                            if c.is_ascii_alphabetic() {
                                ch_input = ChannelInputState::WaitingSecond(c);
                                break;
                            }
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

            if let Ok(frame_u32) = frame_rxs[i].try_recv() {
                if frame_u32.len() == display_buffers[i].len() {
                    display_buffers[i].copy_from_slice(&frame_u32);
                    // Hand the buffer back to the worker's recycle pool
                    // (drop it if the pool is full — the worker will just
                    // allocate). Done before drawing the overlays, which
                    // operate on `display_buffers[i]`.
                    let _ = recycle_txs[i].try_send(frame_u32);

                    let format_str = if *is_pal { "PAL" } else { "NTSC" };
                    let channel_name = get_fpv_channel_name(*absolute_freq_mhz);
                    let display_text = if let Some(ch) = channel_name {
                        format!("{} · Channel {} [BW]", format_str, ch)
                    } else {
                        format!("{} · {:.2} MHz [BW]", format_str, absolute_freq_mhz)
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

                    // Keybinding hints at the bottom
                    let hint_y = (*height).saturating_sub(20);
                    draw_text_with_bg(
                        &mut display_buffers[i],
                        *width,
                        *height,
                        10,
                        hint_y,
                        "Q:Quit  S:Skip  N:Next  C:Channel",
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

                    let _ = window.update_with_buffer(&display_buffers[i], *width, *height);
                } else if debug {
                    eprintln!(
                        "[DEBUG] dropped frame on window {}: size {} != display {}",
                        i,
                        frame_u32.len(),
                        display_buffers[i].len()
                    );
                }
            } else {
                window.update();
            }

            i += 1;
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

        // --- 5.8 GHz Band L (Lowband / D-band) ---
        5362 => Some("L1"),
        5399 => Some("L2"),
        5436 => Some("L3"),
        5510 => Some("L5"),
        5547 => Some("L6"),
        5584 => Some("L7"),
        5621 => Some("L8"),

        // --- 5.8 GHz Band U (Ultrabando) ---
        5300 => Some("U1"),
        5325 => Some("U2"),
        5348 => Some("U3"),
        5373 => Some("U4"),
        5398 => Some("U5"),
        5423 => Some("U6"),
        5448 => Some("U7"),

        // --- Overlapping/Duplicate 5.8 GHz channels ---
        5473 => Some("U8/L4"),

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
/// `fpv_drone_analog_rs::bands::get_all_channels()`. The overlapping
/// bands (A/B/E/F/R and L) now use identical anchor frequencies in
/// both places — `bands.rs::LOWBAND_FREQS` was reconciled onto this
/// table's standard 48-channel "L" spec so `--channel L1` and the
/// display label agree. The "U" (Ultra-low) and 2.4 GHz video
/// channels live only here because `bands.rs` doesn't model them yet
/// (so `--channel U4` isn't resolvable — that's a separate gap, not
/// a divergence). A follow-up unification pass should move this whole
/// list into `bands.rs` and have both paths read it.
const FPV_CHANNELS_MHZ: &[f64] = &[
    // 5.8 GHz bands
    5865.0, 5845.0, 5825.0, 5805.0, 5785.0, 5765.0, 5745.0, 5725.0, // A
    5733.0, 5752.0, 5771.0, 5790.0, 5809.0, 5828.0, 5847.0, 5866.0, // B
    5705.0, 5685.0, 5665.0, 5645.0, 5885.0, 5905.0, 5925.0, 5945.0, // E
    5740.0, 5760.0, 5780.0, 5800.0, 5820.0, 5840.0, 5860.0, // F (Fatshark)
    5658.0, 5695.0, 5732.0, 5769.0, 5806.0, 5843.0, 5880.0, 5917.0, // R (Raceband)
    5362.0, 5399.0, 5436.0, 5473.0, 5510.0, 5547.0, 5584.0, 5621.0, // L (Lowband)
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
