# fpv-viewer-rs

[![CI](https://github.com/isaacbentley/fpv-viewer-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/isaacbentley/fpv-viewer-rs/actions/workflows/ci.yml)
[![License: GPL-3.0-or-later](https://img.shields.io/github/license/isaacbentley/fpv-viewer-rs.svg)](https://choosealicense.com/licenses/gpl-3.0/)

A real-time, cross-SDR desktop viewer for analog FPV drone video signals.

## Features
- **Cross-Platform SDR Support**: Natively interfaces with Ettus USRP, HackRF One, and (behind the `aaronia` feature) Aaronia Spectran V6 devices via `orecchiette-sdr-source-rs`.
- **Offline Playback**: Supports replaying SigMF (`.sigmf-meta`/`.sigmf-data`) datasets and raw interleaved-cf32 IQ files.
- **Wideband Sweeping**: Automatically scans the 5.8 GHz FPV band (5.645–5.945 GHz) to find and lock onto active analog FPV signals.
- **Live Video Rendering**: Real-time monochrome frame display using `minifb`.
- **Weak-Signal Reconstruction**: The decode path uses `orecchiette-fpv-drone-analog-rs` 0.3's phase-1 recovery pipeline — matched-filter sync acquisition, robust OLS line-locked-clock TBC (straight verticals), and spatial+temporal SmartDOC dropout concealment — all active by default.
- **Temporal Noise Reduction**: Leverages the multi-field ring buffer from `orecchiette-fpv-drone-analog-rs` for robust motion-weighted denoising of noisy analog signals.
- **Weak-Signal Detection**: scan and standard-detection paths accumulate spectra across batches (`SpectralIntegrator`) for several dB of extra sensitivity, and video deemphasis is applied by default (`--deemphasis-tau`, 0 to disable) so VTX pre-emphasis doesn't leave HF noise emphasized in the picture. At ≥ 25 MSPS decode rates, `--demod pll` swaps in a PLL demodulator measured to keep sync a full noise-step deeper than the discriminator (keep the default `disc` at lower rates).

## Supported Platforms

- **Linux**: Full support for all SDRs and offline files. Prebuilt release binaries target aarch64 (Raspberry Pi 5-class).
- **Windows**: Builds from source with UHD installed (no prebuilt binaries — CI has no unattended UHD install for Windows yet).
- **macOS**: Full support for HackRF, USRP, and offline files; prebuilt release DMGs target Apple Silicon. (Native Aaronia hardware drivers are currently unsupported on macOS).

## Installation

Ensure you have the required SDR drivers installed on your system (e.g., UHD for USRP, `hackrf` for HackRF One).

```bash
git clone https://github.com/isaacbentley/fpv-viewer-rs.git
cd fpv-viewer-rs
cargo build --release
```

The Aaronia Spectran V6 backend is behind the off-by-default `aaronia`
cargo feature because it links against the native AARTSAAPI SDK. If you
have the RTSA-Suite PRO / AARTSAAPI SDK installed, build with:

```bash
cargo build --release --features aaronia
```

## Command Line Help

```text
Real-time Analog FPV Viewer

Usage: fpv-viewer <COMMAND>

Commands:
  file     Replay a SigMF or raw IQ file
  usrp     Live capture from an Ettus USRP B2xx
  hackrf   Live capture from a HackRF One
  aaronia  Live capture from an Aaronia Spectran V6
  help     Print this message or the help of the given subcommand(s)

Options:
  -h, --help     Print help
  -V, --version  Print version
```

The `aaronia` subcommand only exists in binaries built with
`--features aaronia`; a default build's `--help` will not list it.

## Usage

### Auto-Scan Mode
Sweep the 5.8 GHz band using a USRP and auto-tune to the strongest FPV signal:

```bash
cargo run --release -- usrp
```

### Direct Channel Tuning
Tune a HackRF directly to FPV channel R8 (5917 MHz):

```bash
cargo run --release -- hackrf --channel R8
```

### Aaronia Spectran V6 Streaming
Stream directly using the native Aaronia SDK (needs the `aaronia`
feature and the AARTSAAPI SDK installed):

```bash
cargo run --release --features aaronia -- aaronia sdk --channel E4
```

### Offline Playback
Replay an I/Q file (will automatically read the `.sigmf-meta` if present):

```bash
cargo run --release -- file /path/to/capture.sigmf-data
```

## Shortcuts

While the viewer is running:
- **`N`**: Next channel (when in auto-scan mode, abandons current lock and resumes sweeping).
- **`S`**: Skip/Blacklist current frequency.
- **`C` + `[Band][Chan]`**: Direct tune (e.g., press `C`, then `R`, then `8` to tune to R8).

## License

This project is licensed under the GNU General Public License v3.0 or later (GPL-3.0-or-later).
