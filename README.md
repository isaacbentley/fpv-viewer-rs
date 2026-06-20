# fpv-viewer-rs

A real-time, cross-SDR desktop viewer for analog FPV drone video signals.

## Features
- **Cross-Platform SDR Support**: Natively interfaces with Ettus USRP, HackRF One, and Aaronia Spectran V6 devices via `sdr-source-rs`.
- **Offline Playback**: Supports replaying `.sigmf` datasets and compressed `.rtsa` offline files.
- **Wideband Sweeping**: Automatically scans entire frequency bands (e.g., 5.8 GHz, 1.2 GHz) to find and lock onto active analog FPV signals.
- **Live Video Rendering**: Real-time monochrome frame display using `minifb`.
- **Temporal Noise Reduction**: Leverages the multi-field ring buffer from `fpv-drone-analog-rs` for robust motion-weighted denoising of noisy analog signals.

## Supported Platforms

- **Linux**: Full support for all SDRs and offline files.
- **Windows**: Full support for all SDRs and offline files.
- **macOS**: Full support for HackRF, USRP, and offline files. (Native Aaronia hardware drivers are currently unsupported on macOS).

## Installation

Ensure you have the required SDR drivers installed on your system (e.g., UHD for USRP, `hackrf` for HackRF One). If using Aaronia Spectran V6 devices, ensure the RTSA-Suite PRO or AARTSAAPI is installed.

```bash
git clone https://github.com/isaacbentley/fpv-viewer-rs.git
cd fpv-viewer-rs
cargo build --release
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
Stream directly using the native Aaronia SDK:

```bash
cargo run --release -- aaronia sdk --channel E4
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
