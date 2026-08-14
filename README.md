# fpv-viewer-rs

[![CI](https://github.com/isaacbentley/fpv-viewer-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/isaacbentley/fpv-viewer-rs/actions/workflows/ci.yml)
[![License: GPL-3.0-or-later](https://img.shields.io/github/license/isaacbentley/fpv-viewer-rs.svg)](https://choosealicense.com/licenses/gpl-3.0/)

A real-time desktop viewer for analog FPV drone video, supporting live
capture from USRP, HackRF, and Aaronia hardware as well as offline
playback of recorded files.

Signal processing is provided by
[orecchiette-fpv-drone-analog-rs](https://github.com/isaacbentley/orecchiette-fpv-drone-analog-rs);
this repository is the application around it.

## Features

- **Multiple SDR backends.** Ettus USRP, HackRF One, and — behind the
  `aaronia` feature — the Aaronia Spectran V6.
- **Offline playback.** SigMF datasets (`.sigmf-meta` and `.sigmf-data`)
  and raw interleaved cf32 I/Q files.
- **Automatic band sweep.** Scans the 5.8 GHz FPV band
  (5,645–5,945 MHz), locks onto active signals, and displays them.
  `--scan-bands all` extends this to every band in the channel table,
  1.24–5.945 GHz. Tune centres are planned against the real channel list
  rather than stepped uniformly, so the sweep spends tunes only where
  channels are: 6 tunes for the 40 channels of the 5.8 GHz band at
  61.44 MSPS, or 45 for all 128 channels.
- **Live monochrome rendering** in a desktop window.
- **Weak-signal decoding**, enabled by default:
  - Matched-filter sync acquisition, a line-locked clock for straight
    vertical edges, and dropout concealment combining spatial and
    temporal sources.
  - Multi-field noise reduction weighted by local motion, so static
    regions are cleaned without smearing movement.
  - Spectra accumulated across batches during scanning for additional
    sensitivity.
  - Video deemphasis, inverting the transmitter's pre-emphasis so
    high-frequency noise is not left emphasized in the picture
    (`--deemphasis-tau`, `0` to disable).
- **Optional PLL demodulator.** At 25 MSPS and above, `--demod pll` holds
  sync approximately one noise step deeper than the discriminator. The
  discriminator remains preferable at lower rates.
- **Optional neural denoiser.** Built with `--features neural-vsr`,
  `--denoise` starts with it enabled and **`D`** toggles it during
  playback. It improves a degraded signal and costs a small amount of
  detail on a clean one, so live toggling is useful.

## Platform support

- **Linux** — all SDRs and offline playback. Prebuilt release binaries
  target aarch64 (Raspberry Pi 5 class).
- **macOS** — HackRF, USRP, and offline playback. Prebuilt release DMGs
  target Apple Silicon. Aaronia's native drivers are unavailable on
  macOS.
- **Windows** — builds from source with UHD installed. No prebuilt
  binaries, as CI has no unattended UHD installation for Windows.

## Installation

Install the relevant SDR drivers first, such as UHD for USRP or `hackrf`
for HackRF One.

```bash
git clone https://github.com/isaacbentley/fpv-viewer-rs.git
cd fpv-viewer-rs
cargo build --release
```

Two optional features are available:

```bash
cargo build --release --features aaronia      # Aaronia Spectran V6
cargo build --release --features neural-vsr   # neural denoiser
```

`aaronia` is disabled by default because it links against the native
AARTSAAPI SDK, which requires RTSA-Suite PRO or the AARTSAAPI SDK to be
installed. `neural-vsr` is disabled by default because it introduces a
dependency on ONNX Runtime.

## Usage

Sweep the band with a USRP and tune to the strongest detected signal:

```bash
cargo run --release -- usrp
```

Tune a HackRF directly to channel R8 (5,917 MHz):

```bash
cargo run --release -- hackrf --channel R8
```

Replay a capture, reading the `.sigmf-meta` sidecar automatically when
present:

```bash
cargo run --release -- file /path/to/capture.sigmf-data
```

Stream from an Aaronia Spectran V6:

```bash
cargo run --release --features aaronia -- aaronia sdk --channel E4
```

Replay a file with the neural denoiser enabled from the start:

```bash
cargo run --release --features neural-vsr -- file --denoise /path/to/capture.sigmf-data
```

## Commands and options

```text
Usage: fpv-viewer <COMMAND>

Commands:
  file     Replay a SigMF or raw IQ file
  usrp     Live capture from an Ettus USRP B2xx
  hackrf   Live capture from a HackRF One
  aaronia  Live capture from an Aaronia Spectran V6
```

The `aaronia` subcommand is present only in builds made with
`--features aaronia`.

Frequently used options, with the full set available from `--help` on any
subcommand:

| Option | Description |
| :--- | :--- |
| `--scan-bands 5.8\|all` | Bands the auto-scan sweeps. `5.8` (default) covers 5,645–5,945 MHz; `all` adds the 5.3 GHz L/D bands and 1.2 GHz, at proportionally more tunes per sweep. |
| `--demod auto\|disc\|pll` | FM demodulator selection. `auto` uses the PLL at 25 MSPS and above, the discriminator below. |
| `--deemphasis-tau <s>` | Video deemphasis time constant. Default 0.75 µs; `0` disables. |
| `--denoise` | Start with the neural denoiser enabled (requires `--features neural-vsr`). |
| `--denoise-model <path>` | ONNX model to load. Defaults to `models/temporal_denoiser.onnx`. |
| `--temporal-window <n>` | Fields retained for temporal denoising and dropout repair. Default 5, giving roughly +7 dB on static scenes at about 83 ms latency; `1` disables temporal processing. |
| `--debug` | Print per-frame decode metrics and save frames 1–3 and 30–32 as PNG files. |

## Keyboard shortcuts

| Key | Action |
| :--- | :--- |
| **`N`** | Next channel. Abandons the current lock and resumes sweeping. |
| **`S`** | Skip and blacklist the current frequency. |
| **`D`** | Toggle the neural denoiser. |
| **`C`** then band and channel | Direct tune. For R8, press `C`, then `R`, then `8`. |
| **`Esc`** or **`Q`** | Quit. |

## Verifying a decode

To determine whether a decoding problem originates in a capture or in the
decoder, generate a reference file that is correct by construction and
replay that instead:

```bash
cd ../orecchiette-fpv-drone-analog-rs
cargo run --release --example make_reference_capture -- --standard pal
cargo run --release --example make_reference_capture -- --standard ntsc
```

Replayed with `--debug`, these should report the expected standard and
geometry with `SyncQ` near 1.00 on every frame; the reference PAL and
NTSC files decode at 0.99 and 1.00 respectively with no interpolated
rows. If the reference files decode correctly and a real capture does
not, the fault lies in the capture.

The reference files carry no transmitter pre-emphasis, so
`--deemphasis-tau 0` reproduces the generated waveform exactly and
appears sharper. The default of 0.75 µs also decodes them correctly but
with softer edges, as there is no pre-emphasis to invert. Against a real
transmitter the default is correct.

## License

GNU General Public License v3.0 or later (GPL-3.0-or-later).
