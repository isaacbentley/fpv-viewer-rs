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

- **Multiple SDR backends.** Ettus USRP, HackRF One, and the Aaronia
  Spectran V6 — over the RTSA HTTP interface or the native SDK.
- **Offline playback.** SigMF datasets (`.sigmf-meta` and `.sigmf-data`)
  and raw interleaved cf32 I/Q files.
- **Automatic band sweep on every backend.** Scans the 5.8 GHz FPV band
  (5,645–5,945 MHz), locks onto active signals, and displays them.
  `--scan-bands all` extends this to every band in the channel table,
  1.24–5.945 GHz. Tune centres are planned against the real channel list
  rather than stepped uniformly, so the sweep spends tunes only where
  channels are: 6 tunes for the 40 channels of the 5.8 GHz band at
  61.44 MSPS, or 45 for all 128 channels. The sweep's PAL/NTSC verdict
  carries into the decoder, so the standard measured on air is the one
  the picture is decoded with.
- **Band panel.** While sweeping, a window shows what the detector is
  looking at: an energy bar per 5 MHz probe across the band, the planned
  tune windows and how long ago each was visited, every channel in the
  table at its true frequency, and each hit labelled with its channel,
  standard, and localized carrier. Once locked, the same data sits as a
  strip under the picture; `B` cycles strip, full panel, and off.
- **Decoding costs the same whether the capture is wide or narrow.** A
  wide capture is what the sweep needs, but the video inside it is only
  about 14 MHz wide, so the down-converter decimates as far as that
  channel allows before anything else touches the samples. A 61.44 MSPS
  capture decodes at 15.36 MSPS; a capture already that narrow is left
  alone. Measured on one core, that is the difference between 5.3 and
  0.8 CPU-seconds of work per second of signal
  (`cargo run --release --example profile_decode`), and live it is the
  difference between dropping 89% of what arrives and dropping 0.8%.
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
  sync approximately one noise step deeper than the discriminator. Its
  loop cannot track a typical FPV deviation much below 20 MSPS, so
  asking for it holds the decode rate up rather than decimating, and the
  viewer says so; that costs the CPU the decimation would have saved.
  The discriminator is the default everywhere the decode rate lands
  below the crossover, which on a wide capture is everywhere.
- **Optional neural denoiser.** Built with `--features neural-vsr`,
  `--denoise` starts with it enabled and **`D`** toggles it during
  playback. It improves a degraded signal and costs a small amount of
  detail on a clean one, so live toggling is useful.

## Platform support

- **Linux** — all SDRs and offline playback. Prebuilt release binaries
  target aarch64 (Raspberry Pi 5 class).
- **macOS** — HackRF, USRP, offline playback, and Aaronia over the RTSA
  HTTP interface. Prebuilt release DMGs target Apple Silicon. Only the
  Aaronia native-SDK path (`aaronia sdk`) is unavailable on macOS.
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

The Aaronia backend is included by default. The AARTSAAPI SDK is
resolved at runtime rather than link time, so building requires nothing
extra — and the RTSA HTTP backend needs no SDK at all; only
`aaronia sdk` does, on the machine running it. `--no-default-features`
restores the lean build without the backend.

One optional feature:

```bash
cargo build --release --features neural-vsr   # neural denoiser
```

`neural-vsr` is disabled by default because it introduces a dependency
on ONNX Runtime.

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
cargo run --release -- aaronia sdk --channel E4
```

Sweep the band from an RTSA HTTP server instead (no SDK needed):

```bash
cargo run --release -- aaronia http http://atc.local:54664
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

The `aaronia` subcommand is present in default builds; it is absent
only from `--no-default-features` builds.

Frequently used options, with the full set available from `--help` on any
subcommand:

| Option | Description |
| :--- | :--- |
| `--scan-bands 5.8\|all` | Bands the auto-scan sweeps. `5.8` (default) covers 5,645–5,945 MHz; `all` adds the 5.3 GHz L/D bands and 1.2 GHz, at proportionally more tunes per sweep. |
| `--sample-rate <hz>` | Override the capture rate. Aaronia runs 61.44 MHz divided by powers of two (61.44, 30.72, 15.36, 7.68 MSPS and lower); a request maps to the nearest. The HTTP default of 61.44 MSPS is about 246 MB/s in `f16`, more than gigabit Ethernet carries, and the RTSA server discards data once its outbound buffer passes 8 MB. Measured over Wi-Fi, even 15.36 MSPS (61 MB/s) fell a few percent short and dropped most packets; a wired link is the fix, and `--sample-rate 15360000` is the widest span worth trying on a marginal one. This is a network limit only: decoding decimates to a fixed working rate, so a wide capture costs no more CPU than a narrow one. |
| `--stream-format f16\|f32\|int16` | Aaronia HTTP only: IQ wire format. `f16` (default) halves network bandwidth against `f32` with no visible cost on analog video. |
| `--demod auto\|disc\|pll` | FM demodulator selection. `auto` uses the PLL at 25 MSPS and above, the discriminator below — measured against the *decode* rate, which is below the capture rate on a wide capture, so `auto` normally picks the discriminator. `pll` holds the decode rate at or above 25 MSPS instead. |
| `--deemphasis-tau <s>` | Video deemphasis time constant, in seconds. Default 0.15 µs (`0.00000015`); `0` disables. It is a single pole, so its attenuation grows without limit: 0.15 µs costs 2.7 dB at 1 MHz and 11.1 dB at 4.2 MHz, where 0.75 µs costs 13.6 and 24.8 dB. NTSC luma runs to about 4.2 MHz, so a long time constant softens the picture badly — measured against a live transmitter, detail fell from 38.6 with it off to 1.6 at 0.75 µs. Lengthen it if your transmitter's pre-emphasis is strong and the picture looks noisy. |
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
| **`B`** | Cycle the band panel: strip, full, off. Closing the scan window, or `Q` in it, quits. |
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
appears sharpest. The default of 0.15 µs also decodes them correctly,
slightly softer, as there is no pre-emphasis to invert.

## License

GNU General Public License v3.0 or later (GPL-3.0-or-later).
