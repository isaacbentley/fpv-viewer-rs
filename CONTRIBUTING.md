# Contributing to fpv-viewer-rs

Local setup and the checks a change has to pass before a pull request.

## Quick start

```bash
git clone https://github.com/isaacbentley/fpv-viewer-rs.git
cd fpv-viewer-rs

cargo build --workspace
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all --check
```

CI runs `cargo test`, `cargo clippy --all-targets -- -D warnings` and
`cargo fmt --all --check` (all `--locked`) on the `stable` toolchain,
for `ubuntu-latest` and `macos-latest`, with SoapySDR installed from
the system package manager. There is no Windows job: the SoapySDR
backend links `libSoapySDR`, and the runners have no unattended
SoapySDR install.

The crate declares `rust-version = "1.89"`, set by `wide` and
`safe_arch` in the dependency tree.

### The backend features

Both SDR backends are features, and both are **on by default**
(`default = ["soapy", "aaronia"]`), so CI builds and tests them. They
share one capture interface in `src/sdr/`: a backend starts a capture
thread and hands the viewer `IqPacket`s over a channel.

- `soapy` links `libSoapySDR` (via `pkg-config`), so building needs it
  installed — `libsoapysdr-dev` or `brew install soapysdr`. Talking to a
  radio also needs that radio's SoapySDR module, at run time only.
- `aaronia` drives `sdr-aaronia-rs`'s unified source directly, with the
  crate's own default features (and its `sdr-source` facade) off. The
  AARTSAAPI SDK is resolved at runtime, not link time, so building
  requires no SDK. Only `aaronia sdk` needs one, at run time, on the
  machine running it; the `aaronia http` backend needs no SDK at all.

`--no-default-features` builds with neither (file replay only); add one
back with `--features soapy` or `--features aaronia`. Clippy should pass
for each of those combinations, not only the default.

To co-develop against a local checkout of `sdr-aaronia-rs`, add the
`[patch]` block shown in the comment above its entry in `Cargo.toml`.

## Adding features or fixing bugs

- Add a unit test when the new code carries non-trivial logic.
- `cargo clippy --workspace --all-targets -- -D warnings` must pass.
- Update `README.md` in the same commit as any change to CLI flags,
  file formats, or supported platforms.

## Code style

`rustfmt` defaults. Run `cargo fmt --all` before pushing.

Clippy runs with `-D warnings` in CI. Suppress a lint with an `// ALLOW:`
comment giving the reason.

Comments state what the code does and what was measured, with the
measurement's conditions. They do not narrate what the code used to do
or why an earlier version was wrong — `git log` holds that.

## Pull requests

- Commit messages: what changed, and why.
- Fill out the pull request template.

## License

By contributing, you agree your contributions will be licensed under
GPL-3.0-or-later, the same as the rest of the project.
