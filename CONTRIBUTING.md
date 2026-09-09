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
for `ubuntu-latest` and `macos-latest`. There is no Windows job: the
USRP backend links `libuhd` via pkg-config and the runners have no
unattended UHD install.

The crate declares `rust-version = "1.89"`, set by `wide` and
`safe_arch` in the dependency tree.

### The `aaronia` feature

The Aaronia Spectran V6 backend is **on by default**
(`default = ["aaronia"]`), so CI builds and tests it. The AARTSAAPI SDK
is resolved at runtime, not link time, so building requires no SDK. Only
`aaronia sdk` needs one, at run time, on the machine running it; the
`aaronia http` backend needs no SDK at all.

`--no-default-features` builds without the backend.

To co-develop against a local checkout of `sdr-aaronia-rs`, uncomment
the `[patch]` block in `Cargo.toml`.

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
