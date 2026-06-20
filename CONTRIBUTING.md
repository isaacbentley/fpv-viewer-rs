# Contributing to fpv-viewer-rs

First off, thank you for considering contributing to `fpv-viewer-rs`! It's people like you that make the open-source SDR community thrive. This document explains how the test suite is organized, which tools you'll need, and how to verify your changes locally before opening a pull request.

## Quick Start

```bash
git clone https://github.com/isaacbentley/fpv-viewer-rs.git
cd fpv-viewer-rs

# Run the standard validation suite
cargo test                                        # unit + integration + properties
cargo test --test spec_coverage -- --nocapture    # spec inventory report
cargo clippy --all-features --all-targets -- -D warnings
cargo fmt --all --check
```

Everything below this line is optional but recommended if your pull request touches the parser, decompressor, or FFI surface.

## Test Pyramid

The crate is modeled on a layered test pyramid. Each tier catches a different class of bug; together they form the "definition of done" for our releases.

### 1. Unit Tests

Located alongside the code they test in `src/**/*.rs`. These cover format conversion, frequency parsing, builder defaults, struct sizes, and other basic utilities.

Run: `cargo test --lib`

### 2. Integration Tests (`tests/integration_test.rs`)

End-to-end tests against bundled `.rtsa` captures stored in Git LFS. The fixtures gracefully skip when LFS content isn't available. To run them fully, ensure you pull the LFS files:

```bash
git lfs pull
cargo test --test integration_test
```

### 3. Property Tests (`tests/properties.rs`)

We use `proptest` to verify invariants of the parser, decompressor, and validator surfaces.

Run: `cargo test --test properties`

For a deeper verification pass (useful before submitting a PR):
```bash
PROPTEST_CASES=4096 cargo test --test properties --release
```

### 4. Invariant-Coverage Inventory (`tests/spec_coverage.rs`)

A single page enumerating which documented invariants the test suite enforces. **If you are adding a new invariant-bound test, please add a row to `ENFORCED` in `tests/spec_coverage.rs`.**

Run: `cargo test --test spec_coverage -- --nocapture`

### 5. Miri (Nightly)

Catches undefined behavior in safe Rust code. Because Miri refuses to interpret FFI, we only run it on the pure-Rust modules.

```bash
rustup +nightly component add miri
cargo +nightly miri test --lib decompression
cargo +nightly miri test --lib http_streaming
cargo +nightly miri test --lib utils
```

### 6. Code Coverage

Code coverage is automatically reported to Codecov on every PR. Coverage is informational only and will not fail your build. To check coverage locally:

```bash
cargo install cargo-llvm-cov
cargo llvm-cov --html
open target/llvm-cov/html/index.html
```

### 7. Criterion Benchmarks (`benches/`)

Three benchmark harnesses track the hot paths across releases. Benchmarks are run on demand to investigate performance implications of a PR.

```bash
cargo bench                                       # Run all benchmarks
cargo bench --bench parse_int16_packet            # Run a specific harness
```

### 8. Mutation Testing

We occasionally run `cargo mutants` to rewrite mutable expressions and ensure our test suite catches the changes.

```bash
cargo install cargo-mutants
cargo mutants --no-shuffle --in-place=false --timeout=180
```

### 9. ASAN / UBSAN

The C FFI boundary is exercised under AddressSanitizer and UndefinedBehaviorSanitizer via `tests/asan/c_smoke.c`.

```bash
rustup toolchain install nightly
rustup component add rust-src --toolchain nightly
bash tests/asan/run_asan.sh
```
*(Note: ASAN scripts currently require a Linux environment.)*

## Adding a New Spec Invariant

When you encounter a documented invariant in `docs/*.md` that the test suite doesn't currently enforce:
1. Name the test `prop_<short_description>` or `spec_<area>_<behavior>`.
2. Add a `///` doc comment describing the invariant the test pins.
3. Add a row to the `ENFORCED` table in `tests/spec_coverage.rs`.
4. Run `cargo test --test spec_coverage -- --nocapture` and confirm your row appears.

## Code Style

We use standard `rustfmt` defaults. Please run `cargo fmt --all` before pushing.

Clippy is run with `-D warnings` in CI. If a lint is genuinely wrong for the situation, allow it with a `// SAFETY:` or `// ALLOW:` justification comment explaining why.

## Pull Requests

- **Commit messages:** Conventional-commits style is preferred but not required. Describe *why* the change is needed and *what* it changes.
- **Templates:** Please fill out the Pull Request template when opening a PR. Checkboxes are provided for CI validations.

## License

By contributing, you agree your contributions will be licensed under GPL-3.0-or-later, the same as the rest of the project.
