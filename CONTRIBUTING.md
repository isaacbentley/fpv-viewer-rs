# Contributing to fpv-viewer-rs

First off, thank you for considering contributing to `fpv-viewer-rs`! It's people like you that make the open-source SDR community thrive. This document explains how to set up your environment and verify your changes locally before opening a pull request.

## Quick Start

```bash
git clone https://github.com/isaacbentley/fpv-viewer-rs.git
cd fpv-viewer-rs

# Run the standard validation suite
cargo test
cargo clippy --all-features --all-targets -- -D warnings
cargo fmt --all --check
```

## Adding Features or Fixing Bugs

When adding new features (like support for a new SDR hardware driver or a new rendering mode):
- Please try to add a small unit test if the new functionality contains complex logic.
- Ensure your code compiles and passes `cargo clippy` without warnings.
- Keep the `README.md` updated if you change command-line flags or supported platforms.

## Code Style

We use standard `rustfmt` defaults. Please run `cargo fmt --all` before pushing.

Clippy is run with `-D warnings` in CI. If a lint is genuinely wrong for the situation, allow it with a `// ALLOW:` justification comment explaining why.

## Pull Requests

- **Commit messages:** Describe *why* the change is needed and *what* it changes.
- **Templates:** Please fill out the Pull Request template when opening a PR. Checkboxes are provided for CI validations.

## License

By contributing, you agree your contributions will be licensed under GPL-3.0-or-later, the same as the rest of the project.
