//! `--debug` frame snapshots.
//!
//! Snapshots go into a directory of their own, one per run, and are
//! only ever created, never opened: a name that already exists — a file
//! from somewhere else, or a symlink someone left in a shared directory
//! — is an error, not something to write through.

use image::ImageEncoder;
use std::io::BufWriter;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

/// Where every run's snapshot directory is created, relative to the
/// working directory.
pub const SNAPSHOT_ROOT: &str = "fpv_frames";

/// Snapshots written by this process so far; part of each file name, so
/// two locks on the same channel in one run cannot collide.
static SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// This run's snapshot directory: `fpv_frames/run-<unix seconds>-<pid>`.
/// Chosen once per process; created on the first save.
pub fn run_dir() -> &'static Path {
    static DIR: OnceLock<PathBuf> = OnceLock::new();
    DIR.get_or_init(|| {
        let secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        Path::new(SNAPSHOT_ROOT).join(format!("run-{secs}-{}", std::process::id()))
    })
}

/// A snapshot's file name: a run-wide sequence number, the window's
/// channel index, the channel frequency and the field number.
///
/// The frequency can come from a file's metadata, so it is only printed
/// when it is a plausible one; the name never holds anything but ASCII
/// letters, digits, `_` and `.`.
pub fn file_name(channel_index: usize, display_mhz: f64, frame: u64) -> String {
    let seq = SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let mhz = if display_mhz.is_finite() && (0.0..1.0e7).contains(&display_mhz) {
        format!("{display_mhz:.0}MHz")
    } else {
        "unknownMHz".to_string()
    };
    format!("fpv_frame_{seq:04}_ch{channel_index}_{mhz}_{frame}.png")
}

/// Encode `rgb` (8-bit RGB, `width` x `height`) as a PNG at `dir/name`,
/// creating `dir` if needed. Fails rather than replace or follow
/// anything already at that path.
pub fn save_rgb(
    dir: &Path,
    name: &str,
    rgb: &[u8],
    width: u32,
    height: u32,
) -> anyhow::Result<PathBuf> {
    std::fs::create_dir_all(dir)?;
    let path = dir.join(name);
    let file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)?;
    image::codecs::png::PngEncoder::new(BufWriter::new(file)).write_image(
        rgb,
        width,
        height,
        image::ExtendedColorType::Rgb8,
    )?;
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "fpv-viewer-snapshot-test-{tag}-{}-{}",
            std::process::id(),
            SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn a_snapshot_is_written_once_and_never_replaced() {
        let dir = scratch("once").join("nested");
        let rgb = [255u8, 0, 0, 0, 255, 0, 0, 0, 255, 9, 9, 9];
        let path = save_rgb(&dir, "a.png", &rgb, 2, 2).unwrap();
        let written = std::fs::read(&path).unwrap();
        assert_eq!(&written[..8], b"\x89PNG\r\n\x1a\n");

        // The same name again is refused, and the first file is intact.
        assert!(save_rgb(&dir, "a.png", &[0u8; 12], 2, 2).is_err());
        assert_eq!(std::fs::read(&path).unwrap(), written);
        let _ = std::fs::remove_dir_all(dir.parent().unwrap());
    }

    #[cfg(unix)]
    #[test]
    fn a_planted_symlink_is_not_written_through() {
        let dir = scratch("symlink");
        std::fs::create_dir_all(&dir).unwrap();
        let victim = dir.join("victim.txt");
        std::fs::write(&victim, b"keep me").unwrap();
        std::os::unix::fs::symlink(&victim, dir.join("a.png")).unwrap();
        // Dangling ones too: creating through it would make the target.
        let absent = dir.join("absent.txt");
        std::os::unix::fs::symlink(&absent, dir.join("b.png")).unwrap();

        assert!(save_rgb(&dir, "a.png", &[0u8; 12], 2, 2).is_err());
        assert!(save_rgb(&dir, "b.png", &[0u8; 12], 2, 2).is_err());
        assert_eq!(std::fs::read(&victim).unwrap(), b"keep me");
        assert!(!absent.exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn names_are_unique_and_carry_the_channel_index() {
        let a = file_name(0, 5865.0, 1);
        let b = file_name(1, 5865.0, 1);
        let c = file_name(0, 5865.0, 1);
        assert!(a.contains("_ch0_5865MHz_1.png"), "{a}");
        assert!(b.contains("_ch1_5865MHz_1.png"), "{b}");
        assert_ne!(a, c, "two locks on one channel must not collide");
    }

    #[test]
    fn a_hostile_frequency_cannot_shape_the_name() {
        for mhz in [f64::NAN, f64::INFINITY, -1.0, 1.0e300] {
            let name = file_name(3, mhz, 2);
            assert!(name.contains("_ch3_unknownMHz_2.png"), "{name}");
            assert!(name.len() < 64);
            assert!(
                name.chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '.')
            );
        }
        assert!(run_dir().starts_with(SNAPSHOT_ROOT));
    }
}
