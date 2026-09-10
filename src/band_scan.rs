//! The live band-scan panel: what the sweep is seeing, drawn with the
//! viewer's own 8×8 bitmap font.
//!
//! Every element is data the scanner already produces. The bars are the
//! sweep's per-probe energies — the numbers the energy gate and cluster
//! ranking run on, not a separate estimate. Detections are the detector's
//! `DetectionResult`s as reported: localized carrier, standard, confidence.
//! Tune windows are the planner's. Nothing here measures anything new.
//!
//! Two renderings share one model. While sweeping, the panel fills its
//! own window; once locked it overlays the bottom of the picture as a
//! strip, or the full panel if the operator asks (`B` cycles strip /
//! full / off).
//!
//! Rendering is rectangles and bitmap text only — `draw_rect` and
//! `draw_string`, nothing the frame renderer cannot already do. The font
//! maps both letter cases onto one set of glyphs and has no `?` or `·`, so
//! labels stay to letters, digits, `.`, `-` and `/`. The channel grid
//! carries identity by row and position because forty channel names do
//! not fit in 720 px as text.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU8, Ordering};
use std::time::{Duration, Instant};

use minifb::{Key, Window, WindowOptions};
use orecchiette_fpv_drone_analog_rs::bands::FpvBand;
use orecchiette_fpv_drone_analog_rs::detector::ProbeEnergy;
use orecchiette_fpv_drone_analog_rs::types::{DetectionResult, SignalType};

/// How the panel is shown over a locked picture. Process-wide so the
/// operator's choice survives a relock.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PanelMode {
    Strip,
    Full,
    Off,
}

static PANEL_MODE: AtomicU8 = AtomicU8::new(0);

impl PanelMode {
    /// `B` cycles strip → full → off → strip.
    pub fn next(self) -> Self {
        match self {
            PanelMode::Strip => PanelMode::Full,
            PanelMode::Full => PanelMode::Off,
            PanelMode::Off => PanelMode::Strip,
        }
    }

    pub fn load() -> Self {
        match PANEL_MODE.load(Ordering::Relaxed) {
            1 => PanelMode::Full,
            2 => PanelMode::Off,
            _ => PanelMode::Strip,
        }
    }

    pub fn store(self) {
        let v = match self {
            PanelMode::Strip => 0,
            PanelMode::Full => 1,
            PanelMode::Off => 2,
        };
        PANEL_MODE.store(v, Ordering::Relaxed);
    }

    pub fn label(self) -> &'static str {
        match self {
            PanelMode::Strip => "strip",
            PanelMode::Full => "full",
            PanelMode::Off => "off",
        }
    }
}

/// One channel from the table: its name as people say it, its band
/// letter, and where it is.
#[derive(Debug, Clone, PartialEq)]
pub struct Channel {
    pub name: String,
    pub band: char,
    pub hz: f64,
}

/// The letter a band goes by on a VTX menu and in `--channel`.
pub fn band_letter(band: FpvBand) -> char {
    match band {
        FpvBand::BandA => 'A',
        FpvBand::BandB => 'B',
        FpvBand::BandE => 'E',
        FpvBand::Fatshark => 'F',
        FpvBand::Raceband => 'R',
        FpvBand::Lowband => 'L',
        FpvBand::BandD => 'D',
        FpvBand::Band1200 => '1',
        FpvBand::Band3300 => '3',
        FpvBand::Band1200Wide => 'W',
        // `FpvBand` is non-exhaustive; a band this build does not know
        // still gets a row.
        _ => '-',
    }
}

/// Height of the strip overlay in pixels.
pub const STRIP_H: usize = 34;
/// Energies within this distance of a channel count as "under" it, and
/// two hits this close are the same carrier.
const CHANNEL_REACH_HZ: f64 = 3.0e6;
/// Probe spacing the sweep uses; bars are drawn this wide.
const PROBE_STEP_HZ: f64 = 5.0e6;
/// Channels further apart than this start a new axis segment, so the
/// empty gigahertz between 1.3 and 3.3 GHz do not squash the bands.
const SEGMENT_GAP_HZ: f64 = 150.0e6;
/// Confidence at or above which a hit is drawn as confirmed video.
const CONFIRM_AT: f32 = 0.8;
/// The scan window redraws no faster than this.
const REDRAW_EVERY: Duration = Duration::from_millis(40);

const BG: u32 = 0x000b1014;
const LINE: u32 = 0x001e2a31;
const LINE_2: u32 = 0x002a3841;
const MUTED: u32 = 0x007c8c94;
const DIM: u32 = 0x0055646c;
const ACCENT: u32 = 0x00f2a93b;
const VIDEO: u32 = 0x004cc9a6;
const MAYBE: u32 = 0x008fb5ad;
const SKIP: u32 = 0x00b4534a;
const BAR: u32 = 0x002c3a43;
const BAR_HOT: u32 = 0x003f5661;
const WIN_ACTIVE: u32 = 0x00241d10;
const WIN: u32 = 0x00151a1c;

#[derive(Debug, Clone, Copy)]
struct Seg {
    lo_hz: f64,
    hi_hz: f64,
    x0: usize,
    x1: usize,
}

/// A frequency axis in pieces: one contiguous span per group of bands,
/// each given pixels in proportion to its width, with a small gap between.
#[derive(Debug, Clone)]
struct Axis {
    segs: Vec<Seg>,
}

impl Axis {
    fn new(channels_hz: &[f64], x0: usize, x1: usize) -> Self {
        let mut f: Vec<f64> = channels_hz.to_vec();
        f.sort_by(|a, b| a.total_cmp(b));
        let mut spans: Vec<(f64, f64)> = Vec::new();
        for hz in f {
            match spans.last_mut() {
                Some((_, hi)) if hz - *hi <= SEGMENT_GAP_HZ => *hi = hz,
                _ => spans.push((hz, hz)),
            }
        }
        if spans.is_empty() {
            spans.push((5.645e9, 5.945e9));
        }
        for s in &mut spans {
            s.0 -= 10.0e6;
            s.1 += 10.0e6;
        }
        let gap = 6usize;
        let total_px = (x1 - x0).saturating_sub(gap * (spans.len() - 1)) as f64;
        let total_hz: f64 = spans.iter().map(|(lo, hi)| hi - lo).sum();
        // Every segment gets at least a fifth of the axis, so a narrow
        // band beside a wide one stays readable.
        let min_share = 0.2 / spans.len() as f64;
        let mut shares: Vec<f64> = spans.iter().map(|(lo, hi)| (hi - lo) / total_hz).collect();
        let mut extra = 0.0;
        for s in &mut shares {
            if *s < min_share {
                extra += min_share - *s;
                *s = min_share;
            }
        }
        let big: f64 = shares.iter().filter(|s| **s > min_share).sum();
        if extra > 0.0 && big > 0.0 {
            for s in &mut shares {
                if *s > min_share {
                    *s -= extra * (*s / big);
                }
            }
        }
        let mut segs = Vec::new();
        let mut x = x0 as f64;
        for ((lo, hi), share) in spans.iter().zip(shares) {
            let px = (share * total_px).max(24.0);
            segs.push(Seg {
                lo_hz: *lo,
                hi_hz: *hi,
                x0: x.round() as usize,
                x1: (x + px).round() as usize,
            });
            x += px + gap as f64;
        }
        Axis { segs }
    }

    fn seg_for(&self, hz: f64) -> &Seg {
        self.segs
            .iter()
            .find(|s| hz >= s.lo_hz && hz <= s.hi_hz)
            .unwrap_or_else(|| {
                self.segs
                    .iter()
                    .min_by(|a, b| dist(a, hz).total_cmp(&dist(b, hz)))
                    .unwrap()
            })
    }

    fn x(&self, hz: f64) -> usize {
        let s = self.seg_for(hz);
        let t = ((hz - s.lo_hz) / (s.hi_hz - s.lo_hz)).clamp(0.0, 1.0);
        (s.x0 as f64 + t * (s.x1 - s.x0) as f64).round() as usize
    }

    fn px_per_hz(&self, hz: f64) -> f64 {
        let s = self.seg_for(hz);
        (s.x1 - s.x0) as f64 / (s.hi_hz - s.lo_hz)
    }

    fn bar_px(&self, hz: f64) -> usize {
        ((PROBE_STEP_HZ * self.px_per_hz(hz)).round().max(2.0) as usize)
            .saturating_sub(1)
            .max(1)
    }

    /// Tick positions with their MHz labels, spaced so labels never touch.
    fn ticks(&self) -> Vec<(usize, i64)> {
        let mut out = Vec::new();
        for s in &self.segs {
            let ppm = self.px_per_hz(s.lo_hz) * 1e6;
            let step = [50.0, 100.0, 200.0, 500.0, 1000.0]
                .into_iter()
                .find(|st| st * ppm >= 44.0)
                .unwrap_or(1000.0);
            let mut f = (s.lo_hz / (step * 1e6)).ceil() * step * 1e6;
            while f <= s.hi_hz {
                out.push((self.x(f), (f / 1e6).round() as i64));
                f += step * 1e6;
            }
        }
        out
    }
}

fn dist(s: &Seg, hz: f64) -> f64 {
    if hz < s.lo_hz {
        s.lo_hz - hz
    } else if hz > s.hi_hz {
        hz - s.hi_hz
    } else {
        0.0
    }
}

/// Everything the panel knows. Owned by the backend loop so it outlives
/// individual sweeps and rides into lock mode.
#[derive(Debug, Clone)]
pub struct BandScan {
    channels: Vec<Channel>,
    rows: Vec<char>,
    tunes: Vec<f64>,
    span_hz: f64,
    /// Latest energy per probe, keyed by absolute Hz to the nearest 100 kHz.
    energies: HashMap<i64, f32>,
    visited: Vec<Option<Instant>>,
    active_tune: Option<usize>,
    /// Hits with the tune centre that produced them, so revisiting a tune
    /// replaces its own hits and leaves the others alone.
    detections: Vec<(f64, DetectionResult)>,
    lock_hz: Option<f64>,
    skipped: HashSet<u64>,
    last_sweep_end: Option<Instant>,
}

impl BandScan {
    pub fn new(mut channels: Vec<Channel>) -> Self {
        channels.sort_by(|a, b| a.hz.total_cmp(&b.hz));
        let order = "ABEFRLD13W";
        let mut rows: Vec<char> = Vec::new();
        for c in &channels {
            if !rows.contains(&c.band) {
                rows.push(c.band);
            }
        }
        rows.sort_by_key(|b| order.find(*b).unwrap_or(usize::MAX));
        BandScan {
            channels,
            rows,
            tunes: Vec::new(),
            span_hz: 0.0,
            energies: HashMap::new(),
            visited: Vec::new(),
            active_tune: None,
            detections: Vec::new(),
            lock_hz: None,
            skipped: HashSet::new(),
            last_sweep_end: None,
        }
    }

    /// The tune plan the sweep will follow. A changed plan (the sample
    /// rate stepped down) forgets the old visit times.
    pub fn set_plan(&mut self, tunes: Vec<f64>, span_hz: f64) {
        if tunes != self.tunes {
            self.visited = vec![None; tunes.len()];
            self.tunes = tunes;
            self.active_tune = None;
        }
        self.span_hz = span_hz;
    }

    /// Height the full panel wants: the bars plus one grid row per band.
    pub fn full_height(&self) -> usize {
        190 + self.rows.len() * 9
    }

    /// The sweep ran the detector on one packet at `center_hz`.
    /// `first_of_hop` is true for the first packet since a retune; that is
    /// when this tune's previous bars and hits are replaced.
    pub fn record_hop(
        &mut self,
        center_hz: f64,
        first_of_hop: bool,
        probes: &[ProbeEnergy],
        results: &[DetectionResult],
    ) {
        if let Some((k, _)) = self
            .tunes
            .iter()
            .enumerate()
            .min_by(|a, b| (a.1 - center_hz).abs().total_cmp(&(b.1 - center_hz).abs()))
        {
            self.visited[k] = Some(Instant::now());
            self.active_tune = Some(k);
        }
        if first_of_hop {
            let half = self.span_hz / 2.0 + PROBE_STEP_HZ;
            self.energies
                .retain(|k, _| ((*k as f64) * 1e5 - center_hz).abs() > half);
            self.detections
                .retain(|(hop, _)| (*hop - center_hz).abs() > 1.0);
        }
        for p in probes {
            let key = ((center_hz + p.offset_hz) / 1e5).round() as i64;
            self.energies.insert(key, p.energy);
        }
        for r in results {
            let f = r.frequency_hz as f64;
            match self
                .detections
                .iter_mut()
                .find(|(_, d)| (d.frequency_hz as f64 - f).abs() < CHANNEL_REACH_HZ)
            {
                Some(slot) => {
                    if r.rssi_dbm > slot.1.rssi_dbm || r.confidence > slot.1.confidence {
                        *slot = (center_hz, r.clone());
                    }
                }
                None => self.detections.push((center_hz, r.clone())),
            }
        }
    }

    /// The pass over the band finished, found something or not.
    pub fn end_sweep(&mut self) {
        self.last_sweep_end = Some(Instant::now());
        self.active_tune = None;
    }

    pub fn set_lock(&mut self, hz: Option<f64>) {
        self.lock_hz = hz;
    }

    pub fn set_skipped(&mut self, skipped: &HashSet<u64>) {
        self.skipped = skipped.clone();
    }

    fn energy_near(&self, hz: f64) -> Option<f32> {
        let mut best: Option<(f64, f32)> = None;
        for (k, e) in &self.energies {
            let d = ((*k as f64) * 1e5 - hz).abs();
            if d <= CHANNEL_REACH_HZ && best.is_none_or(|(bd, _)| d < bd) {
                best = Some((d, *e));
            }
        }
        best.map(|(_, e)| e)
    }

    /// 25th percentile of the bars: the same floor the energy gate uses.
    fn floor_db(&self) -> f32 {
        let mut v: Vec<f32> = self.energies.values().map(|e| to_db(*e)).collect();
        if v.is_empty() {
            return -100.0;
        }
        v.sort_by(|a, b| a.total_cmp(b));
        v[v.len() / 4]
    }

    fn channel_at(&self, hz: f64) -> Option<&Channel> {
        self.channels
            .iter()
            .filter(|c| (c.hz - hz).abs() <= CHANNEL_REACH_HZ)
            .min_by(|a, b| (a.hz - hz).abs().total_cmp(&(b.hz - hz).abs()))
    }

    fn name_at(&self, hz: f64) -> String {
        self.channel_at(hz)
            .map(|c| c.name.clone())
            .unwrap_or_default()
    }

    /// Pixels the overlay takes from the bottom of a frame in `mode`.
    pub fn overlay_height(&self, mode: PanelMode, frame_h: usize) -> usize {
        match mode {
            PanelMode::Off => 0,
            PanelMode::Strip => STRIP_H.min(frame_h),
            PanelMode::Full => self.full_height().min(frame_h / 2),
        }
    }

    /// Draw over the bottom of a `w × h` picture.
    pub fn draw_overlay(&self, buf: &mut [u32], w: usize, h: usize, mode: PanelMode) {
        match mode {
            PanelMode::Off => {}
            PanelMode::Strip => self.draw_strip(buf, w, h),
            PanelMode::Full => self.draw_full(buf, w, h, self.overlay_height(mode, h)),
        }
    }

    /// The scan window: the panel alone, filling `w × h`.
    pub fn draw_standalone(&self, buf: &mut [u32], w: usize, h: usize) {
        buf.fill(BG);
        self.draw_full(buf, w, h, h);
    }

    fn draw_full(&self, buf: &mut [u32], w: usize, h: usize, panel_h: usize) {
        let y0 = h.saturating_sub(panel_h);
        shade(buf, w, y0, panel_h);
        super::draw_rect(buf, w, h, 0, y0, w, 1, LINE_2);

        let x0 = 44usize;
        let x1 = w.saturating_sub(8).max(x0 + 40);
        let hz_list: Vec<f64> = self.channels.iter().map(|c| c.hz).collect();
        let axis = Axis::new(&hz_list, x0, x1);

        // header: channel count, tune progress or sweep age, lock
        let head_y = y0 + 4;
        let mut head = format!("{} ch", self.channels.len());
        if let Some(k) = self.active_tune {
            head.push_str(&format!("  tune {}/{}", k + 1, self.tunes.len()));
        } else if let Some(t) = self.last_sweep_end {
            head.push_str(&format!(
                "  sweep {:.1}s ago",
                t.elapsed().as_secs_f32().min(999.0)
            ));
        }
        super::draw_string(buf, w, h, x0, head_y, &head, MUTED);
        if let Some(lock) = self.lock_hz {
            let s = format!("lock {} {:.1}", self.name_at(lock), lock / 1e6);
            super::draw_string(
                buf,
                w,
                h,
                x1.saturating_sub(s.len() * 8),
                head_y,
                &s,
                ACCENT,
            );
        }

        // layout: bars, axis, channel rows, age row
        let rows_h = self.rows.len() * 9;
        let age_h = if self.tunes.is_empty() { 0 } else { 12 };
        let axis_h = 12;
        let bars_top = y0 + 16;
        let bars_bot = h
            .saturating_sub(rows_h + age_h + axis_h + 4)
            .max(bars_top + 20);
        let bars_h = bars_bot - bars_top;
        let fy = |db: f32| -> usize {
            let t = ((db.clamp(-100.0, -40.0) + 100.0) / 60.0) as f64;
            bars_bot - (t * bars_h as f64).round() as usize
        };

        for (k, t) in self.tunes.iter().enumerate() {
            let xa = axis.x(t - self.span_hz / 2.0);
            let xb = axis.x(t + self.span_hz / 2.0);
            let c = if Some(k) == self.active_tune {
                WIN_ACTIVE
            } else {
                WIN
            };
            super::draw_rect(
                buf,
                w,
                h,
                xa,
                bars_top - 2,
                xb.saturating_sub(xa),
                bars_h + 2,
                c,
            );
        }
        for db in [-40.0f32, -60.0, -80.0, -100.0] {
            let y = fy(db);
            super::draw_rect(buf, w, h, x0, y, x1 - x0, 1, LINE);
            let s = format!("{}", db as i32);
            super::draw_string(
                buf,
                w,
                h,
                x0.saturating_sub(s.len() * 8 + 4),
                y.saturating_sub(4),
                &s,
                DIM,
            );
        }
        let floor = self.floor_db();
        for (k, e) in &self.energies {
            let hz = (*k as f64) * 1e5;
            let db = to_db(*e);
            let bw = axis.bar_px(hz);
            let x = axis.x(hz).saturating_sub(bw / 2);
            let y = fy(db);
            let c = if db > floor + 6.0 { BAR_HOT } else { BAR };
            super::draw_rect(buf, w, h, x, y, bw, bars_bot - y, c);
        }
        let yf = fy(floor);
        let mut x = x0;
        while x < x1 {
            super::draw_rect(buf, w, h, x, yf, 3, 1, DIM);
            x += 7;
        }
        for (_, d) in &self.detections {
            let hz = d.frequency_hz as f64;
            let confirmed = d.confidence >= CONFIRM_AT;
            let is_lock = self
                .lock_hz
                .is_some_and(|l| (l - hz).abs() < CHANNEL_REACH_HZ);
            let c = if is_lock {
                ACCENT
            } else if confirmed {
                VIDEO
            } else {
                MAYBE
            };
            let x = axis.x(hz);
            let y = fy(d.rssi_dbm).saturating_sub(4).max(bars_top);
            if confirmed {
                super::draw_rect(buf, w, h, x, y, 2, bars_bot - y, c);
                super::draw_rect(buf, w, h, x.saturating_sub(2), y.saturating_sub(2), 6, 5, c);
            } else {
                let mut yy = y;
                while yy < bars_bot {
                    super::draw_rect(buf, w, h, x, yy, 1, 2, c);
                    yy += 4;
                }
                outline(buf, w, h, x.saturating_sub(2), y.saturating_sub(2), 6, 5, c);
            }
            let label = format!("{} {} {:.1}", self.name_at(hz), std_label(d), hz / 1e6);
            let lx = if x + 8 + label.len() * 8 < x1 {
                x + 6
            } else {
                x.saturating_sub(label.len() * 8 + 6)
            };
            super::draw_string(buf, w, h, lx, y.saturating_sub(9).max(bars_top), &label, c);
        }
        super::draw_rect(buf, w, h, x0, bars_bot, x1 - x0, 1, LINE_2);
        for (x, mhz) in axis.ticks() {
            super::draw_rect(buf, w, h, x, bars_bot, 1, 4, LINE_2);
            let s = format!("{}", mhz);
            super::draw_string(
                buf,
                w,
                h,
                x.saturating_sub(s.len() * 4),
                bars_bot + 6,
                &s,
                MUTED,
            );
        }
        // channel grid: one row per band, each cell at its true frequency
        let grid_top = bars_bot + axis_h + 2;
        for (r, band) in self.rows.iter().enumerate() {
            let yy = grid_top + r * 9;
            super::draw_string(
                buf,
                w,
                h,
                x0.saturating_sub(12),
                yy,
                &band.to_string(),
                MUTED,
            );
            for c in self.channels.iter().filter(|c| c.band == *band) {
                let cell_w = axis.bar_px(c.hz).max(4);
                let hot = self
                    .energy_near(c.hz)
                    .map(|e| ((to_db(e) - floor) / 45.0).clamp(0.0, 1.0))
                    .unwrap_or(0.0);
                let x = axis.x(c.hz).saturating_sub(cell_w / 2);
                let is_skip = self.skipped.contains(&(c.hz.round() as u64));
                let is_lock = self
                    .lock_hz
                    .is_some_and(|l| (l - c.hz).abs() < CHANNEL_REACH_HZ);
                let det = self
                    .detections
                    .iter()
                    .find(|(_, d)| (d.frequency_hz as f64 - c.hz).abs() < CHANNEL_REACH_HZ);
                let fill = if is_skip { SKIP } else { lerp(BAR, VIDEO, hot) };
                super::draw_rect(buf, w, h, x, yy + 1, cell_w, 7, fill);
                if is_lock {
                    outline(buf, w, h, x, yy + 1, cell_w, 7, ACCENT);
                } else if let Some((_, d)) = det {
                    let c = if d.confidence >= CONFIRM_AT {
                        VIDEO
                    } else {
                        MAYBE
                    };
                    outline(buf, w, h, x, yy + 1, cell_w, 7, c);
                }
            }
        }
        // age row: seconds since each tune was last visited. Text when the
        // tunes are far enough apart to read it; otherwise one tick per
        // tune, fading from lit (just visited) to dim (half a minute ago).
        if age_h > 0 {
            let ay = grid_top + rows_h + 2;
            super::draw_string(buf, w, h, x0.saturating_sub(28), ay, "age", DIM);
            let xs: Vec<usize> = self.tunes.iter().map(|t| axis.x(*t)).collect();
            let spacing = xs
                .windows(2)
                .map(|p| p[1].saturating_sub(p[0]))
                .min()
                .unwrap_or(usize::MAX);
            for (k, x) in xs.iter().enumerate() {
                let now = Some(k) == self.active_tune;
                let age = self.visited[k].map(|v| v.elapsed().as_secs_f32());
                if spacing >= 48 {
                    let s = if now {
                        "now".to_string()
                    } else {
                        match age {
                            Some(a) => format!("{:.0}s", a.min(999.0)),
                            None => "-".to_string(),
                        }
                    };
                    let c = if now { ACCENT } else { MUTED };
                    super::draw_string(buf, w, h, x.saturating_sub(s.len() * 4), ay, &s, c);
                } else {
                    let tw = spacing.saturating_sub(2).clamp(1, 12);
                    let c = if now {
                        ACCENT
                    } else {
                        match age {
                            Some(a) => lerp(MUTED, LINE, (a / 30.0).clamp(0.0, 1.0)),
                            None => LINE,
                        }
                    };
                    super::draw_rect(buf, w, h, x.saturating_sub(tw / 2), ay + 1, tw, 6, c);
                }
            }
        }
    }

    fn draw_strip(&self, buf: &mut [u32], w: usize, h: usize) {
        let strip_h = STRIP_H.min(h);
        let y0 = h - strip_h;
        shade(buf, w, y0, strip_h);
        super::draw_rect(buf, w, h, 0, y0, w, 1, LINE_2);

        let left_w = 8 * 16;
        let right_w = 8 * 15;
        let x0 = 8 + left_w;
        let x1 = w.saturating_sub(8 + right_w);
        if x1 <= x0 + 40 {
            return;
        }
        let hz_list: Vec<f64> = self.channels.iter().map(|c| c.hz).collect();
        let axis = Axis::new(&hz_list, x0, x1);
        let top = y0 + 4;
        let bot = h.saturating_sub(4).max(top + 4);
        let hh = bot - top;

        for (k, t) in self.tunes.iter().enumerate() {
            let xa = axis.x(t - self.span_hz / 2.0);
            let xb = axis.x(t + self.span_hz / 2.0);
            let c = if Some(k) == self.active_tune {
                WIN_ACTIVE
            } else {
                WIN
            };
            super::draw_rect(buf, w, h, xa, top, xb.saturating_sub(xa), hh, c);
        }
        let floor = self.floor_db();
        for (k, e) in &self.energies {
            let hz = (*k as f64) * 1e5;
            let db = to_db(*e);
            let t = ((db.clamp(-100.0, -40.0) + 100.0) / 60.0) as f64;
            let bh = ((t * hh as f64).round() as usize).max(1);
            let bw = axis.bar_px(hz);
            let x = axis.x(hz).saturating_sub(bw / 2);
            let c = if db > floor + 6.0 { BAR_HOT } else { BAR };
            super::draw_rect(buf, w, h, x, bot - bh, bw, bh, c);
        }
        for s in &self.skipped {
            let hz = *s as f64;
            let bw = axis.bar_px(hz);
            super::draw_rect(
                buf,
                w,
                h,
                axis.x(hz).saturating_sub(bw / 2),
                top,
                bw,
                hh,
                SKIP,
            );
        }
        for (_, d) in &self.detections {
            let x = axis.x(d.frequency_hz as f64);
            let c = if d.confidence >= CONFIRM_AT {
                VIDEO
            } else {
                MAYBE
            };
            super::draw_rect(buf, w, h, x, top + hh / 3, 1, hh - hh / 3, c);
        }
        if let Some(l) = self.lock_hz {
            super::draw_rect(buf, w, h, axis.x(l).saturating_sub(1), top, 2, hh, ACCENT);
        }

        let ty = y0 + (strip_h.saturating_sub(8)) / 2;
        match self.lock_hz {
            Some(l) => {
                let s = format!("lock {} {:.1}", self.name_at(l), l / 1e6);
                super::draw_string(buf, w, h, 8, ty, &s, ACCENT);
            }
            None => super::draw_string(buf, w, h, 8, ty, "scanning", MUTED),
        }
        let right = if self.active_tune.is_some() {
            "sweeping".to_string()
        } else if let Some(t) = self.last_sweep_end {
            format!("sweep {:.0}s ago", t.elapsed().as_secs_f32().min(999.0))
        } else {
            String::new()
        };
        super::draw_string(
            buf,
            w,
            h,
            w.saturating_sub(8 + right.len() * 8),
            ty,
            &right,
            MUTED,
        );
    }
}

/// The window the sweep draws into. One per backend loop, kept across
/// empty sweeps so the operator sees one window, not a new one per pass;
/// dropped when a lock hands over to the picture window.
pub struct ScanWindow {
    window: Window,
    buf: Vec<u32>,
    w: usize,
    h: usize,
    last_draw: Option<Instant>,
}

impl ScanWindow {
    pub fn open(height: usize) -> anyhow::Result<Self> {
        let w = 720;
        let h = height.max(120);
        let mut window = Window::new("Band scan", w, h, WindowOptions::default())?;
        // Redraws are already throttled here; minifb's own limiter would
        // otherwise sleep on the packet thread to hold a frame rate.
        window.set_target_fps(0);
        Ok(ScanWindow {
            window,
            buf: vec![BG; w * h],
            w,
            h,
            last_draw: None,
        })
    }

    /// Redraw from `band`, rate-limited. Returns `false` once the operator
    /// has closed the window or pressed `Q` / `Esc`, which ends the scan.
    pub fn update(&mut self, band: &BandScan) -> bool {
        if self.last_draw.is_some_and(|t| t.elapsed() < REDRAW_EVERY) {
            return self.window.is_open();
        }
        self.last_draw = Some(Instant::now());
        band.draw_standalone(&mut self.buf, self.w, self.h);
        // Top right: the header's lock slot, empty while sweeping.
        super::draw_string(
            &mut self.buf,
            self.w,
            self.h,
            self.w.saturating_sub(8 + 6 * 8),
            4,
            "q quit",
            DIM,
        );
        let _ = self.window.update_with_buffer(&self.buf, self.w, self.h);
        let pressed = self.window.get_keys_pressed(minifb::KeyRepeat::No);
        self.window.is_open() && !pressed.contains(&Key::Q) && !pressed.contains(&Key::Escape)
    }
}

fn to_db(e: f32) -> f32 {
    10.0 * (e + 1e-12).log10()
}

fn std_label(d: &DetectionResult) -> &'static str {
    match d.signal_type {
        SignalType::AnalogVideoNtsc => "ntsc",
        SignalType::AnalogVideoPal => "pal",
        _ => "vid",
    }
}

/// Darken `height` rows from `y` toward the panel ground, keeping a little
/// of what was there so the picture stays visible through the overlay.
fn shade(buf: &mut [u32], w: usize, y: usize, height: usize) {
    for row in y..y + height {
        let start = row * w;
        let end = (start + w).min(buf.len());
        if start >= end {
            break;
        }
        for px in &mut buf[start..end] {
            *px = lerp(*px, BG, 0.86);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn outline(
    buf: &mut [u32],
    w: usize,
    h: usize,
    x: usize,
    y: usize,
    width: usize,
    height: usize,
    c: u32,
) {
    super::draw_rect(buf, w, h, x, y, width, 1, c);
    super::draw_rect(buf, w, h, x, y + height.saturating_sub(1), width, 1, c);
    super::draw_rect(buf, w, h, x, y, 1, height, c);
    super::draw_rect(buf, w, h, x + width.saturating_sub(1), y, 1, height, c);
}

fn lerp(a: u32, b: u32, t: f32) -> u32 {
    let ch = |shift: u32| -> u32 {
        let va = ((a >> shift) & 0xff) as f32;
        let vb = ((b >> shift) & 0xff) as f32;
        (va + (vb - va) * t).round().clamp(0.0, 255.0) as u32
    };
    (ch(16) << 16) | (ch(8) << 8) | ch(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn band58() -> Vec<Channel> {
        orecchiette_fpv_drone_analog_rs::bands::get_all_channels()
            .into_iter()
            .filter(|c| (5_645e6..=5_945e6).contains(&(c.frequency_hz as f64)))
            .map(|c| Channel {
                name: format!("{}{}", band_letter(c.band), c.channel),
                band: band_letter(c.band),
                hz: c.frequency_hz as f64,
            })
            .collect()
    }

    fn hit(hz: f64, rssi: f32, conf: f32) -> DetectionResult {
        DetectionResult {
            channel: None,
            frequency_hz: hz as u64,
            confidence: conf,
            rssi_dbm: rssi,
            bandwidth_hz: 0,
            signal_type: SignalType::AnalogVideoNtsc,
        }
    }

    fn probes(n: usize) -> Vec<ProbeEnergy> {
        (0..n)
            .map(|i| ProbeEnergy {
                offset_hz: -20e6 + i as f64 * 5e6,
                energy: 1e-8,
                confidence: 0.0,
            })
            .collect()
    }

    #[test]
    fn panel_mode_cycles_through_all_three() {
        let m = PanelMode::Strip;
        assert_eq!(m.next(), PanelMode::Full);
        assert_eq!(m.next().next(), PanelMode::Off);
        assert_eq!(m.next().next().next(), PanelMode::Strip);
    }

    #[test]
    fn rows_follow_the_band_order_people_use() {
        let b = BandScan::new(band58());
        assert_eq!(b.rows, vec!['A', 'B', 'E', 'F', 'R']);
    }

    #[test]
    fn revisiting_a_tune_replaces_only_its_own_bars_and_hits() {
        let mut b = BandScan::new(band58());
        b.set_plan(vec![5.7e9, 5.8e9], 49e6);
        b.record_hop(5.7e9, true, &probes(9), &[hit(5.695e9, -60.0, 0.95)]);
        b.record_hop(5.8e9, true, &probes(9), &[hit(5.806e9, -70.0, 0.8)]);
        assert_eq!(b.energies.len(), 18);
        assert_eq!(b.detections.len(), 2);
        // Second pass over the first tune: it is quiet now.
        b.record_hop(5.7e9, true, &probes(9), &[]);
        assert_eq!(b.energies.len(), 18);
        let left: Vec<u64> = b.detections.iter().map(|(_, d)| d.frequency_hz).collect();
        assert_eq!(left, vec![5_806_000_000]);
        // A second packet of the same hop merges rather than replaces.
        b.record_hop(5.8e9, false, &probes(9), &[hit(5.806e9, -65.0, 0.95)]);
        assert_eq!(b.detections.len(), 1);
        assert_eq!(b.detections[0].1.rssi_dbm, -65.0);
    }

    #[test]
    fn axis_is_monotonic_and_splits_far_apart_bands() {
        let all: Vec<f64> = orecchiette_fpv_drone_analog_rs::bands::get_all_channels()
            .iter()
            .map(|c| c.frequency_hz as f64)
            .collect();
        let axis = Axis::new(&all, 44, 712);
        assert!(axis.segs.len() >= 2, "1.2 GHz and 5.8 GHz share a segment");
        let mut sorted = all.clone();
        sorted.sort_by(|a, b| a.total_cmp(b));
        let xs: Vec<usize> = sorted.iter().map(|f| axis.x(*f)).collect();
        assert!(xs.windows(2).all(|p| p[0] <= p[1]));
        assert!(*xs.first().unwrap() >= 44 && *xs.last().unwrap() <= 712);
        // Every segment is wide enough to read.
        assert!(axis.segs.iter().all(|s| s.x1 - s.x0 >= 24));
        for (x, _) in axis.ticks() {
            assert!((44..=712).contains(&x));
        }
    }

    #[test]
    fn drawing_stays_inside_every_buffer_it_is_given() {
        let mut b = BandScan::new(band58());
        b.set_plan(vec![5.67e9, 5.72e9, 5.77e9, 5.82e9, 5.87e9, 5.92e9], 49e6);
        b.record_hop(5.87e9, true, &probes(9), &[hit(5.865e9, -55.0, 0.95)]);
        b.set_lock(Some(5.865e9));
        let mut skipped = HashSet::new();
        skipped.insert(5_806_000_000u64);
        b.set_skipped(&skipped);
        for (w, h) in [(720usize, 235usize), (720, 576), (320, 240), (64, 40)] {
            let mut buf = vec![0x00404040u32; w * h];
            b.draw_standalone(&mut buf, w, h);
            b.draw_overlay(&mut buf, w, h, PanelMode::Strip);
            b.draw_overlay(&mut buf, w, h, PanelMode::Full);
            b.draw_overlay(&mut buf, w, h, PanelMode::Off);
        }
        let mut buf = vec![0x00404040u32; 720 * 576];
        b.draw_overlay(&mut buf, 720, 576, PanelMode::Strip);
        // The strip touched the bottom rows and left the top alone.
        assert!(buf[..720 * 100].iter().all(|p| *p == 0x00404040));
        assert!(buf[720 * 560..].iter().any(|p| *p != 0x00404040));
        assert_eq!(b.overlay_height(PanelMode::Strip, 576), STRIP_H);
        assert!(b.overlay_height(PanelMode::Full, 576) <= 288);
    }
}
