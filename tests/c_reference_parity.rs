//! Parity against the pinned upstream Xiph C reference, the oracle this crate ports.
//!
//! Two things are checked, and both skip gracefully when the artifacts are absent
//! (AGENTS.md rule 4), so this test never fails a run that lacks the data:
//!
//! 1. the port against the committed golden for every clip in `test_data/` — the
//!    existing `tests/parity.rs` harness extended to the whole clip set;
//! 2. the committed golden against a *freshly built* scalar C oracle, which is what
//!    makes the golden reproducible from upstream sources instead of an opaque blob:
//!    `scripts/c_reference.sh` builds the `_c` path from the pinned commit and writes
//!    one output per clip; the CI job points `RNNOISE_C_REFERENCE_DIR` at that
//!    directory.
//!
//! The tolerance is the one `doc/parity.md` defines: every difference must be a single
//! rounding — `max_diff <= 2` LSB between the port and the reference, `<= 1` between two
//! builds of the same path — and the relative energy stays below `1e-6`. The test asserts
//! the spread, not equality, because a different compiler or `-O` level can round
//! differently.
//!
//! With the recipe in `scripts/c_reference.sh` (the pure-C path, not the SSE2 `vec_avx.h`
//! path) the measured values are 3 samples port-vs-golden and 4 samples golden-vs-fresh on
//! the speech clip, and 0 on all three deterministic clips. Three orders of magnitude of
//! headroom, so a real divergence cannot hide inside the gate.

use std::path::{Path, PathBuf};

use rnnoise::{DenoiseState, FRAME_SIZE};

const MAX_LSB_PORT: i32 = 2;
const MAX_LSB_ORACLE: i32 = 1;
const MAX_REL: f64 = 1e-6;

/// clip name → (clip path, committed golden path)
fn clips() -> Vec<(&'static str, PathBuf, PathBuf)> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("test_data");
    let mut out = vec![(
        "testing",
        root.join("testing.raw"),
        root.join("ref_out.raw"),
    )];
    for name in ["tone_noise", "silence", "zero"] {
        out.push((
            name,
            root.join("clips").join(format!("{name}.raw")),
            root.join("goldens").join(format!("{name}.raw")),
        ));
    }
    out
}

fn load_i16(path: &Path) -> Option<Vec<i16>> {
    let bytes = std::fs::read(path).ok()?;
    if bytes.is_empty() {
        return None;
    }
    Some(
        bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| i16::from_le_bytes([c[0], c[1]]))
            .collect(),
    )
}

/// Run the port over a clip exactly like the C demo does (first frame is look-ahead
/// and is not written, floats truncate toward zero on the way back to `i16`).
fn port_output(input: &[i16]) -> Vec<i16> {
    let mut st = DenoiseState::new();
    let mut frame = [0.0f32; FRAME_SIZE];
    let mut out_buf = [0.0f32; FRAME_SIZE];
    let mut output = Vec::new();
    let mut first = true;
    for chunk in input.as_chunks::<FRAME_SIZE>().0 {
        for (dst, &src) in frame.iter_mut().zip(chunk) {
            *dst = src as f32;
        }
        st.process_frame(&mut out_buf, &frame);
        if !first {
            output.extend(out_buf.iter().map(|&x| x as i16));
        }
        first = false;
    }
    output
}

/// Per-sample spread between two `i16` outputs.
#[derive(Debug)]
struct Spread {
    samples: usize,
    differing: usize,
    max: i32,
    rel: f64,
}

fn spread(reference: &[i16], other: &[i16]) -> Spread {
    let mut differing = 0;
    let mut max = 0;
    let mut ss_ref = 0.0;
    let mut ss_diff = 0.0;
    for (&r, &o) in reference.iter().zip(other) {
        let d = (r as i32 - o as i32).abs();
        if d != 0 {
            differing += 1;
        }
        max = max.max(d);
        ss_ref += (r as f64).powi(2);
        ss_diff += (d as f64).powi(2);
    }
    Spread {
        samples: reference.len(),
        differing,
        max,
        rel: if ss_ref > 0.0 { ss_diff / ss_ref } else { 0.0 },
    }
}

#[test]
fn port_matches_the_committed_goldens() {
    let oracle = std::env::var("RNNOISE_C_REFERENCE_DIR").ok();
    let mut ran = 0;
    for (name, clip, golden) in clips() {
        let Some(input) = load_i16(&clip) else {
            eprintln!("parity: clip {name} absent, skipped");
            continue;
        };
        let Some(golden_samples) = load_i16(&golden) else {
            eprintln!("parity: golden for {name} absent, skipped");
            continue;
        };
        let got = port_output(&input);
        assert_eq!(got.len(), golden_samples.len(), "{name}: output length");
        let s = spread(&golden_samples, &got);
        eprintln!(
            "port vs golden  {name:<10} samples={samples} differing={differing} max={max} rel={rel:.3e}",
            samples = s.samples,
            differing = s.differing,
            max = s.max,
            rel = s.rel,
        );
        assert!(
            s.max <= MAX_LSB_PORT,
            "{name}: max sample diff {max}",
            max = s.max
        );
        assert!(
            s.rel < MAX_REL,
            "{name}: relative energy diff {rel:.3e}",
            rel = s.rel
        );

        if let Some(dir) = &oracle {
            let fresh = load_i16(&Path::new(dir).join(format!("{name}.raw")));
            let Some(fresh) = fresh else {
                eprintln!("  oracle output for {name} absent, skipped");
                continue;
            };
            let vs_fresh = spread(&golden_samples, &fresh);
            eprintln!(
                "  golden vs fresh C build {name:<10} differing={differing} max={max} rel={rel:.3e}",
                differing = vs_fresh.differing,
                max = vs_fresh.max,
                rel = vs_fresh.rel,
            );
            assert!(
                vs_fresh.max <= MAX_LSB_ORACLE,
                "{name}: the committed golden is not reproducible by the pinned C reference"
            );
            assert!(
                vs_fresh.rel < MAX_REL,
                "{name}: golden/C-build spread too large"
            );
        }
        ran += 1;
    }
    if ran == 0 {
        eprintln!("parity: no clips or goldens present, nothing checked");
    }
}
