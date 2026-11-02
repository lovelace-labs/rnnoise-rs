//! Parity against the pinned upstream Xiph C reference, the oracle this crate ports.
//!
//! Three things are checked, and all skip gracefully when the artifacts are absent
//! (AGENTS.md rule 4), so this test never fails a run that lacks the data:
//!
//! 1. the port against the committed golden for every clip in `test_data/`;
//! 2. the committed golden against a *freshly built* scalar C oracle, which is what
//!    makes the golden reproducible from upstream sources instead of an opaque blob:
//!    `scripts/c_reference.sh` builds the `_c` path from the pinned commit and writes
//!    one output per clip; the CI job points `RNNOISE_C_REFERENCE_DIR` at that directory;
//! 3. the committed golden against the historical golden it was replaced with, so the
//!    difference between the two stays a single rounding.
//!
//! Between the port and the golden the gate is **exactness**, not a spread: the goldens come
//! from the pinned pure-C recipe, the port reproduces that path operation for operation, and
//! it does so deterministically (its own `log10`, no FMA contraction), so a differing sample
//! is a behaviour change that has to be justified against the reference — not a rounding to
//! absorb. The 1-LSB allowance stays only where a *different compiler or libc* is compared
//! with committed bytes: a fresh C build on another machine, and the golden the earlier clang
//! build (FP contraction enabled) produced.
//!
//! Measured with the recipe in `scripts/c_reference.sh`: 0 differing samples for the port on
//! all four clips, 0 between the goldens and a fresh gcc build, and 4 samples (1 LSB) against
//! the historical clang golden on the speech clip.
#![cfg(feature = "embedded-model")]
// Two gates here, and both need the **float** model: the committed goldens are that model's
// output, and `embedded-model-int8` is a different, deliberately non-bit-exact model (rel energy
// 4.8e-6, up to 13 LSB on the speech clip). Checking the int8 path against these goldens would
// assert a property the int8 path never claimed — its accuracy gate is `quantized_accuracy` in
// `tests/parity.rs` (`rel < 5e-3`), not bit-exactness. With `default-features = false` there is no
// default model at all, so the file compiles out rather than failing; the all-features job still
// runs it.

use std::path::{Path, PathBuf};

use rnnoise::{DenoiseState, FRAME_SIZE};

const MAX_LSB_BUILD: i32 = 1;
const MAX_REL: f64 = 1e-6;

struct Clip {
    name: &'static str,
    input: PathBuf,
    golden: PathBuf,
    /// The golden this clip's golden replaced, kept for provenance: `ref_out.raw` came from a
    /// clang build with FP contraction, which rounds differently.
    historical: Option<PathBuf>,
}

fn clips() -> Vec<Clip> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("test_data");
    let goldens = root.join("goldens");
    let mut out = vec![Clip {
        name: "testing",
        input: root.join("testing.raw"),
        golden: root.join("ref_out.raw"),
        historical: Some(goldens.join("testing.c-clang.raw")),
    }];
    for name in ["tone_noise", "silence", "zero"] {
        out.push(Clip {
            name,
            input: root.join("clips").join(format!("{name}.raw")),
            golden: goldens.join(format!("{name}.raw")),
            historical: None,
        });
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

fn report(kind: &str, name: &str, s: &Spread) {
    eprintln!(
        "{kind} {name:<10} samples={samples} differing={differing} max={max} rel={rel:.3e}",
        samples = s.samples,
        differing = s.differing,
        max = s.max,
        rel = s.rel,
    );
}

#[test]
fn port_matches_the_committed_goldens() {
    let oracle = std::env::var("RNNOISE_C_REFERENCE_DIR").ok();
    let mut ran = 0;
    for clip in clips() {
        let Some(input) = load_i16(&clip.input) else {
            eprintln!("parity: clip {} absent, skipped", clip.name);
            continue;
        };
        let Some(golden_samples) = load_i16(&clip.golden) else {
            eprintln!("parity: golden for {} absent, skipped", clip.name);
            continue;
        };
        let got = port_output(&input);
        assert_eq!(
            got.len(),
            golden_samples.len(),
            "{}: output length",
            clip.name
        );
        let s = spread(&golden_samples, &got);
        report("port vs golden     ", clip.name, &s);
        assert_eq!(
            s.differing, 0,
            "{}: the port is no longer bit-exact against the pinned golden (max {} LSB)",
            clip.name, s.max
        );

        if let Some(dir) = &oracle {
            let Some(fresh) = load_i16(&Path::new(dir).join(format!("{}.raw", clip.name))) else {
                eprintln!("  oracle output for {} absent, skipped", clip.name);
                continue;
            };
            let vs_fresh = spread(&golden_samples, &fresh);
            report("  golden vs build  ", clip.name, &vs_fresh);
            assert!(
                vs_fresh.max <= MAX_LSB_BUILD,
                "{}: the committed golden is not reproducible by the pinned C reference",
                clip.name
            );
            assert!(
                vs_fresh.rel < MAX_REL,
                "{}: golden/C-build spread too large",
                clip.name
            );
        }

        if let Some(path) = &clip.historical {
            let Some(old) = load_i16(path) else {
                eprintln!("  historical golden for {} absent, skipped", clip.name);
                continue;
            };
            let vs_old = spread(&golden_samples, &old);
            report("  golden vs clang  ", clip.name, &vs_old);
            assert!(
                vs_old.max <= MAX_LSB_BUILD,
                "{}: the pinned-recipe golden and the historical clang golden differ by more \
                 than one rounding (see doc/parity.md)",
                clip.name
            );
            assert!(
                vs_old.rel < MAX_REL,
                "{}: golden/golden spread too large",
                clip.name
            );
        }
        ran += 1;
    }
    if ran == 0 {
        eprintln!("parity: no clips or goldens present, nothing checked");
    }
}
