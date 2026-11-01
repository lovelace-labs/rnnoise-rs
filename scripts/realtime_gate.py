#!/usr/bin/env python3
"""Real-time gate: what the allocation-counting test cannot see.

`tests/real_time_no_alloc.rs` proves the frame path allocates nothing *today*. This
script is the durable half: it reads the sources and refuses a refactor that puts an
allocation, a lock, a lazy static, I/O or logging inside a function that runs per
frame, or that reaches for runtime feature detection outside the dispatch layer.

The rules it enforces are the real-time review rules
(no threads, no `catch_unwind`, nothing before `main()`, no async in the DSP path); this
makes them tool-enforced, so the contract cannot regress between two reviews.

    make rt-gate      # part of `make check` and of CI

Two scopes, on purpose:

- **per function**: allocation and lazy statics are fine in construction functions —
  building state is the point of the issue. Also exempt: functions gated `cfg(test)` or
  behind the opt-in `simd-tier-override` feature, which are not the shipping frame path.
- **per file**: locks, threads, `catch_unwind` and await points have no construction-time
  use in a DSP module, so they are refused everywhere in the frame path.
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"

# Files where the runtime feature probe is allowed: the dispatch layer (per call, but only
# a cached load after the first probe), the construction-time warmup that fills that cache
# before the first frame, and the tier tests that call a tier directly.
DETECTION_ALLOWED = {
    "src/simd/dispatch.rs",
    "src/simd/tier.rs",
    "src/simd/x86_64/select.rs",
    "src/simd/aarch64/select.rs",
    "src/simd/warmup.rs",
    "src/fft/dispatch.rs",
    "src/simd/tests.rs",
    "src/fft/tests.rs",
}

# Modules that are on the per-frame path. `weights.rs` (and the model-builder part of
# `legacy.rs`) is construction only, and `src/bin/` is the CLI, so neither is in scope.
FRAME_PATH_FILES = [
    "src/denoise.rs",
    "src/pitch.rs",
    "src/celt_lpc.rs",
    "src/common.rs",
    "src/nnet.rs",
    "src/params.rs",
    "src/legacy.rs",
    "src/capi.rs",
    "src/simd/dispatch.rs",
    "src/simd/scalar.rs",
    "src/simd/tier.rs",
    "src/simd/x86_64/select.rs",
    "src/simd/x86_64/dense.rs",
    "src/simd/x86_64/dense_blocks.rs",
    "src/simd/x86_64/sparse8x4.rs",
    "src/simd/x86_64/xcorr.rs",
    "src/simd/x86_64/sse2.rs",
    "src/simd/x86_64/sse41.rs",
    "src/simd/x86_64/vnni.rs",
    "src/simd/aarch64/select.rs",
    "src/simd/aarch64/dotprod.rs",
    "src/fft/dispatch.rs",
    "src/fft/kiss_fft.rs",
    "src/fft/scalar.rs",
    "src/fft/x86_64.rs",
    "src/fft/spectrum.rs",
]

# Functions that build state (model, tables, FFT plans): they may allocate and may use a
# lazy static. The rule is about what runs inside `process_frame`.
CONSTRUCTION_FUNCTIONS = {
    "new",
    "new_with_model",
    "default",
    "quantize",
    "from_bytes",
    "warmup",
    "common",
    "default_model",
    "legacy_common",
    "build_stages",
    "f32_array",
    "i32_array",
    "complete_ascending_dense_form",
    "block_major_form",
    "gru",
    "take",
    "rnnoise_create",
    "rnnoise_model_from_file",
    "rnnoise_model_from_buffer",
    "read_blob",
    "parse",
    "load",
}

FORBIDDEN_IN_FUNCTIONS = {
    "heap allocation": [r"vec!\[", r"\bVec<", r"\bBox<", r"format!", r"to_vec", r"to_string", r"\bCow<", r"\bString\b"],
    "lazy static": [r"\bOnceLock\b", r"\blazy\b", r"\bOnce\b"],
    "i/o or logging": [r"println!", r"eprintln!", r"dbg!", r"log::", r"\bstd::fs\b", r"\bstd::io\b", r"\bprint!\b", r"\bInstant\b", r"\bstd::env\b"],
}

FORBIDDEN_IN_FILES = {
    "lock": [r"\bMutex\b", r"\bRwLock\b", r"\bSemaphore\b", r"\brwlock\b", r"\bspin_lock\b"],
    "forbidden work": [r"thread::spawn", r"catch_unwind", r"\basync\b", r"\.await\b"],
}

DETECTION = [r"is_x86_feature_detected", r"is_aarch64_feature_detected"]

FN_RE = re.compile(r"^\s*(?:pub(?:\(crate\))?\s+)?(?:unsafe\s+)?fn\s+([A-Za-z0-9_]+)\s*[<(]")
TEST_MOD_RE = re.compile(r"^\s*(?:#\[cfg\(test\)\]\s*)?mod\s+tests\s*\{")
# Test-only or opt-in-feature-gated code is not the shipping frame path.
NOT_IN_SHIPPING_PATH_RE = re.compile(r"#\[cfg\(\S*(test|simd-tier-override)")


def strip_comments(text: str) -> str:
    """Drop `//` and doc comments: a SAFETY paragraph that *mentions* CPUID is not a probe."""
    return "\n".join(line.split("//")[0] for line in text.splitlines())


def function_bodies(code: str):
    """Yield (fn_name, body, is_test_only) for every function definition."""
    lines = code.splitlines()
    i = 0
    while i < len(lines):
        if TEST_MOD_RE.match(lines[i]):
            depth = 0
            started = False
            while i < len(lines):
                depth += lines[i].count("{") - lines[i].count("}")
                started = started or "{" in lines[i]
                if started and depth == 0:
                    break
                i += 1
            i += 1
            continue
        m = FN_RE.match(lines[i])
        if not m:
            i += 1
            continue
        name = m.group(1)
        attrs = "\n".join(lines[:i][-6:])
        test_only = bool(NOT_IN_SHIPPING_PATH_RE.search(attrs))
        depth = 0
        body = []
        while i < len(lines):
            depth += lines[i].count("{") - lines[i].count("}")
            body.append(lines[i])
            if depth == 0 and i > 0:
                break
            i += 1
        yield name, "\n".join(body), test_only
        i += 1


def main() -> int:
    violations = []
    files = [ROOT / rel for rel in FRAME_PATH_FILES if (ROOT / rel).is_file()]

    for path in files:
        rel = path.relative_to(ROOT).as_posix()
        code = strip_comments(path.read_text(encoding="utf-8"))
        for rule, patterns in FORBIDDEN_IN_FILES.items():
            for pat in patterns:
                hit = re.search(pat, code)
                if hit:
                    violations.append(f"{rel}: {rule}: {hit.group(0)}")
        for name, body, test_only in function_bodies(code):
            if name in CONSTRUCTION_FUNCTIONS or test_only:
                continue
            for rule, patterns in FORBIDDEN_IN_FUNCTIONS.items():
                for pat in patterns:
                    hit = re.search(pat, body)
                    if hit:
                        violations.append(f"{rel}: fn {name} — {rule}: {hit.group(0)}")
        if rel not in DETECTION_ALLOWED:
            for pat in DETECTION:
                if re.search(pat, code):
                    violations.append(f"{rel}: runtime feature detection outside the dispatch layer")

    print(f"rt-gate: {len(files)} frame-path files checked — "
          f"{len(FORBIDDEN_IN_FILES)} file-level rule groups, "
          f"{len(FORBIDDEN_IN_FUNCTIONS)} function-level rule groups, feature detection")
    if violations:
        print("rt-gate FAILED:")
        for v in violations:
            print(f"  {v}")
        return 1
    print("rt-gate ok: nothing on the per-frame path allocates, locks, does I/O, logs,")
    print("          spawns, or probes CPU features outside the dispatch layer.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
