#!/usr/bin/env bash
# Build the upstream Xiph RNNoise C reference — the scalar (`_c`) compute path, the
# same path this crate ports — and run it over the test clips.
#
# `test_data/ref_out.raw` and `test_data/goldens/*.raw` are the output of exactly this
# build. CI re-runs it and compares against the committed goldens: the port must be
# bit-exact with them, a fresh build on another compiler is allowed 1 LSB (doc/parity.md).
# So a golden is reproducible from upstream sources instead of being an opaque blob: a
# change in the C reference or in this port shows up here.
#
# Usage: scripts/c_reference.sh [output-dir]
#   output-dir  where the per-clip outputs are written (default: .c_reference)
#
# Needs: git, curl, cc, sha256sum (or shasum). Skips with exit 0 if the pinned
# upstream sources or the model cannot be fetched (offline runner), so `make check`
# stays green without network access.
set -eu

PINNED_COMMIT="70f1d256acd4b34a572f999a05c87bf00b67730d"
MODEL_HASH="0a8755f8e2d834eff6a54714ecc7d75f9932e845df35f8b59bc52a7cfe6e8b37"

REPO="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-$REPO/.c_reference}"
CACHE="${RNNOISE_REF_CACHE:-$REPO/bup/rnnoise-$PINNED_COMMIT}"
ARCH="$(uname -m)"

skip() {
  echo "c_reference: $1 — skipping (the parity test skips gracefully when the oracle is absent)" >&2
  exit 0
}

command -v git >/dev/null  || skip "git missing"
command -v curl >/dev/null || skip "curl missing"
command -v cc >/dev/null   || skip "no C compiler"

# --- sources --------------------------------------------------------------------
if [ ! -d "$CACHE" ]; then
  mkdir -p "$(dirname "$CACHE")"
  git clone --quiet https://github.com/xiph/rnnoise.git "$CACHE" ||
    skip "could not fetch the pinned upstream sources"
fi
git -C "$CACHE" checkout --quiet "$PINNED_COMMIT" || skip "pinned commit not in the clone"
[ "$(git -C "$CACHE" rev-parse HEAD)" = "$PINNED_COMMIT" ] || skip "clone is at the wrong commit"
cd "$CACHE"

# The model is not in the git tree: `download_model.sh` fetches it and validates it
# against `model_version`. Keep only the two files the build needs.
if [ ! -f "$CACHE/src/rnnoise_data.c" ]; then
  tarball="$CACHE/rnnoise_data-$MODEL_HASH.tar.gz"
  curl -fsSL -o "$tarball" "https://media.xiph.org/rnnoise/models/rnnoise_data-$MODEL_HASH.tar.gz" ||
    skip "could not fetch the model"
  if command -v sha256sum >/dev/null; then
    actual="$(sha256sum "$tarball" | awk '{print $1}')"
  else
    actual="$(shasum -a 256 "$tarball" | awk '{print $1}')"
  fi
  [ "$actual" = "$MODEL_HASH" ] || { rm -f "$tarball"; skip "model checksum mismatch ($actual)"; }
  # -C after the member list is GNU-tar-only; BSD tar (macOS) errors. Extract in the
  # cache directory itself.
  ( cd "$CACHE" && tar -xzf "$tarball" src/rnnoise_data.c src/rnnoise_data.h )
fi

[ "$(cat "$CACHE/model_version")" = "$MODEL_HASH" ] ||
  skip "model_version in the clone ($MODEL_HASH expected) changed"

# --- scalar oracle ---------------------------------------------------------------
# `nnet_default.c` pins `RTCD_ARCH = c`, and the `vec.h` branch must be the pure-C one:
# `-U__SSE2__ -U__AVX__` is what selects it on x86-64 (the compiler predefines `__SSE2__`,
# so `vec.h` would otherwise include `vec_avx.h` — a different `tanh_approx`/`sigmoid_approx`
# and a different `sgemv`, i.e. a different set of roundings), and `-DDISABLE_NEON` is the
# equivalent on AArch64. Without both, the build is not the path this crate ports and it
# disagrees with the goldens by ~1800 1-LSB samples instead of a handful.
# Upstream's autotools default is the int8/OSCE compute path (--disable-dnn-debug-float):
# a different model of the network, not this oracle.
# `-ffp-contract=off` keeps `a*b + c` uncontracted, as the Rust port does, so the residual
# against the port is pure rounding. `-Wno-cpp` hides upstream's `#warning`s.
FLAGS=(-O2 -Wall -Wno-cpp -ffp-contract=off -U__SSE2__ -U__AVX__ -Iinclude -Isrc -DDISABLE_NEON)
mkdir -p "$OUT"
cc "${FLAGS[@]}" \
  src/denoise.c src/rnn.c src/pitch.c src/kiss_fft.c src/celt_lpc.c \
  src/nnet.c src/nnet_default.c src/parse_lpcnet_weights.c \
  src/rnnoise_tables.c src/rnnoise_data.c examples/rnnoise_demo.c \
  -lm -o "$OUT/rnnoise_demo_c"

echo "c_reference: built scalar oracle from $PINNED_COMMIT (arch=$ARCH, model=$MODEL_HASH)"

# --- clips ----------------------------------------------------------------------
clips=("$REPO/test_data/testing.raw")
for c in "$REPO"/test_data/clips/*.raw; do
  [ -e "$c" ] && clips+=("$c")
done

for clip in "${clips[@]}"; do
  name="$(basename "$clip" .raw)"
  "$OUT/rnnoise_demo_c" "$clip" "$OUT/$name.raw"
  echo "  $name.raw -> $OUT/$name.raw"
done

if [ "${WRITE_GOLDENS:-0}" = "1" ]; then
  # Refreshing a golden is only correct after a behaviour change that has been justified
  # against the reference, because the gate asserts bit-exactness against these bytes.
  # The golden the earlier clang build produced is kept as
  # test_data/goldens/testing.c-clang.raw for provenance.
  mkdir -p "$REPO/test_data/goldens"
  for clip in "${clips[@]}"; do
    name="$(basename "$clip" .raw)"
    if [ "$(dirname "$clip")" = "$REPO/test_data/clips" ]; then
      cp "$OUT/$name.raw" "$REPO/test_data/goldens/$name.raw"
    else
      cp "$OUT/$name.raw" "$REPO/test_data/ref_out.raw"
    fi
  done
  echo "c_reference: goldens refreshed in test_data/ (pure-C path)"
fi
