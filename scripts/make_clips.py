#!/usr/bin/env python3
"""Generate the deterministic test clips under `test_data/clips/`.

Every clip is reproducible from this script (fixed LCG, no random seed), so a golden
can be regenerated end-to-end: `make clips` then `make reference` (which runs the
pinned scalar C reference and refreshes `test_data/goldens/`).

The clips complement `test_data/testing.raw` (real speech):

  tone_noise.raw  a 440 Hz tone over a noise floor — drives the pitch search and the
                  GRU with a stable period, which the speech clip does not isolate.
  silence.raw     a low-level noise floor — the "no speech" branch of the front-end.
  zero.raw        digital silence — the path where `process_frame` skips the network.

Format: little-endian signed 16-bit mono at 48 kHz, whole frames (480 samples).
"""
import math
import os

DUR_S = 1.0
RATE = 48_000
N = int(DUR_S * RATE)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "test_data", "clips")


def lcg(seed, n):
    """Deterministic generator (Numerical Recipes LCG), values in [-1, 1)."""
    s = seed
    for _ in range(n):
        s = (s * 1664525 + 1013904223) & 0xFFFFFFFF
        yield (s >> 8) / 16777216.0 - 0.5


def write(name, samples):
    path = os.path.join(OUT, name)
    with open(path, "wb") as f:
        for s in samples:
            f.write(int(s).to_bytes(2, "little", signed=True))
    print(f"{name}: {len(samples)} samples ({len(samples) / 480:.0f} frames)")


def main():
    os.makedirs(OUT, exist_ok=True)

    tone = [
        int(3000.0 * math.sin(i * 2 * math.pi * 440.0 / RATE) + 2000.0 * v)
        for i, v in enumerate(lcg(1, N))
    ]
    write("tone_noise.raw", tone)

    quiet = [int(8.0 * v) for v in lcg(7, N)]
    write("silence.raw", quiet)

    write("zero.raw", [0] * N)


if __name__ == "__main__":
    main()
