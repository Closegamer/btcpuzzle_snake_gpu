# Results and measurements

All numbers below were measured on a single **RTX 5070 Ti Laptop** under WSL Ubuntu 22.04, CUDA 12.8
via PTX `compute_90` (JIT). Timings include the full GPU path: key -> point -> compressed pubkey ->
SHA-256 -> RIPEMD-160 -> address compare.

`writtenBy('Closegamer', 2026)`

## Throughput

| mode | speed | notes |
|---|---|---|
| `--spread 71 --load 90` | 162.9 Mkeys/s | 83 °C, ~104 W |
| `--spread 71 --load 100` | 179.5 Mkeys/s | 2460 MHz |
| `--exhaust` (fixed popcount slices) | 173–216 Mkeys/s | 216 Mkeys/s on the densest slices |
| `scan_range` (arbitrary range, +G walk, batch inversion 128) | ~570 Mkeys/s | best per-key path in this repo |

Why the difference: in the snake walk one key also pays for the 1 -> 160 point ladder (~1 doubling
per key), while `scan_range` pays only one point addition per key. Batching the inversion (128
points per inversion instead of 16) alone was worth **+55 %** for `scan_range`.

## Exhausted slices of puzzle 71

Deterministic, gap-free sweeps over complete fixed-popcount slices (`--exhaust`), i.e. every key of
puzzle 71 with exactly *k* set bits among the low 70 bits:

| slice | keys | wall time |
|---|---|---|
| k = 1..12 | 2.5e9 (k<=10) + ... | 2 247 s (k=11), 12 484 s (k=12) |
| k = 60 | 2 163 842 859 360 | 12 296 s (~3.4 h) |
| k = 61..71 | 9.4e11 | 2 216 s (k=61) and less for the rest |

Total **4 272 728 461 792 keys checked, 0 keys found** (k = 1..12 and k = 60..71).

Nothing is known about the middle slices (k = 13..59). The largest "reasonable" next slice,
k = 13, is `C(70,12) = 1.06e13` keys, i.e. ~17 h at 173 Mkeys/s.

## Why this is a lottery

For a flat prior the expected number of keys to test is `2^(P-1)` per puzzle, independent of the
order in which you test them. Puzzle 71 (`2^70 = 1.18e21` keys) at 180 Mkeys/s needs about
**208 000 years** for one full sweep; at 570 Mkeys/s (range scanner) about **66 000 years**.
A year of continuous mining covers `5.7e15` keys = `4.8e-6` of the band.

Heuristic orderings were evaluated too (key bit statistics, n-gram frequencies, positional priors,
ray/parabola/multiplier families, layered searches around the positional median). None of them beat a
uniform prior by more than ~20 %, so the honest statement is: this is a lottery ticket with a very
small probability, not a solvable problem with this hardware.

## Self-tests

`./snake --test` runs, and refuses to continue on failure:

- CPU field arithmetic vs a reference, and CPU point arithmetic vs a known generator multiple;
- a 1M-key GPU field check against the CPU;
- a planted-key GPU test (a key is planted, the kernel must find it);
- a sample record through the same path that reports real finds.

`cuda/scan_range` additionally finds the real key of a solved puzzle when its address and range are
given, which is the end-to-end check of the hashing pipeline.
