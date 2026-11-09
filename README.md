# btcpuzzle_snake_gpu — CUDA brute-forcer for the Bitcoin puzzle (keys 71..160)

`writtenBy('Closegamer', 2026)`

A single-file CUDA brute-forcer for the [Bitcoin puzzle](https://privatekeys.pw/puzzles/bitcoin-puzzle-tx)
keyspace. One C++ file (`cuda/snake.cu`), no external dependencies, plus a small tool that scans an
arbitrary key range.

The idea ("snake") is to walk the whole ladder of puzzles with one 160-bit path instead of testing
independent random keys per puzzle:

```
V                     random 159-bit number
key_P  =  2^(P-1) + (V mod 2^(P-1))         P = 71..160
```

Consecutive puzzles differ by roughly one doubling, so the GPU walks 71 -> 160 with **one point
addition per puzzle** and hashes one key per puzzle per step. After a full pass `V` is shifted one
bit (the "snake step") and the walk repeats. Solved puzzles (75, 80, ... 135) are used as test
vectors: when the walk passes exactly through a known key, the program reports a hit.

## Quick start

Requirements: an NVIDIA GPU, Windows with WSL2 (Ubuntu 22.04) or Linux, and a CUDA 12.x toolkit
inside that Linux environment. The binary is not stored in git: the launchers build it on first run.

The tree is checked by a 151-case mode/parameter matrix; see `docs/TESTING.md` and the quick
`tools/smoke_test.sh`.

### Windows (WSL)

```bat
git clone https://github.com/Closegamer/btcpuzzle_snake_gpu.git
cd btcpuzzle_snake_gpu
run_windows.bat
```

`run_windows.bat` with no arguments builds the binary if needed (or if the sources changed) and
starts the **snake walk over puzzles 71..160**. Arguments are passed to the binary:

```bat
run_windows.bat --help
run_windows.bat --test
run_windows.bat --spread 71 --load 90
run_windows.bat --freeze 71 --load 90
run_windows.bat --seconds 60
```

### Linux / WSL

```bash
git clone https://github.com/Closegamer/btcpuzzle_snake_gpu.git
cd btcpuzzle_snake_gpu
bash cuda/run.sh
```

`cuda/run.sh` does the same as the Windows launcher: detects CUDA, builds on first run, then starts
the snake. It forwards all arguments:

```bash
bash cuda/run.sh --help
bash cuda/run.sh --test
bash cuda/run.sh --spread 71 --kmin 13 --kmax 59 --load 90
bash cuda/run.sh --snake --seconds 60
```

If you prefer to build and run manually:

```bash
bash cuda/build.sh
cd cuda
export LD_LIBRARY_PATH=/usr/lib/wsl/lib   # under WSL only
./snake                                   # snake walk (default)
./snake --help                            # all modes
```

The CUDA toolkit is auto-detected in `CUDA_HOME` (if it points at a toolkit), then
`~/cudaroot/usr/local/cuda-12.8`, `/usr/local/cuda-12.8`, `/usr/local/cuda`, `/opt/cuda`.
Set `CUDA_HOME` yourself if your toolkit lives elsewhere.

## Modes

Run `./snake --help` (or `run_windows.bat --help`) for the full list.

| mode | what it does |
|---|---|
| `--snake` (default) | The main search: one 160-bit path visits every unsolved puzzle 71..160 once per step; after a full pass the path shifts one bit. |
| `--spread P` | One puzzle only: random keys with a fixed number of set bits, the popcount slices `kmin..kmax` split across GPU walkers. |
| `--exhaust P` | Deterministic, gap-free walk over complete fixed-popcount slices of puzzle `P` (`P = 4..128`; progress in `exhaust_results.txt`). |
| `--freeze P` | Freeze `M` ones at random positions and sweep every other low bit (`P = 3..128`); the mask database `freeze_masks_p<P>.txt` resumes later runs. |
| `--test` | Built-in CPU/GPU self-tests, planted-key check, sample record. Short and safe. |
| `--ladder` | Demo: solve known puzzles `--from A` to `--to B` in front of you. |
| `--show N` | Print the first `N` snake steps as 160-bit keys, then exit (no search). |

Common options: `--load PCT` (GPU duty cycle, default 100), `--seconds N` (stop after N seconds),
`--nocool` (disable the 10-minute cooling break every 50 minutes), `--addr FILE` (address list,
default `../allPuzzles.js`), `--watch` (visualise walkers against a known key).

`snake` reads the puzzle addresses from `../allPuzzles.js` (one compressed P2PKH address per line,
puzzle `p` on line `p`), so run it from the `cuda/` directory of this repository.

## Output

- `cuda/found.txt` — real hits only, one record per found key (hex, WIF, address, timestamp).
- `cuda/snake.log` — the console output as a log. Progress is appended as one newline-terminated
  line every 10 seconds (the live line on screen is still redrawn in place), so `tail -f snake.log`
  works.
- `cuda/freeze_masks_p<P>.txt` — freeze-mode progress. Never delete it: it is the "do not repeat"
  list of already swept masks.
- `cuda/exhaust_results.txt` — completed `--exhaust` slices.

## Range scanner (bonus)

`cuda/scan_range` checks an **arbitrary consecutive key range** `start .. start+count-1` against one
address. It walks with `+G` per key and inverts in batches, which is the fastest way to test a
specific slice of a puzzle band:

```bash
bash cuda/build_scan.sh          # generates scan_range.cu, then builds ./cuda/scan_range
./cuda/scan_range --start 1770887431076116955136 --count 2000000000 --target <address>
```

Measured on the same laptop: **~570 Mkeys/s** (batch inversion of 128 points). For comparison, the
snake walk checks all 77 unsolved puzzles per step, so its aggregate counter is higher (~340 M
key-checks/s in a short smoke test), but every key still pays one point addition.

## Measured performance (RTX 5070 Ti Laptop, WSL, PTX compute_90)

| what | speed |
|---|---|
| `--spread 71 --load 90` | 162.9 Mkeys/s (83 °C) |
| `--spread 71 --load 100` | 179.5 Mkeys/s |
| `--snake` (all unsolved puzzles) | ~340 M key-checks/s aggregate (1-second smoke test) |
| `--exhaust` slices | 173-216 Mkeys/s |
| `scan_range` (batched inversion) | ~570 Mkeys/s |

Exhausted and reported here for honesty (details in `docs/RESULTS.md`): puzzle 71's slices
**k = 1..12** (few set bits) and **k = 60..71** (almost all bits set) — 4.27e12 keys, **0 keys found**.

## Honest expectations

The traversal order does not change the odds: for a flat prior, the chance per checked key is
`2^-(P-1)`, so a full sweep of puzzle 71 (`2^70` keys) at ~180 Mkeys/s takes about **208 000 years**
on one laptop. Slices and heuristics only change *which* keys are tried first. Treat this as a
research/learning tool and a lottery ticket, not a plan.

## License

MIT — see `LICENSE`. Copyright (c) 2026 Closegamer.

## Donations

If this code was useful to you:

```
bc1qh9crcvmfvx2qrg8lwm5leg05ytx87uctd9l3lp   (BTC)
```
