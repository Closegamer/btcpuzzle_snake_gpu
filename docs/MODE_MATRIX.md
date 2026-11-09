# btcpuzzle_snake_gpu mode matrix — 2026-11-09 15:52:01

Clean clone of this repository at commit **`88884ca`**.

- GPU: NVIDIA GeForce RTX 5070 Ti Laptop GPU
- CUDA: `~/cudaroot/usr/local/cuda-12.8`
- Build: `cuda/build.sh` + `cuda/build_scan.sh` — OK
- Cases: **151**, passed **151**, failed **0**, wall time 123.5 s
- The clone is temporary; runtime files (`found.txt`, freeze masks, `exhaust_results.txt`) do not touch the real repository.

## Matrix

| # | case | command | expected | rc | verdict | note |
|---:|---|---|---|---:|---|---|
| 1 | snake | `snake --help` | rc=0 | 0 | PASS | all modes listed |
| 2 | snake | `snake -h` | rc=0 | 0 | PASS |  |
| 3 | snake | `snake --bogus` | rc!=0 | 1 | PASS |  |
| 4 | snake | `snake` | timeout | timeout | PASS | no args = default snake (killed after 3 s) |
| 5 | snake | `snake --spread` | rc!=0 | 1 | PASS | missing value |
| 6 | snake | `snake --exhaust` | rc!=0 | 1 | PASS | missing value |
| 7 | snake | `snake --freeze` | rc!=0 | 1 | PASS | missing value |
| 8 | snake | `snake --kmin` | rc!=0 | 1 | PASS | missing value |
| 9 | snake | `snake --kmax` | rc!=0 | 1 | PASS | missing value |
| 10 | snake | `snake --frz` | rc!=0 | 1 | PASS | missing value |
| 11 | snake | `snake --from` | rc!=0 | 1 | PASS | missing value |
| 12 | snake | `snake --to` | rc!=0 | 1 | PASS | missing value |
| 13 | snake | `snake --ref` | rc!=0 | 1 | PASS | missing value |
| 14 | snake | `snake --ones` | rc!=0 | 1 | PASS | missing value |
| 15 | snake | `snake --chunkfrom` | rc!=0 | 1 | PASS | missing value |
| 16 | snake | `snake --load` | rc!=0 | 1 | PASS | missing value |
| 17 | snake | `snake --seconds` | rc!=0 | 1 | PASS | missing value |
| 18 | snake | `snake --addr` | rc!=0 | 1 | PASS | missing value |
| 19 | snake | `snake --show` | rc!=0 | 1 | PASS | missing value |
| 20 | snake | `snake --snake --spread 71` | rc!=0 | 1 | PASS |  |
| 21 | snake | `snake --snake --exhaust 16` | rc!=0 | 1 | PASS |  |
| 22 | snake | `snake --snake --freeze 16` | rc!=0 | 1 | PASS |  |
| 23 | snake | `snake --snake --ladder` | rc!=0 | 1 | PASS |  |
| 24 | snake | `snake --vis` | rc!=0 | 1 | PASS |  |
| 25 | snake | `snake --snake --vis` | rc!=0 | 1 | PASS |  |
| 26 | snake | `snake --snake --seconds 1` | rc=0 | 0 | PASS |  |
| 27 | snake | `snake --snake --seconds 0.5` | rc=0 | 0 | PASS |  |
| 28 | snake | `snake --snake --seconds 1 --nocool` | rc=0 | 0 | PASS |  |
| 29 | snake | `snake --snake --seconds 1 --load 5` | rc=0 | 0 | PASS |  |
| 30 | snake | `snake --snake --seconds 1 --load 1` | rc=0 | 0 | PASS | clamps to 5 |
| 31 | snake | `snake --snake --seconds 1 --load 200` | rc=0 | 0 | PASS | clamps to 100 |
| 32 | snake | `snake --snake --seconds 1 --load abc` | rc=0 | 0 | PASS | atoi(0) -> 5 |
| 33 | snake | `snake --snake --seconds 1 --load -5` | rc=0 | 0 | PASS | clamps to 5 |
| 34 | snake | `snake --snake --seconds 1 --addr ../allPuzzles.js` | rc=0 | 0 | PASS |  |
| 35 | snake | `snake --snake --seconds 1 --addr <clone>/allPuzzles.js` | rc=0 | 0 | PASS |  |
| 36 | snake | `snake --snake --addr /nonexistent` | rc!=0 | 1 | PASS |  |
| 37 | snake | `snake --snake --addr <tmp>/short.js` | rc!=0 | 1 | PASS | 100 lines |
| 38 | snake | `snake --snake --addr <tmp>/bad.js` | rc!=0 | 1 | PASS | bad Base58 at line 71 |
| 39 | snake | `snake --snake --seconds 1 --watch --ref 3 --ones 3` | rc=0 | 0 | PASS |  |
| 40 | snake | `snake --snake --seconds 1 --watch --ref 16 --ones 8` | rc=0 | 0 | PASS |  |
| 41 | snake | `snake --snake --seconds 1 --watch --ref 70` | rc=0 | 0 | PASS |  |
| 42 | snake | `snake --snake --seconds 1 --watch --ref 71` | rc!=0 | 1 | PASS |  |
| 43 | snake | `snake --snake --seconds 1 --check` | rc=0 | 0 | PASS | ignored outside freeze |
| 44 | snake | `snake --snake --seconds 1 --kmin 5 --kmax 5` | rc=0 | 0 | PASS | ignored in snake |
| 45 | snake | `snake --snake --seconds 1 --chunkfrom 5` | rc=0 | 0 | PASS | ignored in snake |
| 46 | snake | `snake --snake --seconds 1 --frz 5` | rc=0 | 0 | PASS | ignored in snake |
| 47 | snake | `snake --snake --seconds 1 --from 1 --to 2` | rc=0 | 0 | PASS | ignored in snake |
| 48 | snake | `snake --show 1` | rc=0 | 0 | PASS |  |
| 49 | snake | `snake --show 2` | rc=0 | 0 | PASS |  |
| 50 | snake | `snake --show 0 --seconds 1` | rc=0 | 0 | PASS | show 0 = normal snake |
| 51 | snake | `snake --show -1 --seconds 1` | rc=0 | 0 | PASS | show -1 = normal snake |
| 52 | snake | `snake --test` | rc=0 | 0 | PASS |  |
| 53 | snake | `snake --snake --test` | rc=0 | 0 | PASS |  |
| 54 | snake | `snake --alert-test` | rc=0 | 0 | PASS | repo has no alert.py |
| 55 | snake | `snake --spread 16 --kmin 2 --kmax 2 --seconds 1` | rc=0 | 0 | PASS |  |
| 56 | snake | `snake --spread 70 --kmin 2 --kmax 2 --seconds 1` | rc=0 | 0 | PASS |  |
| 57 | snake | `snake --spread 71 --seconds 1` | rc=0 | 0 | PASS |  |
| 58 | snake | `snake --spread 160 --kmin 2 --kmax 2 --seconds 1` | rc=0 | 0 | PASS |  |
| 59 | snake | `snake --spread 160 --kmin 160 --kmax 160 --seconds 1` | rc=0 | 0 | PASS |  |
| 60 | snake | `snake --spread 16 --kmin 1 --kmax 1 --seconds 1` | rc=0 | 0 | PASS |  |
| 61 | snake | `snake --spread 16 --kmin 20 --kmax 30 --seconds 1` | rc=0 | 0 | PASS |  |
| 62 | snake | `snake --spread 16 --kmin 5 --kmax 3 --seconds 1` | rc=0 | 0 | PASS |  |
| 63 | snake | `snake --spread 16 --kmin 2 --kmax 2 --load 5 --seconds 1` | rc=0 | 0 | PASS |  |
| 64 | snake | `snake --spread 16 --kmin 2 --kmax 2 --nocool --seconds 1` | rc=0 | 0 | PASS |  |
| 65 | snake | `snake --spread 3 --watch --ref 3 --ones 3 --seconds 1` | rc=0 | 0 | PASS | spread watch shows the walker; --ones is not used here |
| 66 | snake | `snake --spread 16 --watch --ref 16 --ones 8 --seconds 1` | rc=0 | 0 | PASS |  |
| 67 | snake | `snake --spread 16 --watch --ref 16 --ones 0 --seconds 1` | rc=0 | 0 | PASS |  |
| 68 | snake | `snake --spread 16 --watch --ref 70 --ones 8 --seconds 1` | rc=0 | 0 | PASS |  |
| 69 | snake | `snake --spread 2` | rc!=0 | 1 | PASS |  |
| 70 | snake | `snake --spread 161` | rc!=0 | 1 | PASS |  |
| 71 | snake | `snake --spread 0` | rc!=0 | 1 | PASS |  |
| 72 | snake | `snake --exhaust 4 --kmin 2 --kmax 2` | rc=0 | 0 | PASS |  |
| 73 | snake | `snake --exhaust 16 --kmin 2 --kmax 2` | rc=0 | 0 | PASS |  |
| 74 | snake | `snake --exhaust 71 --kmin 2 --kmax 2` | rc=0 | 0 | PASS |  |
| 75 | snake | `snake --exhaust 128 --kmin 2 --kmax 2` | rc=0 | 0 | PASS |  |
| 76 | snake | `snake --exhaust 3` | rc!=0 | 1 | PASS |  |
| 77 | snake | `snake --exhaust 0` | rc!=0 | 1 | PASS |  |
| 78 | snake | `snake --exhaust 129` | rc!=0 | 1 | PASS |  |
| 79 | snake | `snake --exhaust 130` | rc!=0 | 1 | PASS |  |
| 80 | snake | `snake --exhaust 16 --kmin 1 --kmax 1` | rc=0 | 0 | PASS |  |
| 81 | snake | `snake --exhaust 16 --kmin 16 --kmax 16` | rc=0 | 0 | PASS |  |
| 82 | snake | `snake --exhaust 16 --kmin 17 --kmax 17` | rc=0 | 0 | PASS |  |
| 83 | snake | `snake --exhaust 16 --kmin 2 --kmax 2 --force` | rc=0 | 0 | PASS |  |
| 84 | snake | `snake --exhaust 16 --kmin 2 --kmax 2 --load 5` | rc=0 | 0 | PASS |  |
| 85 | snake | `snake --exhaust 16 --kmin 2 --kmax 2 --seconds 0.1` | rc=0 | 0 | PASS | seconds ignored by exhaust |
| 86 | snake | `snake --exhaust 71 --kmin 36 --kmax 36` | rc=0 | 0 | PASS |  |
| 87 | snake | `snake --exhaust 20 --kmin 7 --kmax 7` | rc=0 | 0 | PASS |  |
| 88 | snake | `snake --exhaust 20 --kmin 7 --kmax 8` | rc=0 | 0 | PASS |  |
| 89 | snake | `snake --exhaust 20 --kmin 7 --kmax 7 --chunkfrom 0` | rc=0 | 0 | PASS |  |
| 90 | snake | `snake --exhaust 20 --kmin 7 --kmax 7 --chunkfrom 1` | rc=0 | 0 | PASS |  |
| 91 | snake | `snake --exhaust 20 --kmin 7 --kmax 7 --chunkfrom 2` | rc=0 | 0 | PASS |  |
| 92 | snake | `snake --exhaust 20 --kmin 7 --kmax 7 --chunkfrom 4` | rc=0 | 0 | PASS |  |
| 93 | snake | `snake --exhaust 20 --kmin 7 --kmax 7 --chunkfrom 99` | rc=0 | 0 | PASS |  |
| 94 | snake | `snake --exhaust 16 --kmin 2 --kmax 2 --chunkfrom 1` | rc=0 | 0 | PASS |  |
| 95 | snake | `snake --freeze 16 --frz 8 --seconds 1` | rc=0 | 0 | PASS |  |
| 96 | snake | `snake --freeze 16 --seconds 1` | rc=0 | 0 | PASS |  |
| 97 | snake | `snake --freeze 16 --frz 1 --seconds 1` | rc=0 | 0 | PASS |  |
| 98 | snake | `snake --freeze 16 --frz 15 --seconds 1` | rc=0 | 0 | PASS |  |
| 99 | snake | `snake --freeze 16 --frz 0 --seconds 1` | rc=0 | 0 | PASS |  |
| 100 | snake | `snake --freeze 16 --frz 8 --check --seconds 1` | rc=0 | 0 | PASS |  |
| 101 | snake | `snake --freeze 3 --frz 1 --check --seconds 1` | rc=0 | 0 | PASS |  |
| 102 | snake | `snake --freeze 3 --frz 2 --check --seconds 1` | rc=0 | 0 | PASS |  |
| 103 | snake | `snake --freeze 71 --frz 50 --seconds 1` | rc=0 | 0 | PASS |  |
| 104 | snake | `snake --freeze 71 --seconds 1` | rc=0 | 0 | PASS |  |
| 105 | snake | `snake --freeze 128 --frz 100 --seconds 1` | rc=0 | 0 | PASS |  |
| 106 | snake | `snake --freeze 71 --frz 20 --seconds 1` | rc!=0 | 1 | PASS |  |
| 107 | snake | `snake --freeze 2` | rc!=0 | 1 | PASS |  |
| 108 | snake | `snake --freeze 0` | rc!=0 | 1 | PASS |  |
| 109 | snake | `snake --freeze 129` | rc!=0 | 1 | PASS |  |
| 110 | snake | `snake --freeze 130` | rc!=0 | 1 | PASS |  |
| 111 | snake | `snake --freeze 16 --frz 16 --seconds 1` | rc!=0 | 1 | PASS |  |
| 112 | snake | `snake --freeze 16 --frz 16 --vis --seconds 1` | rc!=0 | 1 | PASS |  |
| 113 | snake | `snake --freeze 129 --vis --seconds 1` | rc!=0 | 1 | PASS |  |
| 114 | snake | `snake --freeze 16 --frz 8 --force --seconds 1` | rc=0 | 0 | PASS |  |
| 115 | snake | `snake --freeze 16 --frz 8 --load 5 --seconds 1` | rc=0 | 0 | PASS |  |
| 116 | snake | `snake --freeze 16 --frz 8 --watch --seconds 1` | rc=0 | 0 | PASS |  |
| 117 | snake | `snake --freeze 16 --frz 8 --vis --seconds 1` | rc=0 | 0 | PASS |  |
| 118 | snake | `snake --freeze 71 --vis --seconds 1` | rc=0 | 0 | PASS |  |
| 119 | snake | `snake --ladder --from 1 --to 5` | rc=0 | 0 | PASS |  |
| 120 | snake | `snake --ladder --from 1 --to 1` | rc=0 | 0 | PASS |  |
| 121 | snake | `snake --ladder --from 0 --to 2` | rc=0 | 0 | PASS |  |
| 122 | snake | `snake --ladder --to 0` | rc=0 | 0 | PASS |  |
| 123 | snake | `snake --ladder --from 5 --to 1` | rc=0 | 0 | PASS |  |
| 124 | snake | `snake --ladder --from 1 --to 5 --seconds 1` | rc=0 | 0 | PASS | seconds ignored by ladder |
| 125 | snake | `snake --ladder --from 1 --to 5 --load 5` | rc=0 | 0 | PASS | load ignored by ladder |
| 126 | snake | `snake --ladder --from 1 --to 5 --watch` | rc=0 | 0 | PASS |  |
| 127 | scan | `scan --plant 12345 --start 12000 --count 1000` | rc=0 | 0 | PASS |  |
| 128 | scan | `scan --plant 51510 --start 1 --count 60000` | rc=0 | 0 | PASS | start is a power of two (regression) |
| 129 | scan | `scan --plant 51510 --start 1 --count 60000 --blocks 2048` | rc=0 | 0 | PASS | same with more blocks |
| 130 | scan | `scan --plant 12345 --start 1 --count 20000 --per 2` | rc=0 | 0 | PASS | per>1 + key 1 (regression) |
| 131 | scan | `scan --plant 1 --start 0 --count 2` | rc=0 | 0 | PASS | start 0 = infinity (regression) |
| 132 | scan | `scan --plant 12345 --start 0x2ee0 --count 1000` | rc=0 | 0 | PASS |  |
| 133 | scan | `scan --target 87f0201511180e729bed2d80da1880c3901f1199 --start 12000 --count 1000` | rc=0 | 0 | PASS |  |
| 134 | scan | `scan --target 1BDyrQ6WoF8VN3g9SAS1iKZcPzFfnDVieY --start 1 --count 60000` | rc=0 | 0 | PASS |  |
| 135 | scan | `scan --plant 12345 --start 12000 --count 1000 --per 64 --blocks 4` | rc=0 | 0 | PASS |  |
| 136 | scan | `scan --plant 12345 --start 12000 --count 1000 --per 0` | rc=0 | 0 | PASS |  |
| 137 | scan | `scan --plant 12345 --start 12000 --count 1000 --blocks 1 --per 1000` | rc=0 | 0 | PASS |  |
| 138 | scan | `scan --plant 12345 --start 12000` | rc=0 | 0 | PASS | default count |
| 139 | scan | `scan --plant 1180591620717411303424 --count 1000000` | rc=0 | 0 | PASS | default start 2^70 |
| 140 | scan | `scan --plant 12345 --start 12000 --count 1000 --target 87f0201511180e729bed2d80da1880c3901f1199` | rc=0 | 0 | PASS |  |
| 141 | scan | `scan --plant 12345 --start 12000 --count 0` | rc!=0 | 1 | PASS |  |
| 142 | scan | `scan` | rc!=0 | 1 | PASS |  |
| 143 | scan | `scan --target xyz --start 0 --count 10` | rc!=0 | 1 | PASS |  |
| 144 | scan | `scan --plant xyz --start 0 --count 10` | rc!=0 | 1 | PASS |  |
| 145 | scan | `scan --start xyz --count 10 --plant 1` | rc!=0 | 1 | PASS |  |
| 146 | scan | `scan --blocks 0 --plant 12345 --start 12000 --count 1000` | rc!=0 | 1 | PASS |  |
| 147 | scan | `scan --blocks -1 --plant 12345 --start 12000 --count 1000` | rc!=0 | 1 | PASS |  |
| 148 | scan | `scan --plant 12345 --start 12000 --count 1000 --per` | rc!=0 | 1 | PASS |  |
| 149 | scan | `scan --plant 12345 --start 12000 --count 1000 --target` | rc!=0 | 1 | PASS |  |
| 150 | scan | `scan --plant 12345 --start 12000 --count 1000 --blocks` | rc!=0 | 1 | PASS |  |
| 151 | check | `check: solved-puzzle test hits are not written to found.txt` | rc=0 | 0 | PASS | found.txt must stay clean |

## Result

All cases passed.

## Notes / observed behaviour

- `--seconds` limits the default snake, `--spread`, `--freeze` and `--freeze --vis`; it is ignored by `--exhaust` and `--ladder` (they run their unit of work to the end).
- `--load` affects the default snake, `--spread`, `--freeze` and `--exhaust` duty cycles; `--ladder` ignores it.
- `--exhaust --chunkfrom` only applies when `kmin == kmax` (one slice); a resumed run now reports the keys checked in this run and marks the slice as resumed.
- `--exhaust` supports puzzles 4..128 and `--freeze` 3..128: the top bit is kept outside the 128-bit mask, so puzzle 129 cannot be represented.
- `--freeze --check` caps the frozen count to the known key's popcount; test hits on solved puzzles are printed but not appended to `found.txt`.
- `scan_range` accepts a Base58 address or a 40-hex hash160 in `--target`, and returns rc 1 when the requested range contains no match (including `--count 0`).
- `--alert-test` returns 0 even though the repo intentionally has no `alert.py`; it prints `alert.py failed`.
- `--show 0` and negative values fall through to the normal snake search.
- `--snake --test` runs the self-tests (test mode wins over the default search).
- Known-puzzle test hits (`--freeze --check`, `--spread --watch`, `--snake --watch`) do not pollute `found.txt`; the clone check confirmed it.

## Bugs found and fixed in this tree

1. `scan_range --blocks 0` died with SIGFPE (division by zero); `--blocks -1` reached CUDA and failed with `invalid configuration argument`. Both now return a clear rc 1 before CUDA.
2. `--exhaust --chunkfrom` reported the whole slice as checked even when it started in the middle (e.g. `--chunkfrom 4` with 4 chunks reported `27,132 / 27,132`). It now reports the keys actually checked, the skipped prefix, and `nothing to do` when the start is past the end.
3. `--exhaust` advertised 3..129 but the GPU unranking table was sized for n<=70: puzzle 72 was undefined behaviour and 129 could not encode its top bit. The table now covers n<=128, the small-nb self-test is fixed, and the valid range is 4..128 (freeze/vis: 3..128).
4. `--freeze --vis --frz` larger than the low-bit count was silently clamped on solved puzzles; it now rejects it like the search mode does.
5. `--exhaust` and `--freeze` appended solved-puzzle test hits to `found.txt`; they no longer do that (real finds still go there).
6. `scan_range` missed a target when the range started at a power of two and the per-thread base offset had that bit set (e.g. `--plant 51510 --start 1 --count 60000` returned 0 hits). `madd()` assumes distinct points, so `P0 + P0` divided by zero. A scanner-specific `madd_scan()` now doubles on equality and handles infinity; verified with `start=1` and `start=0`.
7. `tools/gen_scan_cu.py` sliced `snake.cu` by hardcoded line numbers. Adding four lines to `fe_eq` silently produced a broken `scan_range.cu` (`__constant__` inside a host function). The generator now uses `GEN_SCAN_*` marker comments; the generated scanner is built and exercised in this matrix.
