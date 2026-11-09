# Testing

Requirements: Linux or WSL, an NVIDIA GPU, CUDA 12.x, `bash`, `python3`. The build scripts
auto-detect the toolkit in `CUDA_HOME`, `~/cudaroot/usr/local/cuda-12.8`, `/usr/local/cuda-12.8`,
`/usr/local/cuda` or `/opt/cuda`.

## Quick smoke test

```bash
bash tools/smoke_test.sh
```

It builds `cuda/snake` and `cuda/scan_range` and runs `--help`, `--test`, `--show`, a 2-second
default snake, `--spread`, `--ladder`, two `scan_range` cases and one expected error. It is
non-destructive: it does not create `exhaust_results.txt`, `freeze_masks_p*.txt` or `found.txt`.

## Full mode matrix

```bash
python3 tools/mode_matrix.py
```

This clones the committed tree into a temporary directory, builds both binaries there, and runs
**151 cases** covering every mode and every flag with valid, boundary, invalid and missing values.
It writes `docs/MODE_MATRIX.md` and exits non-zero if any case fails. Runtime is about two minutes
on the reference laptop. Runtime files stay in the temporary clone.

Covered: default snake / `--snake`, `--spread`, `--exhaust`, `--freeze` (including `--vis`),
`--ladder`, `--show`, `--watch`, `--test`, `--alert-test`; `--kmin`, `--kmax`, `--frz`, `--check`,
`--force`, `--chunkfrom`, `--from`, `--to`, `--ref`, `--ones`, `--load`, `--seconds`, `--nocool`,
`--addr`; and `scan_range` with `--start`, `--count`, `--per`, `--blocks`, `--target`, `--plant`.

Verified on 2026-11-09: **151/151 passed** on a clean clone. The report is
`docs/MODE_MATRIX.md`.

## Known behaviour

- `--seconds` limits the default snake, `--spread`, `--freeze` and `--freeze --vis`; `--exhaust`
  and `--ladder` ignore it and finish their unit of work.
- `--load` affects the default snake, `--spread`, `--freeze` and `--exhaust`; `--ladder` ignores it.
- `--exhaust` supports puzzles 4..128 and `--freeze` 3..128. The top bit is kept outside the
  128-bit mask, so puzzle 129 cannot be represented.
- `--exhaust --chunkfrom` applies only when `kmin == kmax`; a resumed run reports the keys checked
  in that run and the skipped prefix.
- `scan_range` returns exit code 1 when the requested range has no match (including `--count 0`).
- `--alert-test` returns 0 even without `alert.py`; it prints `alert.py failed`.
- `--show 0` and negative values fall through to the normal snake search.
