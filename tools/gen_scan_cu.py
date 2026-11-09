#!/usr/bin/env python3
"""Build cuda/scan_range.cu from snake.cu primitives + the range-scanner body.

The two source regions are delimited by marker comments in snake.cu, so editing
snake.cu no longer silently corrupts the generated file (the old line-number
slices broke as soon as a line was added).

    python3 tools/gen_scan_cu.py && bash cuda/build_scan.sh
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SNAKE = ROOT / "cuda" / "snake.cu"
CORE_END = "// ===== GEN_SCAN_CORE_END ====="
HOST_BEGIN = "// ===== GEN_SCAN_HOST_BEGIN ====="
HOST_END = "// ===== GEN_SCAN_HOST_END ====="

text = SNAKE.read_text(encoding="utf-8")
for marker in (CORE_END, HOST_BEGIN, HOST_END):
    if text.count(marker) != 1:
        raise SystemExit(f"{marker!r} must occur exactly once in {SNAKE}")

part1 = text[:text.index(CORE_END)]                    # field, SHA-256/RIPEMD-160, hash160_pub
host_start = text.index(HOST_BEGIN) + len(HOST_BEGIN)
part2 = text[host_start:text.index(HOST_END, host_start)]  # host helpers

body = (ROOT / "tools" / "scan_range_body.cu").read_text(encoding="utf-8")

head = ("// scan_range.cu — GENERATED: snake.cu primitives + the arbitrary-range scanner body.\n"
        "// Rebuild: python3 tools/gen_scan_cu.py && bash cuda/build_scan.sh\n"
        "#include <fstream>\n#include <vector>\n#include <string>\n#include <algorithm>\n"
        "#include <chrono>\n#include <cstdlib>\n\n")

out = ROOT / "cuda" / "scan_range.cu"
out.write_text(head + part1 + "\n// ---- host helpers ----\n" + part2 + "\n" + body,
               encoding="utf-8")
print("wrote", out, len(out.read_text(encoding="utf-8").splitlines()), "lines")
