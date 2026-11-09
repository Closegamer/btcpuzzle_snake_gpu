#!/bin/bash
# Quick, non-destructive smoke test: builds both binaries and exercises every mode
# without writing search state (no exhaust_results.txt / freeze_masks / found.txt).
# The exhaustive 151-case matrix is tools/mode_matrix.py.
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd) || exit 1
cd "$ROOT" || exit 1

echo "=== building snake and scan_range"
bash cuda/build.sh || { echo "snake build FAILED"; exit 1; }
bash cuda/build_scan.sh >/tmp/btcpuzzle_scan_build.log 2>&1 || {
    echo "scan_range build FAILED"; tail -20 /tmp/btcpuzzle_scan_build.log; exit 1;
}

CUDA_HOME=${CUDA_HOME:-}
if [ -z "$CUDA_HOME" ] || [ ! -x "$CUDA_HOME/bin/nvcc" ]; then
    for d in "$HOME/cudaroot/usr/local/cuda-12.8" "$HOME/cudaroot/usr/local/cuda" \
             /usr/local/cuda-12.8 /usr/local/cuda /opt/cuda; do
        if [ -x "$d/bin/nvcc" ]; then CUDA_HOME="$d"; break; fi
    done
fi
if [ -z "$CUDA_HOME" ] || [ ! -x "$CUDA_HOME/bin/nvcc" ]; then
    echo "CUDA toolkit not found"; exit 1
fi
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

cd "$ROOT/cuda" || exit 1
OUT=/tmp/btcpuzzle_smoke.out
pass=0
fail=0

ok() {
    desc=$1; shift
    if "$@" >"$OUT" 2>&1; then
        echo "PASS  $desc"; pass=$((pass + 1))
    else
        echo "FAIL  $desc"; tail -8 "$OUT"; fail=$((fail + 1))
    fi
}
bad() {
    desc=$1; shift
    if "$@" >"$OUT" 2>&1; then
        echo "FAIL  $desc (expected a non-zero exit)"; fail=$((fail + 1))
    else
        echo "PASS  $desc"; pass=$((pass + 1))
    fi
}

ok "help"                     ./snake --help
ok "self-test"                ./snake --test
ok "show one snake step"      ./snake --show 1
ok "default snake, 2 s"       timeout 60 ./snake --snake --seconds 2 --nocool
ok "spread 16, 1 s"           ./snake --spread 16 --kmin 2 --kmax 2 --seconds 1
ok "ladder 1..3"              ./snake --ladder --from 1 --to 3
ok "scan_range planted key"   ./scan_range --plant 12345 --start 12000 --count 1000
ok "scan_range power-of-two start" ./scan_range --plant 51510 --start 1 --count 60000
bad "scan_range rejects --blocks 0" ./scan_range --blocks 0 --plant 12345 --start 12000 --count 1000

rm -f "$OUT"
echo
echo "=== smoke test: $pass passed, $fail failed"
[ "$fail" -eq 0 ]
