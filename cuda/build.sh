#!/bin/bash
# Build snake in Linux/WSL. PTX compute_90 is deliberate: native sm_120 produced wrong results.
set -e
cd "$(dirname "$0")"

# ---- find a CUDA toolkit ----
if [ -z "${CUDA_HOME:-}" ] || [ ! -x "$CUDA_HOME/bin/nvcc" ]; then
    for d in "$HOME/cudaroot/usr/local/cuda-12.8" "$HOME/cudaroot/usr/local/cuda" \
             /usr/local/cuda-12.8 /usr/local/cuda /opt/cuda; do
        if [ -x "$d/bin/nvcc" ]; then CUDA_HOME="$d"; break; fi
    done
fi
if [ -z "${CUDA_HOME:-}" ] || [ ! -x "$CUDA_HOME/bin/nvcc" ]; then
    echo "CUDA toolkit not found." >&2
    echo "Install CUDA 12.x, or set CUDA_HOME, e.g.: export CUDA_HOME=/usr/local/cuda-12.8" >&2
    exit 1
fi
export CUDA_HOME
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo "=== CUDA: $CUDA_HOME"
rm -f snake.tmp
if ! "$CUDA_HOME/bin/nvcc" -O3 -std=c++17 -gencode arch=compute_90,code=compute_90 \
        -Xptxas -v -o snake.tmp snake.cu > build.log 2>&1; then
    echo "=== build FAILED:" >&2
    grep -E "error|Error" build.log | head -20 >&2 || tail -20 build.log >&2
    exit 1
fi
[ -s snake.tmp ] || { echo "=== build FAILED: nvcc produced no binary" >&2; exit 1; }
grep -E "registers|spill|warning" build.log || true
mv -f snake.tmp snake
rm -f build.log
echo "=== built: $(pwd)/snake"
