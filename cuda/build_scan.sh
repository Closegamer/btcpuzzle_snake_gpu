#!/bin/bash
# Build the arbitrary-range scanner (PTX compute_90). Generates scan_range.cu first.
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

python3 ../tools/gen_scan_cu.py
"$CUDA_HOME/bin/nvcc" -O3 -std=c++17 -gencode arch=compute_90,code=compute_90 \
    -o scan_range.tmp scan_range.cu $NVCC_EXTRA
mv -f scan_range.tmp scan_range
echo "scan_range built: ./scan_range --target ADDR --start KEY --count N"
