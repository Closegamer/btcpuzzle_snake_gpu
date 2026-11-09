#!/bin/bash
# Build and run the snake. With no arguments this starts the main snake walk (puzzles 71..160).
# Examples:
#   bash cuda/run.sh                  # snake walk (default)
#   bash cuda/run.sh --help           # all modes and options
#   bash cuda/run.sh --test           # self-tests
#   bash cuda/run.sh --spread 71 --load 90
set -u
cd "$(dirname "$0")" || exit 1

# ---- find a CUDA toolkit (needed only for the first build) ----
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

# ---- build on first run or after the sources change ----
if [ ! -x ./snake ] || [ snake.cu -nt ./snake ] || [ rmd160_steps.h -nt ./snake ]; then
    echo "=== snake is not built (or is older than the sources): building ..."
    bash ./build.sh || exit 1
fi

# ---- run ----
# The terminal gets the raw stream (the live status line is redrawn in place);
# the log gets one newline-terminated line per update instead of \r-joined chunks.
if [ -w /dev/tty ]; then
    stdbuf -oL ./snake "$@" 2>&1 | tee /dev/tty | stdbuf -i0 -o0 tr '\r' '\n' >> snake.log
else
    stdbuf -oL ./snake "$@" 2>&1 | stdbuf -i0 -o0 tr '\r' '\n' >> snake.log
fi
exit "${PIPESTATUS[0]}"
