#!/usr/bin/env bash
set -e

# check GPU presence
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found. Install NVIDIA drivers from https://www.nvidia.com/Download/index.aspx" >&2
  exit 1
fi
GPU_INFO=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -n 1)
if [ -z "$GPU_INFO" ]; then
  echo "No CUDA-capable GPU detected. Install drivers and ensure hardware is present." >&2
  exit 1
fi
echo "GPU & Driver: $GPU_INFO"

# check CUDA toolkit
if ! command -v nvcc >/dev/null 2>&1; then
  echo "nvcc not found. Install the CUDA Toolkit from https://developer.nvidia.com/cuda-downloads" >&2
  exit 1
fi
nvcc --version | head -n 1

# try compiling the sample kernel
TMPPTX=$(mktemp /tmp/vec_add.XXXXXX.ptx)
if ! nvcc kernels/vec_add.cu -ptx -o "$TMPPTX"; then
  echo "Failed to compile CUDA sample. Ensure CUDA Toolkit matches your driver version." >&2
  rm -f "$TMPPTX"
  exit 1
fi
rm -f "$TMPPTX"
echo "Kernel compilation succeeded."

# run the Rust sample
if ! cargo run --release >/tmp/hello_gpu_output.txt 2>&1; then
  cat /tmp/hello_gpu_output.txt
  echo "Rust CUDA sample failed. Check NVML installation and toolkit version." >&2
  exit 1
fi
cat /tmp/hello_gpu_output.txt

echo "All checks passed."
