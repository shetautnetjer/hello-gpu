# agents.md — GPU Environment Codex

Version: 1.0

---

## ⚙️ Execution Context (Containerized Agent)

- **Environment:** Docker / CI container
- **GPU Present:** ❌ NO (use simulation/fallback)
- **nvcc Available:** ❌ NO
- **CUDA Libraries:** ❌ Missing
- **clang Available:** ❌ NO
- **Rust Toolchain:** ✅ YES (can build Rust code)

---

## 📦 Project Configuration

- **CUDA Kernels:** `kernels/vec_add.cu`
- **PTX Target Output:** `$OUT_DIR/vec_add.ptx`
- **Build System:** `build.rs` calls `nvcc -ptx`
- **Runtime Launcher:** `cust` crate via PTX loader
- **Kernel Entry Symbol:** `vec_add`
- **Expected Arch:** `compute_89`
- **Environment Vars Used:**
  - `CUDA_ARCH`
  - `KERNEL_PTX`

---

## 🔧 Simulation & Analysis Instructions

This agent is authorized to:
- Parse `build.rs` for correct nvcc args and fallback behavior
- Validate kernel source structure and PTX expectations
- Simulate PTX kernel metadata
- Simulate output from `GpuReport` when no GPU is present
- Diagnose why CUDA build will fail, and suggest workarounds

If kernel or runtime errors are found:
- Output a JSON file with diagnostics (`target/gpu_env_report.json`)
- Summarize missing paths, expected fix, and kernel status
- Simulate a successful output when properly configured

---

## 🧪 JSON Report Format

The agent may generate the following schema (stored at `target/gpu_env_report.json`):

```json
{
  "agent_version": "v1.0",
  "can_compile_cuda": false,
  "can_run_gpu_kernels": false,
  "missing_tools": ["nvcc", "nvidia-smi", "libcuda.so", "clang"],
  "cuda_arch_expected": "compute_89",
  "kernel": {
    "name": "vec_add",
    "source": "kernels/vec_add.cu",
    "ptx_target": "target/vec_add.ptx",
    "symbol": "vec_add",
    "arg_signature": ["*mut f32", "*const f32", "usize"]
  },
  "build_rs_flags": {
    "arch": "compute_89",
    "output_dir": "OUT_DIR",
    "ptx_path_env": "KERNEL_PTX"
  },
  "fallback_mode": "simulate",
  "simulated_report": {
    "device": "Simulated NVIDIA RTX 4090",
    "sm": [8, 9],
    "driver_version": "535.104",
    "ptx_version": [8, 0],
    "kernel_ok": true
  },
  "recommendations": [
    "Skip nvcc in containers where /usr/local/cuda is missing",
    "Precompile PTX in host dev machine and ship it with container",
    "Use cust's get_function() to validate PTX symbol presence at runtime"
  ]
}

## 📜 Generating a Simulated Report

Run the helper script `scripts/gen_gpu_report.rs` to create `target/gpu_env_report.json` with simulated results when no GPU is present.

```bash
cargo run --bin gen_gpu_report
# or
rustc scripts/gen_gpu_report.rs && ./gen_gpu_report
```

