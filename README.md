Hello-GPU  —  Self-Diagnosing Rust + CUDA Starter
A zero-config template that discovers your NVIDIA GPU, verifies the CUDA
tool-chain, and runs a sanity-check kernel — all from pure Rust.

✨ What this repo does
Step	Action	Why it matters
1. Detect	Queries the driver for GPU name, PCI-ID, SM major / minor, total VRAM, driver & runtime versions, PTX ISA level.	Confirms exactly which features (FP-8, Tensor Cores, etc.) you can target.
2. Compile	Uses nvcc at build-time to turn a sample CUDA file into PTX.<br/>• Defaults to generic PTX 8.0 (compute_80).<br/>• Optionally obeys CUDA_ARCH=sm_89 (or any sm_xy) so you get hardware-specific ops.	Guarantees kernel code can JIT on this machine and avoids runtime compile hiccups.
3. Execute	Allocates 1 Mi elements on both host & device, launches a vec_add kernel, copies back, and validates result.	Proves host ↔ device memory paths, kernel launch, PTX JIT, and stream sync all work.
4. Report	Prints a JSON blob with every key detail plus kernel_ok : true/false.	One-shot diagnostic you can paste into bug reports or CI logs.

🗂️ Repo layout
csharp
Copy
hello-gpu/
├─ Cargo.toml          # deps: cust (CUDA bindings), nvml (GPU facts), serde
├─ build.rs            # compiles kernels/vec_add.cu → OUT_DIR/vec_add.ptx
├─ kernels/
│  └─ vec_add.cu       # 15-line demo kernel
└─ src/
   └─ main.rs          # runtime detection + kernel launch + JSON report
🔧 Prerequisites
Component	Min version	Install (PowerShell)
CUDA Toolkit	11.8 (12.x fine)	winget install -e Nvidia.CUDA --version 12.5
NVIDIA driver	Same major as toolkit (≥ 12.x if using toolkit 12)	GeForce or Studio driver; 572.xx works
MSVC Build Tools	2022, x64 workload	winget install -e Microsoft.VisualStudio.2022.BuildTools<br/>→ tick Desktop C++
LLVM / libclang	17+ (needed by bindgen)	winget install -e LLVM.LLVM<br/>setx LIBCLANG_PATH "C:\Program Files\LLVM\bin"
Rust	stable 1.87+	`irm https://sh.rustup.rs

Always build from the Developer PowerShell for VS 2022 (x64) so MSVC linker & Windows SDK are on PATH.

🚀 Quick-start
powershell
Copy
git clone https://github.com/your-org/hello-gpu.git
cd hello-gpu

# (optional) compile for specific arch e.g. Ada Lovelace
$Env:CUDA_ARCH = "sm_89"

cargo run --release
Sample output:

json
Copy
{
  "name": "NVIDIA GeForce RTX 4090",
  "pcie_bus_id": "00000000:01:00.0",
  "sm_major_minor": [8, 9],
  "total_mem_mb": 24564,
  "driver_version": "12.8",
  "runtime_version": [12, 5],
  "ptx_version": [8, 0],
  "kernel_ok": true
}
🛠️ Extending the project
Add a new kernel

Drop .cu into kernels/.

List it in build.rs or add a loop over kernels/*.cu.

Module::from_ptx() + get_function() → launch.

Benchmark

Add criterion = "0.5" to Cargo.toml.

Write benches/gpu_bench.rs to time your kernels per batch.

Multi-GPU

rust
Copy
for i in 0..Device::num_devices()? {
    let dev = Device::get_device(i)?;
    /* create ctx, run kernel, print per-GPU report */
}
🧩 Troubleshooting
Symptom	Fix
Unable to find libclang	Install LLVM or set proper LIBCLANG_PATH.
nvcc fatal : value 'sm_8' is not defined	Make sure $Env:CUDA_ARCH is a full sm_xy like sm_89, or unset it.
link.exe cannot open cudart_static.lib	Open Developer PowerShell (x64) or add %CUDA_PATH%\lib\x64 to LIB.
JSON shows "kernel_ok": false	Driver/JIT mismatch or launch params wrong. Validate PTX arch & grid size.
