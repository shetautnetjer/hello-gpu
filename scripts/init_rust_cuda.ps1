# ── 1. Environment Check ────────────────────────────────────────────────
Write-Host "`nChecking CUDA environment..." -ForegroundColor Cyan

# Verify required environment variables
$requiredVars = @("CCBIN_PATH", "CUDA_PATH", "CUDA_TOOLKIT_ROOT_DIR")
$missingVars = $requiredVars | Where-Object { -not (Get-Item "env:$_" -ErrorAction SilentlyContinue) }

if ($missingVars) {
    Write-Error "Missing required environment variables: $($missingVars -join ', ')"
    Write-Host "Run init_gpu_env.ps1 first to set up the environment"
    exit 1
}

# ── 2. Create/Update .cargo/config.toml ────────────────────────────────────────
Write-Host "`nSetting up Rust-CUDA configuration..." -ForegroundColor Cyan

$cargoConfigDir = ".cargo"
$cargoConfigPath = Join-Path $cargoConfigDir "config.toml"

# Create .cargo directory if it doesn't exist
if (-not (Test-Path $cargoConfigDir)) {
    New-Item -ItemType Directory -Path $cargoConfigDir | Out-Null
}

# Create or update config.toml
$configContent = @"
[target.'cfg(windows)']
rustflags = [
    "-C", "link-arg=/LIBPATH:$env:CUDA_LIBRARY_PATH",
    "-C", "link-arg=cudart.lib",
    "-C", "link-arg=cuda.lib",
    "-C", "link-arg=cudadevrt.lib",
]

[env]
CUDA_PATH = "$env:CUDA_PATH"
CUDA_TOOLKIT_ROOT_DIR = "$env:CUDA_TOOLKIT_ROOT_DIR"
CUDA_LIBRARY_PATH = "$env:CUDA_LIBRARY_PATH"
CCBIN_PATH = "$env:CCBIN_PATH"
"@

Set-Content -Path $cargoConfigPath -Value $configContent
Write-Host "Created/updated $cargoConfigPath"

# ── 3. Create/Update build.rs template ────────────────────────────────────────
Write-Host "`nCreating build.rs template..." -ForegroundColor Cyan

$buildRsContent = @'
use std::{
    env,
    fs,
    path::{Path, PathBuf},
    process::Command,
};

fn main() {
    if env::var_os("SKIP_CUDA").is_some() {
        println!("cargo:warning=SKIP_CUDA=1 - skipping CUDA kernels");
        return;
    }

    if Command::new("nvcc").arg("--version").output().is_err() {
        println!("cargo:warning=nvcc not found - skipping CUDA");
        return;
    }

    let (cl_path, sdk_version) = match find_windows_toolchain() {
        Ok(t) => t,
        Err(e) => {
            println!("cargo:warning={}", e);
            return;
        }
    };

    let out_dir = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let arch = env::var("CUDA_ARCH").unwrap_or_else(|_| "compute_89".to_string()); // Ada default

    for entry in fs::read_dir("kernels").unwrap() {
        let cu = entry.unwrap().path();
        if cu.extension().and_then(|s| s.to_str()) != Some("cu") {
            continue;
        }

        let ptx = out_dir.join(cu.file_stem().unwrap()).with_extension("ptx");

        let mut cmd = Command::new("nvcc");
        cmd.args([
            cu.to_str().unwrap(),
            "-ptx",
            "-arch", &arch,
            "--compiler-bindir", cl_path.to_str().unwrap(),
            "-o", ptx.to_str().unwrap(),
        ])
        .env("WindowsSDKVersion", &sdk_version);

        let status = cmd.status().expect("failed to spawn nvcc");
        if !status.success() {
            panic!("nvcc failed on {:?} (exit {})", cu, status);
        }

        let var = format!(
            "KERNEL_{}_PTX",
            cu.file_stem().unwrap().to_string_lossy().to_uppercase()
        );
        println!("cargo:rustc-env={}={}", var, ptx.display());
        println!("cargo:rerun-if-changed={}", cu.display());
    }
}

fn find_windows_toolchain() -> Result<(PathBuf, String), String> {
    let vswhere = PathBuf::from(env::var("ProgramFiles(x86)").unwrap())
        .join("Microsoft Visual Studio")
        .join("Installer")
        .join("vswhere.exe");

    let output = Command::new(vswhere)
        .args([
            "-products", "*",
            "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
            "-property", "installationPath",
            "-nologo",
        ])
        .output()
        .map_err(|e| format!("vswhere failed: {}", e))?;

    if !output.status.success() {
        return Err("vswhere failed".to_string());
    }

    let vs_path = String::from_utf8_lossy(&output.stdout).trim().to_string();
    let cl_path = PathBuf::from(&vs_path)
        .join("VC")
        .join("Tools")
        .join("MSVC")
        .read_dir()
        .map_err(|e| format!("failed to read MSVC dir: {}", e))?
        .filter_map(|entry| entry.ok())
        .max_by_key(|entry| entry.file_name())
        .ok_or_else(|| "no MSVC version found".to_string())?
        .path()
        .join("bin")
        .join("Hostx64")
        .join("x64")
        .join("cl.exe");

    if !cl_path.exists() {
        return Err("cl.exe not found".to_string());
    }

    let sdk_version = env::var("WindowsSDKVersion")
        .unwrap_or_else(|_| "10.0.26100.0".to_string());

    Ok((cl_path, sdk_version))
}
'@

Set-Content -Path "build.rs" -Value $buildRsContent
Write-Host "Created build.rs template"

# ── 4. Create kernels directory ────────────────────────────────────────────────
Write-Host "`nCreating kernels directory..." -ForegroundColor Cyan

if (-not (Test-Path "kernels")) {
    New-Item -ItemType Directory -Path "kernels" | Out-Null
    Write-Host "Created kernels directory"
}

# ── 5. Create example kernel ────────────────────────────────────────────────
Write-Host "`nCreating example kernel..." -ForegroundColor Cyan

$kernelContent = @'
#include <cuda_runtime.h>

extern "C" __global__ void vec_add(const float* A, const float* B, float* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) C[i] = A[i] + B[i];
}
'@

Set-Content -Path "kernels/vec_add.cu" -Value $kernelContent
Write-Host "Created example kernel at kernels/vec_add.cu"

Write-Host "`nRust-CUDA setup completed successfully" -ForegroundColor Green
Write-Host "`nNext steps:"
Write-Host "1. Add CUDA dependencies to your Cargo.toml:"
Write-Host "   [dependencies]"
Write-Host "   cust = { git = 'https://github.com/Rust-GPU/Rust-CUDA', branch = 'main' }"
Write-Host "   nvml-wrapper = '0.11'"
Write-Host "2. Run 'cargo build' to compile your project"
Write-Host "3. Check the generated PTX files in target/debug/build/<package>/out/" 