// build.rs  ──────────────────────────────────────────────────────────────
// Windows-only helper for hello-gpu.
//  * Finds nvcc, cl.exe (v143), and a CUDA-compatible Windows SDK ≤ 22621
//  * Defaults to compute_89 PTX (Ada) unless CUDA_ARCH is set
//  * Exposes each compiled PTX via env!("KERNEL_<NAME>_PTX")

use std::{
    env,
    fs,
    path::{Path, PathBuf},
    process::Command,
};

fn main() {
    // ── 0. Early exits ───────────────────────────────────────────────
    if env::var_os("SKIP_CUDA").is_some() {
        println!("cargo:warning=SKIP_CUDA=1  →  building without CUDA kernels");
        return;
    }
    if Command::new("nvcc").arg("--version").output().is_err() {
        println!("cargo:warning=nvcc not found  →  set SKIP_CUDA=1 to build CPU-only");
        return;
    }

    // ── 1. Locate MSVC + SDK ─────────────────────────────────────────
    let (cl_path, sdk_version) = match find_windows_toolchain() {
        Ok(t)  => t,
        Err(e) => { println!("cargo:warning={e}  →  building without CUDA"); return; }
    };

    // ── 2. Compile every .cu file to PTX ─────────────────────────────
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let arch = env::var("CUDA_ARCH").unwrap_or_else(|_| "compute_89".into()); // default = Ada

    for entry in fs::read_dir("kernels").expect("kernels dir missing") {
        let cu = entry.expect("dir entry").path();
        if cu.extension().and_then(|s| s.to_str()) != Some("cu") { continue; }

        let ptx = out_dir.join(cu.file_stem().unwrap()).with_extension("ptx");

        // nvcc command
        let mut cmd = Command::new("nvcc");
        cmd.args([
            cu.to_str().unwrap(),
            "-ptx",
            "-arch", &arch,
            "--compiler-bindir", cl_path.to_str().unwrap(),
            "-o", ptx.to_str().unwrap(),
        ])
        // pin SDK for *this* cl.exe invocation
        .env("WindowsSDKVersion", &sdk_version);

        let status = cmd.status().expect("failed to spawn nvcc");
        if !status.success() {
            panic!("nvcc failed on {:?} (exit {})", cu, status);
        }

        // make PTX path visible to Rust (`env!("KERNEL_VEC_ADD_PTX")`)
        let var = format!(
            "KERNEL_{}_PTX",
            cu.file_stem().unwrap().to_string_lossy().to_uppercase()
        );
        println!("cargo:rustc-env={var}={}", ptx.display());
        println!("cargo:rerun-if-changed={}", cu.display());
    }
}

/// Find newest MSVC v143 cl.exe and newest Windows SDK ≤ 22621.
/// Returns (cl.exe path, "10.0.22621.0").
#[cfg(target_os = "windows")]
fn find_windows_toolchain() -> Result<(PathBuf, String), String> {
    // vswhere is installed with every VS / BuildTools bundle
    let vswhere = r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe";
    let vs_root = String::from_utf8(
        Command::new(vswhere)
            .args(["-latest", "-products", "*", "-property", "installationPath"])
            .output()
            .map_err(|_| "vswhere.exe not found; install \"Visual Studio 2022 Build Tools\"")?
            .stdout,
    )
    .unwrap();
    if vs_root.trim().is_empty() {
        return Err("MSVC tool-set not installed".into());
    }

    // newest v143 cl.exe
    let cl = Path::new(vs_root.trim())
        .join(r"VC\Tools\MSVC")
        .read_dir()
        .map_err(|_| "cannot list MSVC versions")?
        .filter_map(Result::ok)
        .map(|p| p.path())
        .filter(|p| p.join("bin\\Hostx64\\x64\\cl.exe").exists())
        .max()
        .ok_or("Hostx64\\x64\\cl.exe (v143) not found")?
        .join(r"bin\Hostx64\x64\cl.exe");

    // newest Windows SDK version ≤ 10.0.22621
    let kits = Path::new(r"C:\Program Files (x86)\Windows Kits\10\Include");
    let sdk = kits.read_dir()
        .map_err(|_| "Windows SDK not installed")?
        .filter_map(Result::ok)
        .filter_map(|e| e.file_name().into_string().ok())
        .filter(|v| v.starts_with("10.0.") && version_leq(v, "10.0.22621.0"))
        .max()
        .ok_or("No Windows SDK ≤ 10.0.22621 found (CUDA 12.5 max)")?;

    Ok((cl, sdk))
}

/// Compare dotted version strings numerically, e.g. "10.0.22000.0" ≤ "10.0.22621.0".
fn version_leq(a: &str, b: &str) -> bool {
    let mut ai = a.split('.').map(|x| x.parse::<u32>().unwrap_or(0));
    let mut bi = b.split('.').map(|x| x.parse::<u32>().unwrap_or(0));
    loop {
        match (ai.next(), bi.next()) {
            (None, None)       => return true,
            (Some(x), Some(y)) => if x < y { return true } else if x > y { return false },
            (Some(x), None)    => return x == 0,
            (None, Some(y))    => return y >= 0,
        }
    }
}