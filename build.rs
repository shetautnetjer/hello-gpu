// build.rs
use std::{
    env,
    ffi::OsString,
    fs,
    path::{Path, PathBuf},
    process::Command,
};

/// Detects the GPU’s compute capability via `nvidia-smi`, returns e.g. "compute_89".
fn detect_arch() -> String {
    let sm = Command::new("nvidia-smi")
        .args([
            "--query-gpu=compute_cap",
            "--format=csv,noheader",
        ])
        .output()
        .ok()
        .and_then(|o| {
            if o.status.success() {
                String::from_utf8(o.stdout).ok()
            } else {
                None
            }
        })
        .unwrap_or_default();

    match sm.trim() {
        v if v.starts_with("8.9") => "compute_89",
        v if v.starts_with("8.6") => "compute_86",
        _                         => "compute_80",   // safe default
    }
    .into()
}

/// Compile one .cu file to PTX and return the PTX path.
fn compile_cu(src: &Path, out_dir: &Path, arch: &str) -> PathBuf {
    let stem = src.file_stem().unwrap().to_string_lossy();
    let dst  = out_dir.join(format!("{stem}.ptx"));

    let status = Command::new("nvcc")
        .args([
            src.to_str().unwrap(),
            "-ptx",
            "-arch", arch,
            "-o",   dst.to_str().unwrap(),
        ])
        .status()
        .expect("failed to spawn nvcc");

    if !status.success() {
        panic!("nvcc failed on {:?} with exit code {:?}", src, status.code());
    }
    dst
}

fn main() {
    // --------------------------------- build config --------------------
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR not set"));
    let arch: String = env::var("CUDA_ARCH").unwrap_or_else(|_| detect_arch());

    // --------------------------------- gather kernels -----------------
    let kernel_dir = Path::new("kernels");
    let cu_files: Vec<PathBuf> = fs::read_dir(kernel_dir)
        .expect("kernels/ dir not found")
        .filter_map(|entry| {
            let path = entry.ok()?.path();
            (path.extension()? == "cu").then_some(path)
        })
        .collect();

    if cu_files.is_empty() {
        panic!("No .cu files found in kernels/");
    }

    // --------------------------------- compile ------------------------
    for cu in &cu_files {
        println!("cargo:rerun-if-changed={}", cu.display());
        let ptx_path = compile_cu(cu, &out_dir, &arch);

        // Export env var like KERNEL_VEC_ADD_PTX
        let var_name = format!(
            "KERNEL_{}_PTX",
            cu.file_stem().unwrap().to_string_lossy().to_uppercase()
        );
        println!("cargo:rustc-env={var_name}={}", ptx_path.display());
    }

    // Optionally expose the chosen arch for downstream crates
    println!("cargo:rustc-env=CUDA_ARCH_USED={arch}");
}