// build.rs
use std::{env, process::Command};

fn main() {
    println!("cargo:rerun-if-changed=kernels/vec_add.cu");

    let out_dir = env::var("OUT_DIR").unwrap();
    let arch    = env::var("CUDA_ARCH").unwrap_or_else(|_| "compute_80".into()); // generic PTX 8.0

    let status = Command::new("nvcc")
        .args([
            "kernels/vec_add.cu",
            "-ptx",
            "-arch", &arch,
            "-o",    &format!("{out_dir}/vec_add.ptx"),
        ])
        .status()
        .expect("failed to spawn nvcc");

    if !status.success() {
        panic!("nvcc failed with exit code {status}");
    }
    println!("cargo:rustc-env=KERNEL_PTX={out_dir}/vec_add.ptx");
}
