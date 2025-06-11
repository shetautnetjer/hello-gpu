use std::{env, fs, path::PathBuf, process::Command};

use serde::Serialize;

#[derive(Serialize)]
struct KernelInfo {
    name: &'static str,
    source: &'static str,
    ptx_target: &'static str,
    symbol: &'static str,
    arg_signature: [&'static str; 3],
}

#[derive(Serialize)]
struct SimulatedReport {
    device: &'static str,
    sm: (u8, u8),
    driver_version: &'static str,
    ptx_version: (u8, u8),
    kernel_ok: bool,
}

#[derive(Serialize)]
struct GpuEnvReport {
    agent_version: &'static str,
    can_compile_cuda: bool,
    can_run_gpu_kernels: bool,
    missing_tools: Vec<&'static str>,
    cuda_arch_expected: &'static str,
    kernel: KernelInfo,
    build_rs_flags: BuildFlags,
    fallback_mode: &'static str,
    simulated_report: SimulatedReport,
    recommendations: Vec<&'static str>,
}

#[derive(Serialize)]
struct BuildFlags {
    arch: &'static str,
    output_dir: String,
    ptx_path_env: &'static str,
}

fn main() {
    let missing = detect_missing_tools(&["nvcc", "nvidia-smi", "clang", "ptxas", "ld"]);

    let out_dir = env::var("OUT_DIR").unwrap_or_else(|_| "target".to_string());

    let report = GpuEnvReport {
        agent_version: "v1.0",
        can_compile_cuda: !missing.contains(&"nvcc"),
        can_run_gpu_kernels: false,
        missing_tools: missing.clone(),
        cuda_arch_expected: "compute_89",
        kernel: KernelInfo {
            name: "vec_add",
            source: "kernels/vec_add.cu",
            ptx_target: &format!("{}/vec_add.ptx", out_dir),
            symbol: "vec_add",
            arg_signature: ["*mut f32", "*const f32", "usize"],
        },
        build_rs_flags: BuildFlags {
            arch: "compute_89",
            output_dir: out_dir.clone(),
            ptx_path_env: "KERNEL_PTX",
        },
        fallback_mode: "simulate",
        simulated_report: SimulatedReport {
            device: "Simulated NVIDIA RTX 4090",
            sm: (8, 9),
            driver_version: "535.104",
            ptx_version: (8, 0),
            kernel_ok: true,
        },
        recommendations: vec![
            "Mount precompiled PTX or ship as artifact",
            "Avoid runtime kernel testing in this container",
            "Use simulated GpuReport to validate pipeline logic",
        ],
    };

    let json_path = PathBuf::from("target/gpu_env_report.json");
    fs::create_dir_all("target").ok();
    fs::write(&json_path, serde_json::to_string_pretty(&report).unwrap())
        .expect("Failed to write gpu_env_report.json");

    println!("cargo:warning=Generated simulated GPU report to {:?}", json_path);
}

fn detect_missing_tools(tools: &[&str]) -> Vec<&'static str> {
    tools
        .iter()
        .copied()
        .filter(|tool| Command::new("which").arg(tool).output().map_or(true, |o| !o.status.success()))
        .collect()
}