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
