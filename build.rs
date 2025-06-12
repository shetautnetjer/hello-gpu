use std::{
    env,
    fs,
    path::{Path, PathBuf},
    process::Command,
};

fn main() {
    if env::var_os("SKIP_CUDA").is_some() {
        println!("cargo:warning=SKIP_CUDA=1 – skipping CUDA kernels");
        return;
    }

    if Command::new("nvcc").arg("--version").output().is_err() {
        println!("cargo:warning=nvcc not found – skipping CUDA");
        return;
    }

    let (cl_path, sdk_version) = match find_windows_toolchain() {
        Ok(t) => t,
        Err(e) => {
            println!("cargo:warning={e}");
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
        println!("cargo:rustc-env={var}={}", ptx.display());
        println!("cargo:rerun-if-changed={}", cu.display());
    }
}

fn find_windows_toolchain() -> Result<(PathBuf, String), String> {
    let vswhere = r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe";
    let vs_root = String::from_utf8(
        Command::new(vswhere)
            .args(["-latest", "-products", "*", "-property", "installationPath"])
            .output()
            .map_err(|_| "vswhere.exe not found. Install VS Build Tools.")?
            .stdout,
    )
    .unwrap();
    if vs_root.trim().is_empty() {
        return Err("MSVC not installed".into());
    }

    let cl = Path::new(vs_root.trim())
        .join(r"VC\Tools\MSVC")
        .read_dir()
        .map_err(|_| "No MSVC toolsets found")?
        .filter_map(Result::ok)
        .map(|p| p.path())
        .filter(|p| p.join("bin\\Hostx64\\x64\\cl.exe").exists())
        .max()
        .ok_or("Could not find Hostx64\\x64\\cl.exe")?
        .join(r"bin\Hostx64\x64\cl.exe");

    let kits = Path::new(r"C:\Program Files (x86)\Windows Kits\10\Include");
    let sdk = kits.read_dir()
        .map_err(|_| "No Windows SDKs found")?
        .filter_map(Result::ok)
        .filter_map(|e| e.file_name().into_string().ok())
        .filter(|v| v.starts_with("10.0.") && v.as_str() <= "10.0.22621")
        .max()
        .ok_or("No Windows SDK ≤ 10.0.22621 found")?;

    Ok((cl, format!("{sdk}.0")))
}