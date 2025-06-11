//! Demo binary: launches the vec_add CUDA kernel, validates the result
//! and prints a JSON health-report about the GPU + driver stack.

use hello_gpu::{generate_report, GpuReport};

// ───────────────────────── entry point ──────────────────────────────────
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let report = generate_report()?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
