//! Demo binary: launches the vec_add CUDA kernel, validates the result
//! and prints a JSON health-report about the GPU + driver stack.

use hello_gpu::generate_report;

// ───────────────────────── entry point ──────────────────────────────────
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let report = generate_report()?;
    
    // Print the report with pretty formatting
    println!("{}", serde_json::to_string_pretty(&report)?);
    
    // If there was an error, exit with non-zero status
    if let Some(error) = report.error_message {
        eprintln!("\n❌ Error: {}", error);
        std::process::exit(1);
    }
    
    // If the kernel test failed, exit with non-zero status
    if !report.kernel_ok {
        eprintln!("\n❌ Kernel test failed");
        std::process::exit(1);
    }
    
    Ok(())
}
