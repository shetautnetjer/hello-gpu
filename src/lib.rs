//! GPU helper functions for other crates (and for `main.rs`).

use cust::{error::CudaResult, memory::DeviceBuffer, prelude::*, version::DriverVersion};
use nvml_wrapper::Nvml;
use serde::Serialize;
use std::error::Error;

/// PTX for the vec-add demo kernel – compiled by build.rs and embedded here.
const VEC_ADD_PTX: &str = include_str!(env!("KERNEL_VEC_ADD_PTX"));

/// Safe wrapper: adds two equal-length slices on the GPU, returns the result.
/// Falls back to a CPU loop if CUDA init fails or no device is present.
pub fn vec_add_gpu(a: &[f32], b: &[f32]) -> Vec<f32> {
    assert_eq!(a.len(), b.len(), "input vectors must be same length");
    let n = a.len();

    // Try to init CUDA; on failure do CPU fallback.
    if let Ok(_ctx) = cust::quick_init() {
        if let Ok(out) = vec_add_cuda(a, b) {
            return out; // fast path ✨
        }
    }

    // --- CPU fallback ---
    a.iter().zip(b).map(|(&x, &y)| x + y).collect()
}

/// Summary about the first CUDA device and kernel sanity check.
#[derive(Serialize)]
pub struct GpuReport {
    pub name: String,
    pub pcie_bus_id: String,
    pub sm_major_minor: (i32, i32),
    pub total_mem_mb: u64,
    pub driver_version: String,
    pub runtime_version: (u32, u32),
    pub ptx_version: (i32, i32),
    pub kernel_ok: bool,
    pub elapsed_us: u128,
}

/// Detects GPU metadata, runs a small kernel and returns a report.
pub fn generate_report() -> Result<GpuReport, Box<dyn Error>> {
    // ── initialise CUDA ────────────────────────────────────────────────
    cust::init(cust::CudaFlags::empty())?;
    let device = Device::get_device(0)?;
    let _ctx = Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)?;

    // ── grab driver/runtime metadata ───────────────────────────────────
    let (rt_major, rt_minor) = cust::version()?;
    let (ptx_major, ptx_minor) = Module::ptx_target_version();
    let (drv_major, drv_minor): (u32, u32) = DriverVersion::get()?.into();

    let nvml = Nvml::init()?;
    let handle = nvml.device_by_index(0)?;

    // ── load PTX & kernel ──────────────────────────────────────────────
    let module = Module::from_ptx(VEC_ADD_PTX, &[])?;
    let func = module.get_function("vec_add")?;

    // ── allocate host/device buffers ───────────────────────────────────
    let n = 1 << 20;
    let h_a = vec![1.0f32; n];
    let h_b = vec![2.0f32; n];
    let mut h_c = vec![0.0f32; n];

    let d_a = DeviceBuffer::from_slice(&h_a)?;
    let d_b = DeviceBuffer::from_slice(&h_b)?;
    let mut d_c = DeviceBuffer::<f32>::uninitialized(n)?;

    let stream = Stream::new(StreamFlags::DEFAULT, None)?;

    // ── launch kernel ──────────────────────────────────────────────────
    let tic = std::time::Instant::now();
    unsafe {
        launch!(
            func<<<(n as u32 + 255) / 256, 256, 0, stream>>>(
                d_a.as_device_ptr(),
                d_b.as_device_ptr(),
                d_c.as_device_ptr(),
                n as i32
            )
        )?;
    }
    stream.synchronize()?;
    let elapsed = tic.elapsed().as_micros();

    // ── copy back & validate ───────────────────────────────────────────
    d_c.copy_to(&mut h_c)?;
    let ok = h_c.iter().all(|&v| (v - 3.0).abs() < 1e-6);

    // ── build JSON report ──────────────────────────────────────────────
    let report = GpuReport {
        name: device.name()?.to_owned(),
        pcie_bus_id: handle.pci_info()?.bus_id().to_string(),
        sm_major_minor: (device.major_version()?, device.minor_version()?),
        total_mem_mb: handle.memory_info()?.total / 1_048_576,
        driver_version: format!("{drv_major}.{drv_minor}"),
        runtime_version: (rt_major, rt_minor),
        ptx_version: (ptx_major, ptx_minor),
        kernel_ok: ok,
        elapsed_us: elapsed,
    };

    Ok(report)
}

/// Internal: launch the CUDA kernel.
fn vec_add_cuda(a: &[f32], b: &[f32]) -> CudaResult<Vec<f32>> {
    // Load module
    let _ctx = cust::quick_init()?; // context guard
    let module = Module::from_ptx(VEC_ADD_PTX, &[])?; // JIT -> cubin
    let func = module.get_function("vec_add")?;

    // Copy data to device
    let d_a = DeviceBuffer::from_slice(a)?;
    let d_b = DeviceBuffer::from_slice(b)?;
    let mut d_c = DeviceBuffer::<f32>::uninitialized(a.len())?;
    let stream = Stream::new(StreamFlags::DEFAULT, None)?;

    // Launch <<<ceil(n/256), 256>>>
    unsafe {
        launch!(
            func<<<((a.len() as u32 + 255) / 256), 256, 0, stream>>>(
                d_a.as_device_ptr(),
                d_b.as_device_ptr(),
                d_c.as_device_ptr(),
                a.len() as i32
            )
        )?;
    }
    stream.synchronize()?;

    // Copy result back
    let mut out = vec![0.0f32; a.len()];
    d_c.copy_to(&mut out)?;
    Ok(out)
}
