//! Demo binary: launches the vec_add CUDA kernel, validates the result
//! and prints a JSON health-report about the GPU + driver stack.

use cust::{
    error::CudaResult,
    memory::DeviceBuffer,
    prelude::*,
    version::DriverVersion,
};
use nvml_wrapper::Nvml;
use serde::Serialize;

// ───────────────────────── constants ────────────────────────────────────
// The PTX is compiled by build.rs and its absolute path is exported as an
// env-var at *compile* time, so we can embed the file here.
const VEC_ADD_PTX: &str = include_str!(env!("KERNEL_VEC_ADD_PTX"));

// ───────────────────────── structs ──────────────────────────────────────
#[derive(Serialize)]
struct GpuReport {
    name:             String,
    pcie_bus_id:      String,
    sm_major_minor:   (i32, i32),
    total_mem_mb:     u64,
    driver_version:   String,
    runtime_version:  (u32, u32),
    ptx_version:      (i32, i32),
    kernel_ok:        bool,
    elapsed_us:       u128,
}

// ───────────────────────── entry point ──────────────────────────────────
fn main() -> CudaResult<()> {
    // ── initialise CUDA ────────────────────────────────────────────────
    cust::init(cust::CudaFlags::empty())?;
    let device = Device::get_device(0)?;
    let _ctx   = Context::create_and_push(
        ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO,
        device,
    )?;

    // ── grab driver/runtime metadata ───────────────────────────────────
    let (rt_major, rt_minor)      = cust::version()?;
    let (_ptx_major, _ptx_minor)  = Module::ptx_target_version();
    let (drv_major, drv_minor)    = DriverVersion::get()?.into();

    let nvml    = Nvml::init().unwrap();
    let handle  = nvml.device_by_index(0).unwrap();

    // ── load PTX & kernel ──────────────────────────────────────────────
    let module = Module::from_ptx(VEC_ADD_PTX, &[])?;
    let func   = module.get_function("vec_add")?;

    // ── allocate host/device buffers ───────────────────────────────────
    let n      = 1 << 20;
    let h_a    = vec![1.0f32; n];
    let h_b    = vec![2.0f32; n];
    let mut h_c = vec![0.0f32; n];

    let d_a    = DeviceBuffer::from_slice(&h_a)?;
    let d_b    = DeviceBuffer::from_slice(&h_b)?;
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
        name:            device.name()?.to_owned(),
        pcie_bus_id:     handle.pci_info().unwrap().bus_id().to_string(),
        sm_major_minor:  (device.major_version()?, device.minor_version()?),
        total_mem_mb:    handle.memory_info().unwrap().total / 1_048_576,
        driver_version:  format!("{drv_major}.{drv_minor}"),
        runtime_version: (rt_major, rt_minor),
        ptx_version:     (_ptx_major, _ptx_minor),
        kernel_ok:       ok,
        elapsed_us:      elapsed,
    };

    println!("{}", serde_json::to_string_pretty(&report).unwrap());
    Ok(())
}