use cust::error::CudaResult;
use cust::prelude::*;
use nvml_wrapper::Nvml;
use serde::Serialize;
use std::ffi::CString;

#[derive(Serialize)]
struct GpuReport<'a> {
    name:            &'a str,
    pcie_bus_id:     String,
    sm_major_minor:  (i32, i32),
    total_mem_mb:    u64,
    driver_version:  String,
    runtime_version: (u32, u32),
    ptx_version:     (i32, i32),
    kernel_ok:       bool,
}

fn main() -> CudaResult<()> {
    // ── host initialisation ──────────────────────────────────────────────
    cust::quick_init()?;

    let device = Device::get_device(0)?;
    let mut ctx = Context::new(device)?;
    ctx.set_current()?;

    // ── gather static properties via cust & NVML ─────────────────────────
    let nvml   = Nvml::init().unwrap();
    let handle = nvml.device_by_index(0).unwrap();

    let (drv_major, drv_minor) = DriverVersion::get()?.into();
    let (rt_major,  rt_minor)  = cust::version()?;

    let report = {
        let (_ptx_major, _ptx_minor) = Module::ptx_target_version();
        GpuReport {
            name:            device.name()?,
            pcie_bus_id:     handle.pci_info().unwrap().bus_id().to_string(),
            sm_major_minor:  (device.major_version()?, device.minor_version()?),
            total_mem_mb:    handle.memory_info().unwrap().total / 1024 / 1024,
            driver_version:  format!("{drv_major}.{drv_minor}"),
            runtime_version: (rt_major, rt_minor),
            ptx_version:     (_ptx_major, _ptx_minor),
            kernel_ok:       false, // will update after test
        }
    };

    // ── load PTX into a module ────────────────────────────────────────────
    let ptx   = std::fs::read(std::env::var("KERNEL_PTX")?)?;
    let c_ptx = CString::new(ptx).unwrap();
    let module = Module::from_ptx(&c_ptx, &[])?;
    let func   = module.get_function("vec_add")?;

    // ── allocate data & launch test kernel ───────────────────────────────
    let n = 1 << 20;
    let mut host_out = vec![0f32; n];

    let (h_a, h_b): (Vec<f32>, Vec<f32>) = (vec![1.0; n], vec![2.0; n]);
    let d_a = DeviceBuffer::from_slice(&h_a)?;
    let d_b = DeviceBuffer::from_slice(&h_b)?;
    let mut d_c = DeviceBuffer::<f32>::uninitialized(n)?;

    let stream = Stream::new(StreamFlags::DEFAULT, None)?;

    unsafe {
        launch!(func<<<(n as u32 + 255) / 256, 256, 0, stream>>>(
            d_a.as_device_ptr(),
            d_b.as_device_ptr(),
            d_c.as_device_ptr(),
            n as i32
        ))?;
    }
    stream.synchronize()?;
    d_c.copy_to(&mut host_out)?;

    // validate
    let ok = host_out.iter().all(|&x| (x - 3.0).abs() < 1e-6);

    // ── print summary ────────────────────────────────────────────────────
    let mut final_report = report;
    final_report.kernel_ok = ok;
    println!("{}", serde_json::to_string_pretty(&final_report).unwrap());

    Ok(())
}
