use cust::{
    context::{Context, ContextFlags},
    device::Device,
    error::CudaResult,
    memory::DeviceBuffer,
    module::Module,
    stream::{Stream, StreamFlags},
    CudaFlags,
};
use nvml_wrapper::Nvml;
use serde::Serialize;

const VEC_ADD_PTX: &str = include_str!(env!("KERNEL_VEC_ADD_PTX"));

pub fn vec_add_gpu(a: &[f32], b: &[f32]) -> Vec<f32> {
    if let Ok(result) = try_vec_add_cuda(a, b) {
        result
    } else {
        a.iter().zip(b).map(|(&x, &y)| x + y).collect()
    }
}

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

pub fn generate_report() -> Result<GpuReport, Box<dyn std::error::Error>> {
    cust::init(CudaFlags::empty())?;
    let device = Device::get_device(0)?;
    let ctx = Context::new(device)?;
    ctx.set_current()?;

    let nvml = Nvml::init()?;
    let handle = nvml.device_by_index(0)?;
    let pci = handle.pci_info()?;

    let module = Module::from_ptx(VEC_ADD_PTX, &[])?;
    let func = module.get_function("vec_add")?;

    let n = 1 << 20;
    let h_a = vec![1.0f32; n];
    let h_b = vec![2.0f32; n];
    let mut h_c = vec![0.0f32; n];

    let d_a = DeviceBuffer::from_slice(&h_a)?;
    let d_b = DeviceBuffer::from_slice(&h_b)?;
    let mut d_c = unsafe { DeviceBuffer::uninitialized(n)? };

    let stream = Stream::new(StreamFlags::DEFAULT, None)?;
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

    d_c.copy_to(&mut h_c)?;
    let ok = h_c.iter().all(|&x| (x - 3.0).abs() < 1e-6);

    Ok(GpuReport {
        name: device.name()?.to_owned(),
        pcie_bus_id: pci.bus_id().to_string(),
        sm_major_minor: device.compute_capability()?,
        total_mem_mb: handle.memory_info()?.total / 1_048_576,
        driver_version: format!("{:?}", nvml.driver_version()?),
        runtime_version: cust::CudaVersion::get_runtime_version()?,
        ptx_version: cust::CudaVersion::get_ptx_version(),
        kernel_ok: ok,
        elapsed_us: elapsed,
    })
}

fn try_vec_add_cuda(a: &[f32], b: &[f32]) -> CudaResult<Vec<f32>> {
    let _ctx = cust::quick_init()?;
    let module = Module::from_ptx(VEC_ADD_PTX, &[])?;
    let func = module.get_function("vec_add")?;

    let d_a = DeviceBuffer::from_slice(a)?;
    let d_b = DeviceBuffer::from_slice(b)?;
    let mut d_c = unsafe { DeviceBuffer::uninitialized(a.len())? };

    let stream = Stream::new(StreamFlags::DEFAULT, None)?;
    unsafe {
        launch!(
            func<<<(a.len() as u32 + 255) / 256, 256, 0, stream>>>(
                d_a.as_device_ptr(),
                d_b.as_device_ptr(),
                d_c.as_device_ptr(),
                a.len() as i32
            )
        )?;
    }
    stream.synchronize()?;

    let mut output = vec![0.0f32; a.len()];
    d_c.copy_to(&mut output)?;
    Ok(output)
}
