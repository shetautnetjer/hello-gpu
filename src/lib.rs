//! GPU helper functions for other crates (and for `main.rs`).

use cust::{error::CudaResult, memory::DeviceBuffer, prelude::*};

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
            return out;                       // fast path ✨
        }
    }

    // --- CPU fallback ---
    a.iter().zip(b).map(|(&x, &y)| x + y).collect()
}

/// Internal: launch the CUDA kernel.
fn vec_add_cuda(a: &[f32], b: &[f32]) -> CudaResult<Vec<f32>> {
    // Load module
    let _ctx          = cust::quick_init()?;                 // context guard
    let module        = Module::from_ptx(VEC_ADD_PTX, &[])?; // JIT -> cubin
    let func          = module.get_function("vec_add")?;

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