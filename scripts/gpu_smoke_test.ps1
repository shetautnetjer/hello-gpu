# ── 1. Environment Check ────────────────────────────────────────────────
Write-Host "`n🔍 Checking CUDA environment..." -ForegroundColor Cyan

# Verify required environment variables
$requiredVars = @("CCBIN_PATH", "CUDA_PATH", "CUDA_TOOLKIT_ROOT_DIR")
$missingVars = $requiredVars | Where-Object { -not (Get-Item "env:$_" -ErrorAction SilentlyContinue) }

if ($missingVars) {
    Write-Error "❌ Missing required environment variables: $($missingVars -join ', ')"
    Write-Host "💡 Run init_gpu_env.ps1 first to set up the environment"
    exit 1
}

# ── 2. Write a minimal CUDA program ────────────────────────────────────────
$cu = @'
#include <cuda_runtime.h>
#include <stdio.h>

// device kernel: C[i] = A[i] + B[i]
__global__ void vec_add(const float* A, const float* B, float* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) C[i] = A[i] + B[i];
}

int main() {
    // Print CUDA device info
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    printf("Found %d CUDA device(s)\n", deviceCount);

    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device %d: %s\n", i, prop.name);
        printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("  Total Memory: %.2f GB\n", prop.totalGlobalMem / 1024.0 / 1024.0 / 1024.0);
    }

    const int N = 1 << 20;                 // 1 M elements
    const size_t bytes = N * sizeof(float);

    // host buffers
    float *hA = (float*)malloc(bytes);
    float *hB = (float*)malloc(bytes);
    float *hC = (float*)malloc(bytes);
    for (int i = 0; i < N; ++i) { hA[i] = 1.0f; hB[i] = 2.0f; }

    // device buffers
    float *dA, *dB, *dC;
    cudaMalloc(&dA, bytes);
    cudaMalloc(&dB, bytes);
    cudaMalloc(&dC, bytes);

    cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice);

    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);
    vec_add<<<grid, block>>>(dA, dB, dC, N);
    cudaMemcpy(hC, dC, bytes, cudaMemcpyDeviceToHost);

    // verify
    bool ok = true;
    for (int i = 0; i < N; ++i)
        if (fabs(hC[i] - 3.0f) > 1e-5f) { ok = false; break; }

    printf(ok ? "✅ GPU vector-add succeeded\n" : "❌ Mismatch!\n");

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(hA); free(hB); free(hC);
    return ok ? 0 : 1;
}
'@
Set-Content -Path "$env:TEMP\vec_add.cu" -Value $cu

# ── 3. Compile with nvcc ───────────────────────────────────────────────────
Write-Host "`n🔨 Compiling CUDA program..." -ForegroundColor Cyan

$ccbin = $env:CCBIN_PATH
if (-not $ccbin) { 
    Write-Warning "CCBIN_PATH not set; nvcc will search PATH."
    Write-Host "💡 Run init_gpu_env.ps1 first to set up the environment"
    exit 1
}

$exe = "$env:TEMP\vec_add.exe"
$compileCmd = "nvcc `"$env:TEMP\vec_add.cu`" -o `"$exe`" -arch=sm_89 -ccbin `"$ccbin`" --allow-unsupported-compiler"
Write-Host "Running: $compileCmd"

try {
    Invoke-Expression $compileCmd
    if ($LASTEXITCODE -ne 0) {
        Write-Error "❌ NVCC compilation failed"
        exit 1
    }
    Write-Host "✅ Compilation successful"
} catch {
    Write-Error "❌ NVCC compilation failed: $_"
    exit 1
}

# ── 4. Run the test ────────────────────────────────────────────────────────
Write-Host "`n🚀 Running GPU smoke test..." -ForegroundColor Cyan
& $exe

if ($LASTEXITCODE -ne 0) {
    Write-Error "❌ GPU smoke test failed"
    exit 1
}

Write-Host "`n✅ GPU smoke test completed successfully" -ForegroundColor Green
