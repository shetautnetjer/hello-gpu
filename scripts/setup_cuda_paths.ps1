# Setup CUDA paths for Rust development
# This script:
# 1. Finds the latest CUDA installation
# 2. Verifies required libraries exist
# 3. Updates .cargo/config.toml with correct paths
# 4. Optionally persists the LIB environment variable

function Find-LatestCudaPath {
    # Find the newest CUDA_PATH* variable
    $cudaPathVar = Get-ChildItem Env:CUDA_PATH* |
                   Sort-Object Name -Descending |
                   Select-Object -First 1
    
    if (-not $cudaPathVar) {
        Write-Error "No CUDA_PATH* env vars found - is the CUDA Toolkit installed?"
        return $null
    }
    
    return $cudaPathVar.Value
}

function Test-CudaLibs {
    param (
        [string]$CudaRoot
    )
    
    $cudaLib = Join-Path $CudaRoot "lib\x64"
    $requiredLibs = @('cudart.lib', 'cuda.lib', 'cudadevrt.lib')
    
    $missing = $requiredLibs | Where-Object { -not (Test-Path (Join-Path $cudaLib $_)) }
    if ($missing) {
        Write-Error "Missing required libraries in ${cudaLib}:`n`t$($missing -join "`n`t")"
        Write-Host "Please re-run the CUDA installer and ensure Developer Libraries are selected."
        return $false
    }
    
    return $true
}

function Update-CargoConfig {
    param (
        [string]$CudaRoot,
        [string]$CudaLib,
        [string]$ClPath
    )
    
    $cargoConfigDir = ".cargo"
    $cargoConfigPath = Join-Path $cargoConfigDir "config.toml"
    
    # Create .cargo directory if it doesn't exist
    if (-not (Test-Path $cargoConfigDir)) {
        New-Item -ItemType Directory -Path $cargoConfigDir | Out-Null
    }
    
    # Escape backslashes for TOML
    $escapedCudaRoot = $CudaRoot.Replace('\', '\\')
    $escapedCudaLib = $CudaLib.Replace('\', '\\')
    $escapedClPath = $ClPath.Replace('\', '\\')
    
    $configContent = @"
[target.'cfg(windows)']
rustflags = [
    "-C", "link-arg=/LIBPATH:$escapedCudaLib",
    "-C", "link-arg=cudart.lib",
    "-C", "link-arg=cuda.lib",
    "-C", "link-arg=cudadevrt.lib",
]

[env]
CUDA_PATH = "$escapedCudaRoot"
CUDA_TOOLKIT_ROOT_DIR = "$escapedCudaRoot"
CUDA_LIBRARY_PATH = "$escapedCudaLib"
CCBIN_PATH = "$escapedClPath"
"@
    
    Set-Content -Path $cargoConfigPath -Value $configContent
    Write-Host "Updated $cargoConfigPath" -ForegroundColor Green
}

function Update-EnvironmentLib {
    param (
        [string]$CudaLib,
        [switch]$Persist
    )
    
    # Update LIB environment variable
    $env:LIB = "$CudaLib;$env:LIB"
    Write-Host "`nLIB => $env:LIB`n" -ForegroundColor Green
    
    # Optionally persist to profile
    if ($Persist) {
        $profileLine = '$env:LIB = "' + "$CudaLib;`$env:LIB" + '"'
        if (-not (Test-Path $PROFILE)) {
            New-Item -ItemType File -Path $PROFILE -Force | Out-Null
        }
        
        if (-not (Select-String -Path $PROFILE -Pattern 'cudart\.lib' -Quiet)) {
            Add-Content -Path $PROFILE -Value "`n# CUDA import libs`n$profileLine"
            Write-Host "Saved to `$PROFILE" -ForegroundColor Green
        } else {
            Write-Host "`$PROFILE already contains CUDA LIB entry" -ForegroundColor Yellow
        }
    }
}

function Find-ClExe {
    # Try to find cl.exe in Visual Studio installation
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $vswhere)) {
        Write-Error "vswhere.exe not found - is Visual Studio installed?"
        return $null
    }
    
    $vsPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if (-not $vsPath) {
        Write-Error "Visual Studio with C++ tools not found"
        return $null
    }
    
    $clPath = Join-Path $vsPath "VC\Tools\MSVC"
    $clPath = Get-ChildItem $clPath | Sort-Object Name -Descending | Select-Object -First 1
    $clPath = Join-Path $clPath.FullName "bin\Hostx64\x64\cl.exe"
    
    if (-not (Test-Path $clPath)) {
        Write-Error "cl.exe not found at expected location"
        return $null
    }
    
    return $clPath
}

# Main execution
function Setup-CudaPaths {
    param (
        [switch]$Persist
    )
    
    Write-Host "Setting up CUDA paths for Rust development..." -ForegroundColor Cyan
    
    # 1. Find CUDA installation
    $cudaRoot = Find-LatestCudaPath
    if (-not $cudaRoot) { return }
    Write-Host "Found CUDA at: $cudaRoot" -ForegroundColor Green
    
    # 2. Verify libraries
    $cudaLib = Join-Path $cudaRoot "lib\x64"
    if (-not (Test-CudaLibs $cudaRoot)) { return }
    
    # 3. Find cl.exe
    $clPath = Find-ClExe
    if (-not $clPath) { return }
    Write-Host "Found cl.exe at: $clPath" -ForegroundColor Green
    
    # 4. Update .cargo/config.toml
    Update-CargoConfig $cudaRoot $cudaLib $clPath
    
    # 5. Update environment
    Update-EnvironmentLib $cudaLib -Persist:$Persist
    
    Write-Host "`nCUDA paths setup completed successfully!" -ForegroundColor Green
    Write-Host "You can now run 'cargo build' to compile your project."
}

# Run the setup
Setup-CudaPaths 