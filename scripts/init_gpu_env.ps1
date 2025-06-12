Write-Host "`nSetting up CUDA + MSVC for nvcc usage" -ForegroundColor Cyan

# Function to find Visual Studio installation
function Find-VisualStudio {
    $vsInstall = & "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" `
        -products * `
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
        -property installationPath `
        -nologo

    if (-not $vsInstall) {
        Write-Error "Could not locate Visual Studio with C++ tools"
        return $null
    }
    return $vsInstall
}

# Function to find CUDA installation
function Find-CudaInstallation {
    $cudaPaths = @(
        "${env:ProgramFiles}\NVIDIA GPU Computing Toolkit\CUDA",
        "${env:ProgramFiles(x86)}\NVIDIA GPU Computing Toolkit\CUDA"
    )

    foreach ($basePath in $cudaPaths) {
        if (Test-Path $basePath) {
            $versions = Get-ChildItem -Path $basePath -Directory | Sort-Object Name -Descending
            if ($versions.Count -gt 0) {
                return $versions[0].FullName
            }
        }
    }
    
    Write-Error "Could not locate CUDA installation"
    return $null
}

# Function to find cl.exe
function Find-ClExe {
    param (
        [string]$vsInstall
    )
    
    $clPath = Get-ChildItem -Path "$vsInstall\VC\Tools\MSVC" -Directory |
        Sort-Object Name -Descending |
        Select-Object -First 1 |
        ForEach-Object {
            Join-Path $_.FullName "bin\Hostx64\x64\cl.exe"
        }

    if (-not (Test-Path $clPath)) {
        Write-Error "Could not find cl.exe inside Visual Studio"
        return $null
    }
    return $clPath
}

# Main setup process
$vsInstall = Find-VisualStudio
if (-not $vsInstall) { exit 1 }

$clPath = Find-ClExe -vsInstall $vsInstall
if (-not $clPath) { exit 1 }

$cudaPath = Find-CudaInstallation
if (-not $cudaPath) { exit 1 }

# Print and export paths
$ccbinDir = Split-Path $clPath
Write-Host "Found cl.exe at: $clPath"
Write-Host "Found CUDA at: $cudaPath"
Write-Host "Set this as -ccbin: $ccbinDir"

# Export environment variables for use in build.rs and other scripts
$env:CCBIN_PATH = $ccbinDir
$env:CUDA_PATH = $cudaPath
$env:CUDA_TOOLKIT_ROOT_DIR = $cudaPath
$env:CUDA_LIBRARY_PATH = Join-Path $cudaPath "lib\x64"

Write-Host "`nExported environment variables:"
Write-Host "  CCBIN_PATH = $ccbinDir"
Write-Host "  CUDA_PATH = $cudaPath"
Write-Host "  CUDA_TOOLKIT_ROOT_DIR = $cudaPath"
Write-Host "  CUDA_LIBRARY_PATH = $env:CUDA_LIBRARY_PATH"

# Add CUDA paths to PATH if not already present
$cudaBinPath = Join-Path $cudaPath "bin"
if ($env:Path -notlike "*$cudaBinPath*") {
    $env:Path = "$cudaBinPath;$env:Path"
    Write-Host "`nAdded CUDA bin to PATH: $cudaBinPath"
}

# Verify nvcc is available
try {
    $nvccVersion = & nvcc --version
    Write-Host "`nNVCC is available:"
    Write-Host $nvccVersion
} catch {
    Write-Error "NVCC not found in PATH"
    exit 1
}
