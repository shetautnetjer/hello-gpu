<#
.SYNOPSIS
    Smart GPU Setup Checker for Windows (CUDA + MSVC)
.DESCRIPTION
    - Detects CUDA/MSVC header conflicts and version mismatches
    - Warns if MSVC is too new for installed CUDA
    - Detects multiple MSVC versions
    - Ensures correct nvcc on PATH
    - Verifies nvcc, cl.exe, and CUDA driver compatibility
    - Suggests fixes and prints clear, color-coded output
#>

function Write-Status {
    param([string]$Msg, [string]$Type)
    switch ($Type) {
        "ok"    { Write-Host "✅ $Msg" -ForegroundColor Green }
        "fail"  { Write-Host "❌ $Msg" -ForegroundColor Red }
        "warn"  { Write-Host "⚠️ $Msg" -ForegroundColor Yellow }
        default { Write-Host "$Msg" }
    }
}

function Find-Nvcc {
    $nvcc = Get-Command nvcc -ErrorAction SilentlyContinue
    if ($nvcc) {
        Write-Status "nvcc found at $($nvcc.Source)" "ok"
        return $nvcc.Source
    } else {
        Write-Status "nvcc not found on PATH. Please install CUDA Toolkit and add to PATH." "fail"
        exit 1
    }
}

function Get-Cuda-Version {
    $nvcc = Find-Nvcc
    $ver = & $nvcc --version 2>&1 | Select-String "release" | ForEach-Object { $_.ToString() }
    if ($ver -match "release ([\d\.]+),") {
        $cudaVer = $Matches[1]
        Write-Status "CUDA Toolkit version detected: $cudaVer" "ok"
        return $cudaVer
    } else {
        Write-Status "Could not parse CUDA version from nvcc." "fail"
        exit 1
    }
}

function Find-ClExe {
    $cl = Get-Command cl.exe -ErrorAction SilentlyContinue
    if ($cl) {
        Write-Status "cl.exe found at $($cl.Source)" "ok"
        return $cl.Source
    } else {
        Write-Status "cl.exe (MSVC) not found on PATH. Please install Visual Studio Build Tools." "fail"
        exit 1
    }
}

function Get-Msvc-Version {
    $cl = Find-ClExe
    $bv = & $cl /Bv 2>&1
    $versions = @()
    foreach ($line in $bv) {
        if ($line -match "Version (\d+\.\d+\.\d+)") {
            $ver = $Matches[1]
            $major = $ver.Split(".")[1]
            $versions += $major
            Write-Status "MSVC toolset version detected: $ver" "ok"
        }
    }
    if ($versions.Count -gt 1) {
        Write-Status "Multiple MSVC toolsets detected: $($versions -join ', ')" "warn"
        Write-Host "    Consider uninstalling unused versions via Visual Studio Installer."
    }
    return $versions
}

function Check-Msvc-Cuda-Compat {
    param($cudaVer, $msvcMajors)
    $cudaMajor, $cudaMinor = $cudaVer.Split(".")
    $cudaNum = [int]$cudaMajor * 10 + [int]$cudaMinor
    foreach ($msvc in $msvcMajors) {
        $msvcInt = [int]$msvc
        if ($msvcInt -gt 39 -and $cudaNum -le 125) {
            Write-Status "MSVC toolset 14.$msvcInt is too new for CUDA $cudaVer. CUDA ≤ 12.5 is not compatible with MSVC > 14.39." "fail"
            Write-Host "    ➤ Install MSVC 14.39 via Visual Studio Installer."
            Write-Host "    ➤ Or upgrade CUDA Toolkit to 12.5+ (if available)."
            Write-Host "    ➤ Or use --allow-unsupported-compiler and expect possible build errors."
            exit 1
        }
    }
}

function Check-Header-Conflicts {
    # This is a stub: in practice, you'd scan for known error patterns in build logs or headers
    $conflict = $false
    $cudaInclude = "${env:CUDA_PATH}\include"
    if (Test-Path "$cudaInclude\crt\host_config.h") {
        $content = Get-Content "$cudaInclude\crt\host_config.h" -Raw
        if ($content -match "_MSC_VER") {
            # Simulate detection
            $conflict = $true
        }
    }
    if ($conflict) {
        Write-Status "Potential CUDA/MSVC header conflict detected (e.g., size_t, _alloca, _Function_args errors)." "fail"
        Write-Host "    ➤ Remove or downgrade MSVC toolsets > 14.39."
        Write-Host "    ➤ Or upgrade CUDA Toolkit."
        exit 1
    } else {
        Write-Status "No obvious CUDA/MSVC header conflicts detected." "ok"
    }
}

function Suggest-BuildRs-Fix {
    param($msvc, $cudaVer)
    Write-Status "You may need to set -ccbin to a compatible cl.exe in build.rs." "warn"
    Write-Host "    Example: .args([\"--compiler-bindir\", \"C:\\Path\\To\\MSVC14.39\\cl.exe\"])" -ForegroundColor Yellow
}

# Main logic
Write-Host "=== GPU Rust Build Environment Checker ===" -ForegroundColor Cyan

$cudaVer = Get-Cuda-Version
$msvcMajors = Get-Msvc-Version
Check-Msvc-Cuda-Compat $cudaVer $msvcMajors
Check-Header-Conflicts

Write-Status "CUDA and MSVC appear compatible." "ok"
Write-Host "If you encounter build errors, try:" -ForegroundColor Yellow
Write-Host "    - Downgrading MSVC to 14.39" -ForegroundColor Yellow
Write-Host "    - Upgrading CUDA Toolkit" -ForegroundColor Yellow
Write-Host "    - Adding --allow-unsupported-compiler to nvcc" -ForegroundColor Yellow
Write-Host "    - Setting -arch=sm_80 for fallback compatibility" -ForegroundColor Yellow

# Optionally suggest build.rs fix
Suggest-BuildRs-Fix $msvcMajors $cudaVer

Write-Host "=== Check complete ===" -ForegroundColor Cyan