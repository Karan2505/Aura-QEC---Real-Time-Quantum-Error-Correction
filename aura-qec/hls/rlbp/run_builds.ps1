# Run the fixed-point Vitis flow (csim, csynth, cosim, Vivado impl) for each build folder
# under hls/rlbp/builds/ and print a one-line summary per build.
#   powershell -File hls/rlbp/run_builds.ps1 s3_oam s5_base s5_oam
param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Builds)
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$vitis = if ($env:VITIS_RUN) { $env:VITIS_RUN } else { "E:\vitis\2025.1\Vitis\bin\vitis-run.bat" }
foreach ($b in $Builds) {
    $dir = Join-Path $root "builds\$b"
    Push-Location $dir
    $env:RLBP_MODE = "fixed"
    if (-not $env:RLBP_STEPS) { $env:RLBP_STEPS = "csim csynth cosim impl" }
    $t0 = Get-Date
    & $vitis --mode hls --tcl run_hls.tcl *> flow.log
    $mins = [math]::Round(((Get-Date) - $t0).TotalMinutes, 1)
    $sol = "rlbp_prj\sol_fixed"
    $pass = (Select-String flow.log -Pattern "PASS \(" | Select-Object -First 1).Line
    $cosim = (Select-String flow.log -Pattern "COSIM.*finished" | Select-Object -First 1).Line
    $lat = if (Test-Path "$sol\sim\report\rlbp_decode_cosim.rpt") { (Get-Content "$sol\sim\report\rlbp_decode_cosim.rpt")[10] } else { "no cosim report" }
    $impl = if (Test-Path "$sol\impl\report\verilog\rlbp_decode_export.rpt") { (Get-Content "$sol\impl\report\verilog\rlbp_decode_export.rpt" | Select-String "LUT:|FF:|DSP:|BRAM:|CP achieved|Timing") -join " | " } else { "no impl report" }
    "=== $b ($mins min) ==="
    "  csim: $pass"
    "  cosim: $cosim"
    "  latency: $lat"
    "  impl: $impl"
    Pop-Location
}
