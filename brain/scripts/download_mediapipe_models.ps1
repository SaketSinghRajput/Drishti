<#
Runs the Python download helper to fetch MediaPipe TFLite models.

Usage (PowerShell):
  .\scripts\download_mediapipe_models.ps1

The script expects `scripts/mediapipe_model_urls.json` to exist. Copy
`scripts/mediapipe_model_urls.sample.json` -> `scripts/mediapipe_model_urls.json`
and edit URLs if necessary.
#>

$root = Split-Path -Parent $PSScriptRoot
$py = "python"
$script = Join-Path $root "scripts\download_mediapipe_models.py"

if (-not (Test-Path $script)) {
    Write-Error "Download script not found: $script"
    exit 1
}

$jsonSample = Join-Path $root "scripts\mediapipe_model_urls.sample.json"
$jsonTarget = Join-Path $root "scripts\mediapipe_model_urls.json"

if (-not (Test-Path $jsonTarget)) {
    if (Test-Path $jsonSample) {
        Write-Host "Creating $jsonTarget from sample (edit it to provide real URLs)..."
        Copy-Item -Path $jsonSample -Destination $jsonTarget -Force
    } else {
        Write-Host "No $jsonTarget or sample found. Create $jsonTarget with mapping of target->URL."
        exit 1
    }
}

Write-Host "Running: $py $script"
& $py $script
