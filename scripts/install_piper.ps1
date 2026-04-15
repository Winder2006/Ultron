Param(
    [string]$Version = "latest"
)

$ErrorActionPreference = "Stop"

$repoApi = if ($Version -eq "latest") {
    "https://api.github.com/repos/rhasspy/piper/releases/latest"
} else {
    "https://api.github.com/repos/rhasspy/piper/releases/tags/$Version"
}

Write-Host "Fetching Piper release metadata: $repoApi"
$release = Invoke-RestMethod -Uri $repoApi -Headers @{ "Accept" = "application/vnd.github+json" }

# Find a Windows x64 zip asset
$asset = $release.assets | Where-Object { $_.name -match "windows.*(amd64|x86_64).*\.zip$" } | Select-Object -First 1
if (-not $asset) {
    throw "Could not find a Windows x64 Piper zip in release assets."
}

$binRoot = Join-Path $PSScriptRoot "..\bin"
$destDir = Join-Path $binRoot "piper"
New-Item -ItemType Directory -Force -Path $destDir | Out-Null

$zipPath = Join-Path $env:TEMP $asset.name
Write-Host "Downloading $($asset.name)"
Invoke-WebRequest -Uri $asset.browser_download_url -OutFile $zipPath

Write-Host "Extracting to $destDir"
Expand-Archive -Path $zipPath -DestinationPath $destDir -Force

$exe = Get-ChildItem -Path $destDir -Filter piper.exe -Recurse | Select-Object -First 1
if (-not $exe) {
    throw "piper.exe not found after extraction in $destDir"
}

Write-Host "Installed Piper at $($exe.FullName)"
return 0
