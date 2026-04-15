param()
$ErrorActionPreference = 'Stop'

# Refuse elevated/admin
$principal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
if ($principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
  Write-Error 'Refusing to run elevated/Admin.'
}

$ws = Join-Path $env:USERPROFILE 'AI_Workspace'
$logs = Join-Path (Get-Location) 'logs'
New-Item -ItemType Directory -Force -Path $ws | Out-Null
New-Item -ItemType Directory -Force -Path $logs | Out-Null

$env:NO_REMOTE = '1'
$env:MOTHER_SECURE = '1'

# Firewall rule for eDEX inbound block
try {
  $edex = (Get-ChildItem -Recurse -Filter edex-ui.exe -ErrorAction SilentlyContinue | Select-Object -First 1).FullName
  if ($edex) {
    New-NetFirewallRule -DisplayName 'Block eDEX inbound' -Program $edex -Direction Inbound -Action Block -Profile Any -ErrorAction SilentlyContinue | Out-Null
  }
} catch {}

# Start eDEX via npm.cmd to avoid opening npm.ps1 in editors
$npm = $null
try { $npm = (Get-Command npm.cmd -ErrorAction SilentlyContinue).Source } catch {}
if (-not $npm) { Write-Error 'npm.cmd not found in PATH. Install Node.js (includes npm).'; exit 1 }
Start-Process -FilePath $npm -WorkingDirectory (Join-Path (Get-Location) 'edex-ui') -ArgumentList 'start'
Start-Sleep -Seconds 4
.\.venv\Scripts\python main.py
