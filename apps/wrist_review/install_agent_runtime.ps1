[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"

$appRoot = $PSScriptRoot
$projectRoot = (Resolve-Path (Join-Path $appRoot "..\..")).Path
$environmentRoot = Join-Path $projectRoot ".venv-agent"
$python = Join-Path $environmentRoot "Scripts\python.exe"

if (-not (Test-Path -LiteralPath $python)) {
    Write-Host "Creating the isolated Python environment..."
    python -m venv $environmentRoot
}

Write-Host "Installing the optional CrewAI and PDF dependencies..."
& $python -m pip install --upgrade pip
& $python -m pip install -r (Join-Path $projectRoot "requirements_agent.txt")

Write-Host "Agent runtime is ready. Run .\start_with_crewai.ps1 to start the app."
