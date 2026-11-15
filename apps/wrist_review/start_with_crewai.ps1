[CmdletBinding()]
param(
    [string]$ModelDir = $env:MURA_APP_MODEL_DIR,
    [string]$LlmModel = "gemini/gemini-3.1-flash-lite",
    [string]$PythonPath = "",
    [ValidateRange(1024, 65535)]
    [int]$Port = 8601
)

$ErrorActionPreference = "Stop"

$appRoot = $PSScriptRoot
$projectRoot = (Resolve-Path (Join-Path $appRoot "..\..")).Path
$managedVariables = @(
    "MURA_APP_MODEL_DIR",
    "MURA_APP_PORT",
    "MURA_CREWAI_ENABLED",
    "MURA_CREWAI_LLM_MODEL",
    "GOOGLE_API_KEY",
    "CREWAI_DISABLE_TELEMETRY",
    "CREWAI_STORAGE_DIR"
)
$previousEnvironment = @{}
foreach ($name in $managedVariables) {
    $previousEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, "Process")
}

if ([string]::IsNullOrWhiteSpace($PythonPath)) {
    $runtimeCandidates = @(
        (Join-Path $projectRoot ".venv-agent\Scripts\python.exe"),
        (Join-Path (Split-Path $projectRoot -Parent) ".venv\Scripts\python.exe")
    )
    foreach ($candidate in $runtimeCandidates) {
        if (-not (Test-Path -LiteralPath $candidate)) {
            continue
        }
        & $candidate -c "import importlib.util, sys; sys.exit(0 if all(importlib.util.find_spec(name) for name in ('crewai', 'torch', 'reportlab')) else 1)"
        if ($LASTEXITCODE -eq 0) {
            $PythonPath = $candidate
            break
        }
    }
}

if ([string]::IsNullOrWhiteSpace($PythonPath) -or -not (Test-Path -LiteralPath $PythonPath)) {
    throw "A Python runtime with CrewAI, PyTorch, and ReportLab is required. Run .\install_agent_runtime.ps1 first."
}
$python = (Resolve-Path -LiteralPath $PythonPath).Path

if ([string]::IsNullOrWhiteSpace($ModelDir)) {
    $ModelDir = Read-Host "Full path to the canonical Model B artifact folder"
}
if (-not (Test-Path -LiteralPath $ModelDir -PathType Container)) {
    throw "The ModelDir folder was not found. Choose the folder that contains the canonical Model B artifact files."
}
$resolvedModelDir = (Resolve-Path -LiteralPath $ModelDir -ErrorAction Stop).Path
$requiredBundleFiles = @(
    "best_resnet18_study_level.pt",
    "run_config.json",
    "selection.json",
    "training_history.json",
    "validation_metrics.json",
    "validation_predictions.csv",
    "validation_predictions_mean.csv",
    "validation_predictions_max.csv",
    "frozen_primary_operating_policy.json"
)
$missingBundleFiles = @($requiredBundleFiles | Where-Object { -not (Test-Path -LiteralPath (Join-Path $resolvedModelDir $_)) })
if ($missingBundleFiles.Count -gt 0) {
    throw "The selected folder is not the full canonical Model B bundle. Missing: $($missingBundleFiles -join ', ')"
}

Write-Host "Paste your Google AI Studio key locally. It is used only for this process and is never written to disk."
$secureApiKey = Read-Host "Google API key" -AsSecureString
$bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureApiKey)

try {
    $apiKey = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr)
    if ([string]::IsNullOrWhiteSpace($apiKey)) {
        throw "A Google API key is required for the real CrewAI explanation."
    }

    $env:MURA_APP_MODEL_DIR = $resolvedModelDir
    $env:MURA_APP_PORT = $Port.ToString()
    $env:MURA_CREWAI_ENABLED = "1"
    $env:MURA_CREWAI_LLM_MODEL = $LlmModel
    $env:GOOGLE_API_KEY = $apiKey
    $env:CREWAI_STORAGE_DIR = "wrist_review"
    $env:CREWAI_DISABLE_TELEMETRY = "true"

    Write-Host "Starting Wrist Review with the optional real CrewAI explanation enabled..."
    Write-Host "Open http://127.0.0.1:$Port and review a supported wrist study."
    & $python (Join-Path $appRoot "server.py")
}
finally {
    if ($null -ne $bstr -and $bstr -ne [IntPtr]::Zero) {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)
    }
    $apiKey = $null
    foreach ($name in $managedVariables) {
        [Environment]::SetEnvironmentVariable($name, $previousEnvironment[$name], "Process")
    }
}
