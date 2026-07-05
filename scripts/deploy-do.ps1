# Deploy to DigitalOcean App Platform
#
# Prerequisites:
#   1. DIGITALOCEAN_ACCESS_TOKEN from https://cloud.digitalocean.com/account/api/tokens
#   2. GitHub linked to DO: https://cloud.digitalocean.com/apps/github/install
#
# Usage:
#   $env:DIGITALOCEAN_ACCESS_TOKEN = "dop_v1_..."
#   .\scripts\deploy-do.ps1

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

if (-not $env:DIGITALOCEAN_ACCESS_TOKEN) {
    Write-Error "Set DIGITALOCEAN_ACCESS_TOKEN (DigitalOcean API token)."
}

$env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")

if (-not (Get-Command doctl -ErrorAction SilentlyContinue)) {
    winget install DigitalOcean.Doctl --accept-package-agreements --accept-source-agreements
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")
}

doctl auth init --access-token $env:DIGITALOCEAN_ACCESS_TOKEN

$geminiKey = $null
if (Test-Path ".env") {
    foreach ($line in Get-Content ".env") {
        if ($line -match '^\s*GEMINI_API_KEY\s*=\s*(.+)\s*$') {
            $geminiKey = $Matches[1].Trim().Trim('"').Trim("'")
            break
        }
    }
}
if (-not $geminiKey) {
    Write-Error "GEMINI_API_KEY not found in local .env"
}

$specPath = ".do/app-deploy.yaml"
$spec = Get-Content ".do/app.yaml" -Raw
$spec = $spec -replace "(?m)^(\s*- key: GEMINI_API_KEY\s*\r?\n\s*scope: RUN_TIME\s*\r?\n\s*type: SECRET)\s*$", "`$1`n        value: $geminiKey"
$spec | Set-Content $specPath -Encoding utf8

$AppName = "color-book-maker"
$existing = doctl apps list --format ID,Spec.Name --no-header 2>$null | Select-String $AppName

if ($existing) {
    $appId = ($existing.ToString() -split '\s+')[0]
    Write-Host "Updating app $appId..."
    doctl apps update $appId --spec $specPath
    doctl apps create-deployment $appId
} else {
    Write-Host "Creating App Platform app..."
    $result = doctl apps create --spec $specPath --format ID,DefaultIngress,Created --no-header
    Write-Host $result
    $appId = ($result -split '\s+')[0]
}

Start-Sleep -Seconds 5
$url = doctl apps get $appId --format DefaultIngress --no-header
Write-Host ""
Write-Host "Live URL: https://$url"
Write-Host "Dashboard: https://cloud.digitalocean.com/apps/$appId"
