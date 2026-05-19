param(
  [string]$Owner = 'R16-srihari',
  [string]$Repo  = 'RK-7-8-integrator-validation',
  [string]$Tag   = 'stk-latest',
  [string]$File  = 'STK_input/Satellite1_Results.csv'
)

# Also publish the Satellite1.opm release asset when present
$Additional = 'STK_input/Satellite1.opm'

# Require at least one of the files to exist
if (-not (Test-Path $File) -and -not (Test-Path $Additional)) {
  Write-Error "No release asset files found: $File or $Additional"
  exit 1
}

# Prefer gh if available
if (Get-Command gh -ErrorAction SilentlyContinue) {
  $filesToUpload = @()
  if (Test-Path $File) { $filesToUpload += (Resolve-Path $File).Path }
  if (Test-Path $Additional) { $filesToUpload += (Resolve-Path $Additional).Path }

  foreach ($f in $filesToUpload) {
    $leaf = Split-Path $f -Leaf
    $asset = gh api repos/$Owner/$Repo/releases/tags/$Tag --jq ".assets[] | select(.name==\"$leaf\") | .id" 2>$null
    if ($asset) { gh api repos/$Owner/$Repo/releases/assets/$asset -X DELETE }
    gh release upload $Tag $f --repo $Owner/$Repo --clobber
  }
  Write-Output "Upload complete"
  exit 0
}

# Fallback: use REST API with GITHUB_TOKEN
$token = $env:GITHUB_TOKEN
if (-not $token) {
  Write-Error "gh not found and GITHUB_TOKEN not set. Please install gh or set GITHUB_TOKEN."
  exit 1
}

$release = Invoke-RestMethod -Headers @{Authorization = "token $token"} -Uri "https://api.github.com/repos/$Owner/$Repo/releases/tags/$Tag"

# Helper to delete existing asset(s) by name
function Remove-AssetIfExists($name) {
  $assets = $release.assets | Where-Object { $_.name -eq $name }
  foreach ($a in $assets) { Invoke-RestMethod -Method Delete -Headers @{Authorization = "token $token"} -Uri $a.url }
}

# Upload files that exist
foreach ($f in @($File, $Additional)) {
  if (-not (Test-Path $f)) { continue }
  $leaf = Split-Path $f -Leaf
  Remove-AssetIfExists $leaf
  $uploadUrl = $release.upload_url -replace '\{.*\}','' + "?name=$leaf"
  Invoke-RestMethod -Method Post -Uri $uploadUrl -Headers @{Authorization = "token $token"; 'Content-Type'='application/octet-stream'} -InFile $f -UseBasicParsing
}

Write-Output "Upload complete"
