param(
  [string]$Owner = 'R16-srihari',
  [string]$Repo  = 'RK-7-8-integrator-validation',
  [string]$Tag   = 'stk-latest',
  [string]$File  = 'STK_input/Satellite1_Results.csv'
)

if (-not (Test-Path $File)) {
  Write-Error "File not found: $File"
  exit 1
}

# Prefer gh if available
if (Get-Command gh -ErrorAction SilentlyContinue) {
  $asset = gh api repos/$Owner/$Repo/releases/tags/$Tag --jq ".assets[] | select(.name==\"$(Split-Path $File -Leaf)\") | .id" 2>$null
  if ($asset) { gh api repos/$Owner/$Repo/releases/assets/$asset -X DELETE }
  gh release upload $Tag $File --repo $Owner/$Repo --clobber
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
$assets = $release.assets | Where-Object { $_.name -eq (Split-Path $File -Leaf) }
foreach ($a in $assets) { Invoke-RestMethod -Method Delete -Headers @{Authorization = "token $token"} -Uri $a.url }

$uploadUrl = $release.upload_url -replace '\{.*\}','' + "?name=$(Split-Path $File -Leaf)"
Invoke-RestMethod -Method Post -Uri $uploadUrl -Headers @{Authorization = "token $token"; 'Content-Type'='application/octet-stream'} -InFile $File -UseBasicParsing
Write-Output "Upload complete"
