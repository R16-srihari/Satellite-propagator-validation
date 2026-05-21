param(
  [string]$Owner = 'R16-srihari',
  [string]$Repo  = 'Satellite-propagator-validation',
  [string]$Tag   = 'stk-latest',
  [string]$File  = 'STK_input/Satellite1_Results.csv'
)

# Also publish the Satellite1.opm release asset when present
$Additional = 'STK_input/Satellite1.opm'

function Get-VersionedAssetName([string]$Path) {
  $leaf = Split-Path $Path -Leaf
  $base = [System.IO.Path]::GetFileNameWithoutExtension($leaf)
  $ext = [System.IO.Path]::GetExtension($leaf)
  $stamp = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssfffZ')
  $nonce = [Guid]::NewGuid().ToString('N').Substring(0, 8)
  return "$base-$stamp-$nonce$ext"
}

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

  $tempDir = Join-Path ([System.IO.Path]::GetTempPath()) ("stk-release-" + [Guid]::NewGuid().ToString('N'))
  New-Item -ItemType Directory -Path $tempDir | Out-Null

  try {
    foreach ($f in $filesToUpload) {
      $versionedName = Get-VersionedAssetName $f
      $stagedFile = Join-Path $tempDir $versionedName
      Copy-Item $f $stagedFile
      gh release upload $Tag $stagedFile --repo $Owner/$Repo
    }
  }
  finally {
    Remove-Item -Recurse -Force $tempDir -ErrorAction SilentlyContinue
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

# Upload files that exist
foreach ($f in @($File, $Additional)) {
  if (-not (Test-Path $f)) { continue }
  $versionedName = Get-VersionedAssetName $f
  $uploadUrl = ($release.upload_url -replace '\{.*\}','') + "?name=$versionedName"
  Invoke-RestMethod -Method Post -Uri $uploadUrl -Headers @{Authorization = "token $token"; 'Content-Type'='application/octet-stream'} -InFile $f -UseBasicParsing
}

Write-Output "Upload complete"
