[CmdletBinding()]
param(
    [switch]$Force
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repository = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$scratchTemp = Join-Path $repository '.scratch\temp\uv-restore'
$scratchCache = Join-Path $repository '.scratch\cache\uv'
[IO.Directory]::CreateDirectory($scratchTemp) | Out-Null
[IO.Directory]::CreateDirectory($scratchCache) | Out-Null

$env:TEMP = $scratchTemp
$env:TMP = $scratchTemp
$env:TMPDIR = $scratchTemp
$env:UV_CACHE_DIR = $scratchCache
$env:UV_PYTHON_DOWNLOADS = 'never'

$python = & (Join-Path $PSScriptRoot 'resolve-repository-python.ps1')

$arguments = @((Join-Path $PSScriptRoot 'restore_uv.py'))
if ($Force) {
    $arguments += '--force'
}
& $python -I -B @arguments
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
