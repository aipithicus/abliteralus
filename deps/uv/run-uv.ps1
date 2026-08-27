Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$python = & (Join-Path $PSScriptRoot 'resolve-repository-python.ps1')
$runner = Join-Path $PSScriptRoot 'run_uv.py'

& $python -I -B $runner @args
exit $LASTEXITCODE
