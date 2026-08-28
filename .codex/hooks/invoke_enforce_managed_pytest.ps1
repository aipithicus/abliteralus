Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$payload = [Console]::In.ReadToEnd()
$repositoryText = & git rev-parse --show-toplevel
if ($LASTEXITCODE -ne 0) {
    throw 'Cannot resolve the repository root for the managed-pytest hook'
}
$repository = [IO.Path]::GetFullPath(($repositoryText | Select-Object -First 1).Trim())
$resolver = Join-Path $repository 'deps\uv\resolve-repository-python.ps1'
$hook = Join-Path $repository '.codex\hooks\enforce_managed_pytest.py'

$python = & $resolver
if ($LASTEXITCODE -ne 0) {
    throw 'Cannot resolve the repository Python for the managed-pytest hook'
}

$payload | & $python -I -B $hook
exit $LASTEXITCODE
