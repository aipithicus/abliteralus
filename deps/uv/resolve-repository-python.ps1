Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repository = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$pythonRoot = [IO.Path]::GetFullPath((Join-Path $repository 'deps\python'))
$aliasPath = Join-Path $pythonRoot 'cpython-3.12-windows-x86_64-none'
$expectedTarget = [IO.Path]::GetFullPath(
    (Join-Path $pythonRoot 'cpython-3.12.6-windows-x86_64-none')
)

if (-not (Test-Path -LiteralPath $aliasPath -PathType Container)) {
    throw "Repository Python alias is unavailable: $aliasPath"
}
$alias = Get-Item -LiteralPath $aliasPath -Force
if ($alias.LinkType -ne 'Junction') {
    throw "Repository Python alias must be a junction: $aliasPath"
}
$resolvedTarget = [IO.Path]::GetFullPath([string]$alias.ResolvedTarget)
if (-not $resolvedTarget.Equals($expectedTarget, [StringComparison]::OrdinalIgnoreCase)) {
    throw "Repository Python alias escapes or selects an undeclared runtime: $resolvedTarget"
}

$python = Join-Path $aliasPath 'python.exe'
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Repository Python executable is unavailable: $python"
}
$python
