<#
Offline checks of the deployment files on Windows: the same three steps as deploy/validate.sh
(hadolint, `kubectl kustomize` + kubeconform, the deployment tests). No Docker, cluster or GPU needed.

    .\deploy\validate.ps1 [-Tools C:\Users\you\tools\bin] [-Python <path to the project's python.exe>]

The tools are single executables (hadolint, kubectl, kubeconform); download them from their GitHub
releases and verify the published checksums. This script only looks for them in -Tools and on PATH.
#>
param(
    [string]$Tools = (Join-Path $env:USERPROFILE "tools\bin"),
    [string]$Python = "python",
    [string]$KubeVersion = "1.34.0"
)
Set-Location (Join-Path $PSScriptRoot "..")
$status = 0
function Find-Tool($name) {
    $local = Join-Path $Tools "$name.exe"
    if (Test-Path $local) { return $local }
    $found = Get-Command $name -ErrorAction SilentlyContinue
    if ($found) { return $found.Source }
    return $null
}

"`n== hadolint"
$hadolint = Find-Tool "hadolint"
if ($hadolint) {
    & $hadolint deploy/docker/Dockerfile.engine deploy/docker/Dockerfile.router deploy/docker/Dockerfile.bench
    if ($LASTEXITCODE -ne 0) { $status = 1 }
} else { "hadolint not found: skipped" }

"`n== kubectl kustomize + kubeconform (Kubernetes $KubeVersion schemas)"
$kubectl = Find-Tool "kubectl"; $kubeconform = Find-Tool "kubeconform"
if ($kubectl -and $kubeconform) {
    foreach ($dir in "deploy/k8s/base", "deploy/k8s/overlays/mock", "deploy/k8s/overlays/kernel-graphs") {
        Write-Host -NoNewline "${dir}: "
        & $kubectl kustomize $dir | & $kubeconform -strict -summary -kubernetes-version $KubeVersion -
        if ($LASTEXITCODE -ne 0) { $status = 1 }
    }
} else { "kubectl or kubeconform not found: skipped" }

"`n== deployment tests"
& $Python -m pytest tests/test_deploy_files.py -q
if ($LASTEXITCODE -ne 0) { $status = 1 }
exit $status
