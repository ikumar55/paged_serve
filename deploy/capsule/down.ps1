<#
Download one file from a Capsule machine and print its size and SHA-256.

    .\deploy\capsule\down.ps1 -Machine <name> -Remote <path relative to the remote home> -Local <local file> [-TimeoutSec 600]

To bring back many files, tar them on the machine first (`tar czf ~/pagedserve_work/tmp/x.tgz ...`,
print its `sha256sum` there), download the archive, and compare the two hashes. A connection
failure ("SSHRTC connection failed") is transient: it prints "no file"; run it again.
Do not copy files off the machine while a measurement is running (the copy is load too).
Scan what you download for the machine's home path and user id before committing it.
#>
param(
    [Parameter(Mandatory)][string]$Machine,
    [Parameter(Mandatory)][string]$Remote,
    [Parameter(Mandatory)][string]$Local,
    [int]$TimeoutSec = 600,
    [string]$OutDir = (Join-Path $env:TEMP "capsule-out")
)
$bin = Join-Path $env:LOCALAPPDATA "Capsule-CLI\bin\capsule.exe"
New-Item -ItemType Directory -Force $OutDir | Out-Null
$o = Join-Path $OutDir "down-$Machine.out"
$e = Join-Path $OutDir "down-$Machine.err"
$p = Start-Process -FilePath $bin -ArgumentList @("scp", "download", "--no-banner", "--unique", $Machine, $Remote, "`"$Local`"") `
    -NoNewWindow -PassThru -RedirectStandardOutput $o -RedirectStandardError $e
if (-not $p.WaitForExit($TimeoutSec * 1000)) { Stop-Process -Id $p.Id -Force; "timed out after $TimeoutSec s" } else { "finished" }
& $bin session endall 2>&1 | Out-Null
if (Test-Path $Local) {
    "size: " + (Get-Item $Local).Length
    "sha256: " + (Get-FileHash $Local -Algorithm SHA256).Hash.ToLower()
} else {
    "no file"
    Get-Content $o -Tail 3
}
