<#
Upload one file to a Capsule machine.

    .\deploy\capsule\up.ps1 -Machine <name> -Source <local file> -Target <path relative to the remote home> [-TimeoutSec 900]

Line endings: a shell script written on Windows has CRLF; convert it to LF first or run
`sed -i 's/\r$//' file` on the machine before using it. Ends every tunnel afterwards, so do not run
two Capsule commands at once. The machine name is never recorded in this repository (D-017).
#>
param(
    [Parameter(Mandatory)][string]$Machine,
    [Parameter(Mandatory)][string]$Source,
    [Parameter(Mandatory)][string]$Target,
    [int]$TimeoutSec = 900,
    [string]$OutDir = (Join-Path $env:TEMP "capsule-out")
)
$bin = Join-Path $env:LOCALAPPDATA "Capsule-CLI\bin\capsule.exe"
New-Item -ItemType Directory -Force $OutDir | Out-Null
$o = Join-Path $OutDir "up-$Machine.out"
$e = Join-Path $OutDir "up-$Machine.err"
$p = Start-Process -FilePath $bin -ArgumentList @("scp", "upload", "--no-banner", "--unique", $Machine, "`"$Source`"", $Target) `
    -NoNewWindow -PassThru -RedirectStandardOutput $o -RedirectStandardError $e
if (-not $p.WaitForExit($TimeoutSec * 1000)) { Stop-Process -Id $p.Id -Force; "timed out after $TimeoutSec s" } else { "finished" }
& $bin session endall 2>&1 | Out-Null
Get-Content $o -Tail 3; Get-Content $e -Tail 3
