<#
Run a shell command (kept in a file) on a Capsule machine and capture its output.

    .\deploy\capsule\run.ps1 -Machine <name> -CmdFile cmd.sh [-TimeoutSec 120] [-OutDir <dir>]

Why a file: `capsule exec` breaks on double quotes inside the command, so the command lives in a
file that uses single quotes only. Blank lines are dropped and the lines are joined with "; ", so
do not end a line with `&` (a background launch must be on one line with what follows it:
`nohup ... & sleep 5 ; echo started`).

`capsule exec` does not exit by itself: this waits up to -TimeoutSec, kills it if needed and then
ends every tunnel (`capsule session endall`), so do not run two of these at once.

Output: <OutDir>\<Machine>.out and .err (default OutDir: $env:TEMP\capsule-out). The machine name
is never recorded in this repository (D-017): pass it on the command line.
#>
param(
    [Parameter(Mandatory)][string]$Machine,
    [Parameter(Mandatory)][string]$CmdFile,
    [int]$TimeoutSec = 120,
    [string]$OutDir = (Join-Path $env:TEMP "capsule-out")
)
$bin = Join-Path $env:LOCALAPPDATA "Capsule-CLI\bin\capsule.exe"
New-Item -ItemType Directory -Force $OutDir | Out-Null
$cmd = ((Get-Content $CmdFile -Raw) -replace "`r", "" -split "`n" | Where-Object { $_.Trim() -ne "" }) -join "; "
$o = Join-Path $OutDir "$Machine.out"
$e = Join-Path $OutDir "$Machine.err"
Remove-Item $o, $e -ErrorAction SilentlyContinue
$p = Start-Process -FilePath $bin -ArgumentList @("exec", "--no-banner", "--unique", $Machine, "`"$cmd`"") `
    -NoNewWindow -PassThru -RedirectStandardOutput $o -RedirectStandardError $e
if (-not $p.WaitForExit($TimeoutSec * 1000)) {
    Stop-Process -Id $p.Id -Force
    "[$Machine] timed out after $TimeoutSec s (output may be partial)"
} else {
    "[$Machine] finished"
}
& $bin session endall 2>&1 | Out-Null
