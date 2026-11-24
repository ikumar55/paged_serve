# Working on a Capsule GPU machine from Windows

Helpers used for every GPU session in this project (`run.ps1`, `up.ps1`, `down.ps1`) and the
conventions that go with them. The machine name is not recorded in this repository (D-017); pass it
with `-Machine`. Log in first with `capsule auth login` (a browser step); `capsule list` should then
show the machines, and "Authentication error" means log in again.

## The pattern

1. **Write the command or script as a file** and upload it (`up.ps1`), converting to LF line endings.
   Everything the project puts on the machine lives under `~/pagedserve_work` with every cache
   redirected there by its `env.sh` (the cleanup duty: STATUS section 6, item 13).
2. **Launch long work detached**, one line, so a dropped connection does not stop it:
   `cd ~/pagedserve_work && sed -i 's/\r$//' job.sh ; nohup bash job.sh > job.log 2>&1 < /dev/null & sleep 5 ; echo launched`
3. **Poll its log** with `run.ps1` (a command file that greps the log for `== ... exit`, a `*-DONE`
   marker and `STAGE-FAILED`). Do not poll faster than every couple of minutes.
4. **When it is done, tar the results on the machine, download the archive, check the hash**
   (`down.ps1`), extract it into its own empty folder, **scan it for the machine's home path, user
   ids, tokens and the machine name** (dependency freezes embed the home path: redact it), copy only
   new files into `results/` (results are write-once), and commit.

## Quirks learned the hard way

- `capsule exec` breaks on double quotes: single quotes only, command in a file.
- Joined lines: never end a line with `&`; put a background launch and what follows on one line.
- `pgrep -f` / `pkill -f` patterns must be **anchored** (for example `'^bash job.sh'` or
  `'^[^ ]*python -m bench.machine_monitor'`), or they match the shell running them (the command
  text contains the pattern) and kill it or miscount.
- Every Capsule call ends all tunnels afterwards, so never run two at once; a long-running job on
  the machine is unaffected.
- "SSHRTC connection failed" is transient: end the sessions and retry; the machine is fine.
- Files that regenerate from the repository (the greedy-evaluation prompts) change when the repository
  changes: use the recorded ids (`--prompts-from`).
- Never run two GPU jobs at once (each script waits for any earlier one); start the machine monitor
  (`bench.machine_monitor`) for the whole session and run the audit before quoting a number.
- Do not delete `~/pagedserve_work` until the last GPU work is done (roadmap item 10); do not push to
  GitHub without the owner's say-so.
