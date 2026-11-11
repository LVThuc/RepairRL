# Server-local grading migration — completed 2026-11-11

Remote experiments no longer depend on the original local computer. Supervisor services `repairrl-grader` (loopback18592) and `repairrl-grader-proxy` (existing loopback18591) serve all grading on the second server. The old reverse SSH grading tunnel and local grader were stopped. GRPO and step1200 evaluation advanced after cutover without trainer restart or optimizer loss. Brief SIGSTOP/SIGCONT pauses preserved their in-memory states during the endpoint switch.

## Why the original dependency existed

The remote Vast container blocks the user namespaces required by the original Bubblewrap executor. The original implementation used local Bubblewrap over a reverse SSH tunnel. The server CPU benchmark demonstrated a working alternative; migration now uses a minimal chroot, unprivileged UID/GID65534, seccomp blocking sockets/process creation/ptrace and resource limits. Trusted tests and expected outputs stay outside the candidate jail. HTTP services bind only to loopback.

## Runtime and verification

Candidate Python remains **3.14.4**. `scripts/build_grader_runtime.py` copied the original interpreter, standard library, and shared-library dependencies into a private runtime archive with file hashes. The server jail is `/workspace/repairrl-grader-runtime`; no host secrets or tests are copied inside it. The archive and manifest are also preserved locally under `/home/thao/Thuc/RepairRL-preservation/server-grader-migration/`.

`ServerSandbox.check()` verified UID65534 and denial of host-file access, sockets and forks. `scripts.validate_server_grader` passed all gold-pass/bug-fail checks on **184 tasks: 20 RunBugRun plus all164 HumanEvalPack tasks**, in118.4 seconds. This is runtime sanity validation, not a proof of identical outcomes for every generated program.

Tests SHA256: `949b6c5849e074a303f61bfd46f45b9da36a03e817e579824221f24f5a14369c`.
HumanEvalPack SHA256: `fe11a0994c92e295fbfa65bafc60deea66b43ddca3a91bc6c891d13f19b66cc7`.

## Execution provenance and comparisons

Isolation changes from Bubblewrap to chroot/seccomp, although candidate Python/files, tests, comparison logic and3-second timeout are preserved. New health responses truthfully identify the server backend. `results/round2/grader-migration.json` records cutover, paused process groups and evaluation prefixes. Step1200 RunBugRun had68 completed tasks at cutover; this run crosses backends. GRPO also crosses backends. Preserve that ledger with results. Existing protocol files describe the grader at process startup; the migration ledger supplies the subsequent change. Do not claim identical backend timing or a purely unchanged execution environment.

Final reporting permits comparison of the documented backends only when the migration validation passes and grader test hashes, Python identity and timeout match. This is an explicit, disclosed compatibility decision; it does not retroactively rewrite historical results.

If a partially evaluated run needs restarting, its old `.protocol.json` health may differ from the new endpoint. Preserve the old manifest and attach migration provenance before updating its grader identity for resumption. Do not silently delete traces or overwrite historical provenance.

## New computer

You may power off the original computer without stopping remote training/evaluation. Clone the source repository, configure SSH to the second server, then retrieve results/checkpoints with rsync. The original computer's ongoing backup/watch processes are optional copies, not dependencies. Remote disk remains ephemeral: do not destroy/recycle the server until artifacts are copied to persistent storage.

```bash
ssh -p 40401 root@125.240.239.50 'supervisorctl status repairrl-grader repairrl-grader-proxy repairrl-grpo repairrl-grpo-gpu0'
ssh -p 40401 root@125.240.239.50 'curl -fsS http://127.0.0.1:18591/health'
mkdir -p remote-artifacts
rsync -a -e 'ssh -p 40401' root@125.240.239.50:/workspace/RepairRL/results/round2/ remote-artifacts/
```

Keep the existing local preservation folder as well; historical round1 checkpoints are not all duplicated on the server. This migration moves the active grading dependency and retains all active inputs/outputs on the server; it is not a claim that every historical local archive was uploaded.
