# 10 — Ownership, active operations and updates

## What is finished versus still running

The original four-arm training/evaluation, original five-case controller check,
and the new 40-case controller dev ablation are complete. The large hard-negative
bank, longer GRPO, queued DPO trials and second-round headline evaluation are
ongoing or pending. [LIVE_STATUS.md](generated/LIVE_STATUS.md) is timestamped;
regenerate it before preparing the final deck.

The second-round controller ablation did not improve repair success. Preserve
that result and do not present V3.1's engineering features as a measured win.

## Remote server and service topology

Only the second server is used: SSH port 40401 on `125.240.239.50`, workspace
`/workspace/RepairRL`, Python `/venv/main/bin/python`. No credentials are stored in
this document; use the owner's authorized SSH key. The first server is obsolete.

The operating guide `/etc/vast-agents-guide.md` was read before setup. Services
are managed by supervisor, not an unmonitored detached shell.

| Service | GPU | Current responsibility |
|---|---:|---|
| `repairrl-rollouts` | 0 | Full M1 train-only rollout bank; current FP16 multi-prompt sampler |
| `repairrl-dpo` | 0 | Wait for bank, train/tune DPO trials, dev rerank, evaluate selected adapter |
| `repairrl-grpo` | 1 | Dense-reward GRPO with intermediate synchronous dev checks |
| `repairrl-grpo-followup` | 1 | Wait for dense run; dev-triggered alternative; select/evaluate |

`RUNNING` in supervisor is a process state, not proof of an optimizer update.
Waiting queues also appear RUNNING. Check their JSON state and actual logs.

```bash
ssh -p 40401 root@125.240.239.50 \
  'supervisorctl status repairrl-rollouts repairrl-grpo repairrl-dpo repairrl-grpo-followup'
```

Remote logs: `/workspace/repairrl-logs/`. Results/states:
`/workspace/RepairRL/results/round2/`. The container's filesystem is not backed by
a persistent host volume; stopping and destroying are different operations.
No stop/destroy command is part of this handoff.

## Local dependencies that must remain available

The private grading service binds `127.0.0.1:18591`. Its SSH reverse tunnel exposes
the same loopback port to remote trainers; it does not expose a public grading
API. Candidate execution still happens inside local Bubblewrap. Both local
service and tunnel are required by remote rewards/evaluation.

Local monitoring PIDs/logs are under `results/round2/`. The controller dev runner
finishes before the local baseline queue uses the GPU. Artifact-sync and report
workers are separate CPU/network jobs. They do not mutate model weights.

Two backups have different roles:

1. `scripts/round2_watch.py` mirrors analysis-friendly results and lightweight
   snapshots, excluding optimizer checkpoint directories.
2. `scripts/preserve_project.py remote-sync` mirrors **full trainer states** to the
   local preservation root without `--delete`. This second copy is essential for
   genuine resumption and retention after remote pruning.

## Source provenance and deferred code

The running GRPO process imported its trainer before later local edits. Its
`experiment.json` records trainer hash
`dd2fed960291e1fbd93390c64821372c1b0817a5b68175cd884d01978e4ea7e3`.
The exact source was reconstructed and hash-verified, not inferred from current
HEAD. Current remote on-disk source and executed-source archives are both preserved.

Canonical local training/evaluation files were aligned with remote on-disk code
for the handoff. The unlaunched async prototype was moved under `experimental/`.
No active trainer or model-evaluation process was restarted for that change.

The earlier local baseline coordinator was launched before the exact-parent path
update. Its output must be checked by recorded `protocol.model`, not assumed from
the current launcher source. If it used the original local merge for round-2 M1,
retain those diagnostic results and rerun M1 with `checkpoints/M1-merged-round2`
before an exact-parent headline comparison. This is a launch-provenance issue;
do not silently merge incompatible rows.

## How documentation stays current

```bash
python -m scripts.seminar_snapshot --watch
```

It exports aggregate immutable round-1 evidence, current dev/queue state and a
function/hash index. It does not export hidden tests, dataset statements or full
candidate programs to public docs. Rebuild figures with the isolated docs
environment when completed-result inputs change. Regenerate the artifact
inventory after newly completed checkpoints are mirrored.

GitHub publication is a separate authenticated step. The connected GitHub app
can publish commits now. The machine's existing SSH identity is a different
account with read-only repository access, so unattended CLI pushes are not
available unless the owner grants write access or configures a project deploy
key. Local refresh/retention continues regardless. Never store a token/key in
the repository to bypass that limitation.

## Practical finalization checklist

- Confirm every active service has reached completed/failed/gate-failed, rather
  than assuming an idle GPU means success.
- Preserve failures and partial runs. A failed scientific gate is a result, not
  a reason to relabel an easier fallback as the original experiment.
- Verify selected models came from dev ranking; retain all intermediate scores.
- Check common task IDs, tests, precision, code/diagnosis budgets and parent identity.
- Separate old NF4/512-token results from new FP16/768-token results.
- Include both controller tables and all predefined benchmark tasks.
- Confirm optimizer/RNG/scheduler/scaler artifacts are complete before calling a state resumable.
- Verify local archive checksums and extract into a fresh directory for restoration testing.
- Commit only source, aggregate evidence, figures and manifests; keep large/private payloads local.
- Publish a new documentation commit after new results complete; record its commit and timestamp.

## Remaining work a presenter must not hide

At handoff time, new training/evaluation results may still be pending. A real GPU
resume validation was not performed while current experiments occupied the GPUs.
The new controller had no measured dev success gain. Uncertainty analysis for the
new full comparison requires its completed traces. These limitations belong in
the final deck if they remain unresolved.
