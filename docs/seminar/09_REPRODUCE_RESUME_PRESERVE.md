# 09 — Reproduce, resume and preserve

## Authoritative local preservation storage

Per the owner's instruction, **local storage is sufficient and is the authoritative
backup**. Drive uploads were stopped. Do not use the abandoned partial cloud
upload as a restoration source.

Local preservation root:

```text
/home/thao/Thuc/RepairRL-preservation/
├── archives/                 # immutable tar.zst snapshots
├── manifests/                # file checksums, environment and inventory
├── base-model-M0/            # pinned untouched model copied out of HF cache
├── remote-live/
│   ├── results-round2/       # full remote mirror INCLUDING checkpoint-* states
│   ├── logs/                 # remote logs and environment/source snapshots
│   ├── supervisor-conf/      # relevant repairrl service definitions
│   └── supervisor-scripts/   # relevant repairrl launch wrappers
└── demo-packet.json          # private, sanitized case replay packet
```

The [storage index](../../artifacts/manifests/storage-index.json) gives actual
archive paths, byte counts and SHA256 hashes. The [local file inventory](../../artifacts/manifests/local-inventory.json)
accounts for historical model/data/log files. Later remote snapshots are inventoried
separately because they continue to grow. No archive is implied by a planned filename;
check its recorded completion status and checksum.

## Restore the project

Clone `https://github.com/LVThuc/RepairRL`. Obtain the preservation folder through
the owner's local disk, external disk or authenticated file transfer; there is
no public artifact-download service. Verify the indexed archive hashes before extraction.

```bash
sha256sum /path/to/preservation/archives/foundation-data-checkpoints.tar.zst
# Compare the output with artifacts/manifests/storage-index.json.

# Extract into a FRESH working directory; do not overwrite an active experiment.
mkdir -p /path/to/restored-RepairRL
tar --zstd -xf /path/to/preservation/archives/foundation-data-checkpoints.tar.zst \
  -C /path/to/restored-RepairRL
```

The foundation snapshot contains `data/`, `checkpoints/` and the upstream vendor
checkout. A separate handoff/runtime archive supplies code, logs/results, service
definitions, environment details and the private replay packet. The M0 archive
provides the untouched pinned base model for offline work. Consult the index for
which additional live-state archive was completed at handoff time.

Public source dependencies: Qwen revision
`2e1fd397ee46e1388853d2af2c993145b0f1098a`, RunBugRun v2 and its recorded release
digest, HumanEvalPack's recorded source revision, and the pinned Python libraries.
Model/dataset public availability does not remove the value of the local copies.

## Rebuild the environment

Original training: Python 3.11.16 and the checked-in lockfiles. The remote run uses
Python 3.12.15 and torch 2.8.0+cu128 with the same TRL/Transformers/PEFT versions;
its complete freeze is preserved. Use CUDA wheels supporting the target GPU;
actual 5070 tensor and 4-bit backward smoke tests passed with CUDA 12.8.
Do not replace a host-injected NVIDIA driver while restoring a container.

```bash
uv venv .venv --python 3.11
# Use the actual archived lock/freeze for the selected experiment, not a generic latest install.
uv pip install --python .venv/bin/python -r requirements.lock
python -m scripts.doctor
```

System candidate execution used Python 3.14.4, Bubblewrap, and the resource limits
in `agent/sandbox.py`. That runtime is separate from the training interpreter.
Restore the grader and run its smoke/gold checks before trusting new scores.

## Adapter versus resumable checkpoint

| Artifact | What it supports | What it lacks |
|---|---|---|
| Final `adapter_model.safetensors` + config/tokenizer | Inference or a fresh training warm start | Optimizer momentum, scheduler, RNG and progress |
| Merged M1 model | Inference and a frozen parent for new branches | Original optimizer state; exact unmerged QLoRA equivalence |
| Complete `checkpoint-*` directory | Trainer continuation under the matching setup | A guarantee of bitwise reproducibility across different hardware/software |
| Partial/in-progress checkpoint directory | Nothing guaranteed yet | May lack complete optimizer/RNG/state files |

Required full-state components: adapter weights/config, `optimizer.pt`,
`scheduler.pt`, `rng_state.pth`, `trainer_state.json`, `training_args.bin`, tokenizer
files and applicable `scaler.pt`. Inventory validation checks file presence and
Torch ZIP CRCs without executing pickle data. Only load trusted owned states.

## Historical full trainer states available

| Arm | Earlier state | Final-budget state | Original target |
|---|---|---|---:|
| SFT M1 | `checkpoints/M1/checkpoint-200` | `checkpoint-250` | 250 updates |
| Weak DPO M2 | `checkpoints/M2/checkpoint-100` | `checkpoint-125` | 125 updates |
| GRPO M3 | `checkpoints/M3/checkpoint-175` | `checkpoint-200` | 200 updates |
| GRPO preflight | `checkpoints/grpo-preflight/M3/checkpoint-20` | Same | 20 updates |
| Pilot arms | `checkpoints/pilot/M*/checkpoint-*` | Small pilot only | Inspect individual trainer_state.json |

The earlier states support continuation of the original unfinished budget. A
state already at its target is physically complete but has no remaining planned
updates. Extending it changes the training/schedule budget and must be reported
as a new experiment.

## New original-arm resume utility

The original training CLIs did not expose a resume flag. The handoff adds
[`training/resume.py`](../../training/resume.py), which validates state/data/parent
identity and delegates restoration to the pinned HF Trainer. It reuses original
trainer construction rather than silently replacing the objective.

```bash
# Structural/data/parent validation; does NOT run a GPU update.
python -m training.resume \
  --metadata checkpoints/M1/repairrl-training.json \
  --checkpoint checkpoints/M1/checkpoint-200 \
  --output checkpoints/resumed-M1 --validate-only

# Actual continuation to the original 250-update budget, when the GPU is free.
CUDA_VISIBLE_DEVICES=0 python -m training.resume \
  --metadata checkpoints/M1/repairrl-training.json \
  --checkpoint checkpoints/M1/checkpoint-200 \
  --output checkpoints/resumed-M1
```

Substitute M2/100 and M3/175 with their matching metadata for the other original
arms. Validation was executed successfully on those three real earlier states.
An actual GPU resume was **not** run during handoff, to avoid interrupting current
experiments. The inventory is not a substitute for a future GPU resume smoke test.

Keep the original M1 parent at the path expected by the metadata. The original
local merge and the second-server merge differ. The utility refuses a parent-hash
mismatch. For original GRPO, also restore its recorded variance report and a
working local Bubblewrap grader. Do not substitute the new dense reward.

## Second-round resumption

`scripts/round2_train.py --resume` locates the last full HF trainer checkpoint in
its experiment output directory. Reuse the exact recorded kind, bank, LR, beta,
precision, epochs/step target and output path from `experiment.json` and the
supervisor wrapper. Do not treat an `snapshots/step-*` adapter as a full state.

The running GRPO process currently uses an earlier trainer revision, archived at
`docs/seminar/sources/round2_train_executed_grpo.py` and verified by its recorded
hash. Restore that revision for the closest continuation of this specific run.
Code under `experimental/async_dev/` was not launched and is not its resume path.

Before the first periodic full checkpoint, newer optimization progress exists
only in process memory. It cannot be recovered merely by copying the step-0
adapter. The non-deleting full-state mirror captures completed remote checkpoints
as they appear and retains them after remote pruning. Inspect the latest live
inventory to determine the last recoverable step, not the last printed progress bar.

## Rerun evaluations

Original protocol example, using the original parent and adapter:

```bash
python -m evaluation.benchmark --arm M1 \
  --model Qwen/Qwen2.5-Coder-1.5B-Instruct --adapter checkpoints/M1 \
  --benchmark runbugrun --data data/prepared/scale-verified/test.jsonl \
  --tests data/prepared/scale-verified/tests.json \
  --max-new-tokens 512 --max-context 2048 --seed 42 --timeout 3 \
  --output results/reproduced-M1-v1-runbugrun.jsonl
```

HumanEvalPack needs its recorded gold gate and dataset path. Second-round CLI
is `evaluation.round2` and uses the private grading service, FP16 flag, 768/4096
budget and named controller. Outputs are fresh or strictly protocol-matched
resumptions; never overwrite historical traces.

## Preservation boundaries

No credentials, private SSH keys, tokens, virtualenv binaries or unrelated host
secrets belong in Git or archives. The virtualenv is represented by interpreter
versions, lockfiles and freezes. Raw published datasets, hidden suites and full
example traces stay outside ordinary Git. Local archives are the handoff medium;
copy them to another local disk if the original machine itself is being retired.
