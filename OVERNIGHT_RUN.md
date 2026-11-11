# Overnight run — 2026-11-11 (Vietnam time)

The jobs run as detached processes and do not depend on an open chat turn.
The current GPU queue is serial:

1. Recover DPO training from full M1, then reload and evaluate M2 on the frozen
   200 RunBugRun cases and 164 HumanEvalPack tasks.
2. GRPO independently starts from the SAME merged M1. Measure reward variance
   on 30 train problems with 4 completions. If that scientific gate fails, try
   8 completions once and use matching generation/accumulation settings.
3. If variance passes, train/save/reload a 20-step GRPO pilot. Verify actual
   nonzero, finite LoRA updates before a fresh 200-step M3 run.
4. Reload and evaluate M3 using the same frozen evaluation protocol. Write
   comparisons including all completed checkpoints.
5. Run a five-problem dev-only v1/v2/v3 controller comparison on fixed M1.
   It is a small ablation, not a replacement headline benchmark.

## Live pointers

- `results/active-sft.json`: completed M0/M1 run and its final scores.
- `results/active-dpo.json`: current DPO recovery supervisor.
- `results/active-grpo.json`: current GRPO supervisor and dependency.
- `results/agent-controller-dev.queue.json`: queued v1/v2/v3 ablation.

Each pointer names a status file recording stage, process ID, logs and errors.
The old failed DPO run and interrupted GRPO variance attempt are preserved.

## DPO memory recovery

The first full DPO attempt failed at optimizer step 6 with CUDA allocator
fragmentation: 5.41 GiB allocated, 4.51 GiB reserved but unused, and an attempted
1.38 GiB allocation. The recovered process uses
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. Its chosen/rejected reference
log probabilities are cached on disk under a key incorporating data fingerprint,
M1-parent checksum, sequence limit and library versions. A second OOM triggers
one explicitly logged 1,536-token training fallback, filtering oversized pairs
rather than truncating targets. The evaluation token budget remains unchanged.

GRPO subprocesses use the same allocator setting. The supervisor has bounded
stage deadlines, one clean pilot/main recovery, a lower-memory pilot fallback,
and one evaluation retry. A failed DPO arm does not invalidate GRPO's M1 parent:
GRPO may proceed once the DPO worker releases the GPU. A failed reward-variance
or learning-update gate is recorded as `gate_failed`; training is not forced.
Hardware faults and unrecoverable failures remain possible and are logged.

## Completed M1 scores

| Benchmark | M0 pass@1 | M1 pass@1 | M0 agent@3 | M1 agent@3 |
|---|---:|---:|---:|---:|
| RunBugRun, 200 | 29.0% | 33.0% | 32.0% | 39.0% |
| HumanEvalPack, 164 | 55.5% | 53.7% | 62.2% | 63.4% |

These are measured v1 results. There is no overall improvement assumption;
HumanEvalPack first-attempt performance declined slightly.

## V3 controller

`agent/repair_v3.py` enforces at most 3 code-generation calls and 2 diagnosis
calls, each diagnosis capped at 96 tokens by default. Diagnosis output must be a
small JSON object containing fault, observed evidence and proposed change. Bad
JSON is ignored without another repair/diagnosis call. Hidden test inputs,
expected outputs and raw traceback messages are not provided to the model.

The next attempt starts from the best feedback-tested candidate. A regression
cannot replace that candidate. AST-equivalent failed patches are rejected before
execution and still consume a code-generation attempt; the trace explicitly
marks that no tests ran for the duplicate. A stuck retry can use fixed-seed
low-temperature sampling. The returned candidate is always the best tested one.
The first code-generation prompt remains identical to v1.

The controller ablation accounts for diagnosis tokens/time separately and prints
brief diagnoses, duplicate flags, scores and best-attempt IDs. No V3 quality gain
is claimed before the queued dev check finishes. SFT/DPO/GRPO are each trained
once; controller variants do not require separate post-training runs.
