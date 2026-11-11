# 03 — Actual post-training implementation

This document separates the completed original runs from the second-round jobs.
Full code links are in [CODE_INDEX.md](generated/CODE_INDEX.md); exact model/data
identities are preserved in per-run JSON metadata and the artifact inventory.

## Shared model and prompt

Base: `Qwen/Qwen2.5-Coder-1.5B-Instruct`, revision
`2e1fd397ee46e1388853d2af2c993145b0f1098a`. Architecture: 28 layers, hidden size
1,536, MLP intermediate size 8,960, 12 attention heads, 2 KV heads, vocabulary
151,936 and tied embeddings.

[`repairrl/common.py::messages`](../../repairrl/common.py) builds the same initial
repair prompt across arms:

````text
SYSTEM: You are a Python code repair assistant. Return ONLY the full corrected Python 3 program.
USER: Fix the provided program so it satisfies the problem specification.
Problem:
{statement}
Buggy program:
```python
{buggy_code}
```
Return corrected Python code, not explanations.
````

The inner Python fence above is part of the user message. In slides, render the
whole prompt as a text panel rather than nesting Markdown fences literally.
Qwen's tokenizer applies the chat template and adds the assistant-generation
prefix. No gold correction or hidden tests enter the input.

## Completed run: precision and adapters

[`repairrl/model.py::load_model`](../../repairrl/model.py) and
[`training/common.py`](../../training/common.py) implement:

```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)
```

The base is frozen. LoRA rank 16/alpha 32 targets `q_proj`, `k_proj`, `v_proj`,
`o_proj`, `gate_proj`, `up_proj`, `down_proj`: 18,464,768 trainable parameters.
Configured dropout is 0.05; DPO's trainer disables dropout by default, so do not
describe 0.05 as effective DPO training dropout.

Original training used FP16, SDPA attention and gradient checkpointing with
`use_reentrant=False`. KV caching was disabled during training. Shared optimizer:
torch AdamW, betas 0.9/0.999, epsilon 1e-8, weight decay 0, gradient norm cap 1,
linear LR decay, no warmup. Logging occurred every five updates. No training-time
dev-loss evaluation or early-stopping selection was used in round 1.

The original environment was Python 3.11.16, torch 2.8.0, Transformers 4.56.2,
TRL 0.23.1, PEFT 0.17.1, bitsandbytes 0.47.0, datasets 4.1.1 and Accelerate 1.10.1.
See `requirements.lock`, `uv.lock` and archived freezes. A new current package
environment is not a substitute for the original run's environment record.

## M1: response-only SFT

Implementation: [`training/train_sft.py`](../../training/train_sft.py).
Config: [`configs/full-sft.yaml`](../../configs/full-sft.yaml).

```python
prompt = tokenizer.apply_chat_template(
    messages(row["statement"], row["buggy_code"]),
    tokenize=False, add_generation_prompt=True,
)
completion = row["fixed_code"] + tokenizer.eos_token
# SFTConfig: completion_only_loss=True, packing=False
```

Teacher-forced causal cross-entropy applies to completion tokens; system/user
tokens and padding are masked with `-100`. A batch-mask smoke check verifies
that both masked and trainable labels exist. Targets are retained whole and
overlength rows are filtered, not truncated.

| Setting | Actual M1 |
|---|---|
| Data | 2,000 verified published pairs |
| Epochs / optimizer steps | 1 / 250 |
| Microbatch / accumulation / effective batch | 1 / 8 / 8 |
| Initial LR / sequence limit | 1e-4 / 2,048 |
| Training + save | 1,560.8 seconds = 26.0 minutes |
| Peak allocated VRAM | 6.46 decimal GB |
| Mean train loss | 0.06527 |
| First / last logged token accuracy | 96.36% / 98.25% |

These accuracy values are teacher-forced token matches on training examples,
not executable repair success. Copy-heavy code and small bug edits make that
distinction especially important. M1 was trained freshly from M0, not the pilot.

## Branch root and reference policies

[`scripts/merge_sft.py`](../../scripts/merge_sft.py) loads the original FP16 base
on CPU, attaches M1, calls `merge_and_unload(safe_merge=True)`, and saves a full
merged model. M2 and M3 independently load it with fresh LoRA adapters.

For PEFT DPO/GRPO, disabling the new adapter exposes the frozen merged M1 policy.
This avoids keeping a second full reference model on the GPU. Original M2 and M3
record the same merge-manifest hash:
`28d4643703eb754f6261cb5676a77bcd8eeb2f3d789125cfde5831923926d943`.

Merging an adapter into unquantized base weights and requantizing is not
bit-identical to the original unmerged QLoRA policy. The second-server merge also
differs numerically from the original local merge. Its exact weights have been
copied locally as `checkpoints/M1-merged-round2`; preserve both parents and use the
appropriate one for each branch.

## M2: original weak-negative DPO

Implementation: [`training/train_dpo.py`](../../training/train_dpo.py).
Reference cache: [`scripts/train_dpo_cached.py`](../../scripts/train_dpo_cached.py).
Config: [`configs/full-dpo.yaml`](../../configs/full-dpo.yaml), with row limit 1,000.

```text
prompt   = source problem + source buggy program
chosen   = source fixed program + EOS
rejected = source buggy program + EOS
```

For chosen/rejected log-ratios relative to M1, the sigmoid objective minimizes
`-log sigmoid(beta * (chosen_log_ratio - rejected_log_ratio))`. The reference
matches the SFT root; it is not the untouched base. Chosen/rejected are full
program responses, not patches or diagnosis traces.

| Setting | Actual M2 |
|---|---|
| Pairs / distinct problems | 1,000 / 1,000 |
| Epochs / updates | 1 / 125 |
| Microbatch / accumulation | 1 / 8 |
| LR / beta / max sequence | 5e-6 / 0.1 / 2,048 |
| Additional supervised loss | None |
| Reference probabilities | Precomputed, persistent lineage-keyed cache |
| Successful run time | 2,201.8 seconds = 36.7 minutes |
| Optimizer-loop time | 1,497.9 seconds; excludes reference precomputation |
| Peak allocated VRAM | 10.53 GB |
| Average / final logged loss | 0.5193 / 0.3803 |
| Final logged preference accuracy | 100% |

The rejected response simply repeats the known buggy input. This can teach
avoidance of copying without teaching discrimination among plausible repairs.
It is explicitly a weak preference construction. The original DPO arm did not
use generated hard negatives and did not outperform M1 on RunBugRun.

An initial CUDA OOM at update 6 was recovered with
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. The final run kept context
2,048 and all 1,000 rows; the supervisor's optional 1,536 fallback was not used.
Failed logs and the recovery run are preserved separately.

## M3: original outcome-reward GRPO

Implementation: [`training/train_grpo.py`](../../training/train_grpo.py).
Reward: [`evaluation/reward.py`](../../evaluation/reward.py).
Config: [`configs/full-grpo.yaml`](../../configs/full-grpo.yaml), with the resolved
run config supplying its actual variance-report path.

The dataset carries prompts and problem IDs. The trusted grader, not the model,
resolves tests. For each prompt, four generated programs receive execution rewards.
The trainer computes group-relative advantages and applies the GRPO policy loss
with a KL reference term. TRL 0.23.1's default configured loss is `dapo`, with
group reward scaling, clipping epsilon 0.2 and one optimization iteration per rollout.

```python
if not grade.valid or any_fatal_execution_failure:
    reward = 0.0
else:
    reward = grade.fraction + (0.05 if grade.success else 0.0)
```

| Setting | Actual M3 |
|---|---|
| Available training rows | 2,000 prompt-only records |
| Main updates / sampled completions | 200 / 800 |
| Microbatch / accumulation / generations | 1 / 4 / 4 |
| LR / KL beta | 5e-6 / 0.04 |
| Prompt / generated-code limits | 1,536 / 512 tokens |
| Sampling | T=0.8, top-p=0.8, top-k=20, repetition penalty=1.1 |
| Main run time / peak allocated VRAM | 3,349.8 seconds / 3.97 GB |
| Variance preflight | 13/30 groups mixed; threshold 20% |
| Separate pilot | 20 updates, saved/reloaded; main restarted from fresh M1 |

200 updates processed about 0.1 epoch in this generation-batch configuration,
not all 2,000 prompts once. The main run had 86/200 mixed-reward groups; 114 groups
had no relative reward discrimination. KL regularization can still contribute;
do not claim every zero-variance group has identically zero total gradient.

## Round 2: what is actually running or queued

### Hard-negative bank and DPO tuning

[`scripts/round2_fast_rollouts.py`](../../scripts/round2_fast_rollouts.py) samples
one greedy and three stochastic M1 repairs per training row. It uses FP16 inference,
eight prompts per group, and batches 24 stochastic sequences. Sampling is T=0.8,
top-p=0.95, top-k=50, repetition penalty=1.1, up to 768 code tokens. The full source pool is 2,000 rows. The owner revised collection to 1,000 processed source examples; a supervised cutoff freezes that prefix and preserves any batch overshoot separately. Pair count is a measured yield, not a promised 800 high-quality examples.

Candidates must parse, exit normally, fail at least one output comparison, differ
from the original bug/gold AST, and pass duplicate checks. Gold is rechecked for
rows yielding negatives. IDs, seeds, sampler versions, outputs and grades preserve
same-prompt provenance. Early NF4 warmup rollouts are stored separately; the FP16
bank records its sequential-to-multi-prompt sampler transition.

[`scripts/round2_jobs.py`](../../scripts/round2_jobs.py) queues three three-epoch DPO
trials after bank completion: LR 3e-6/beta 0.1; LR 1e-5/beta 0.1; and LR 3e-6/beta 0.1 with `rpo_alpha=0.2`. It monitors dev repair success every approximately half epoch.
FP16 LoRA is attempted; a logged OOM can trigger a fresh NF4 fallback trial. Those
are queued settings, not completed training outcomes.

### Longer GRPO

The active remote job uses FP16 LoRA on the second-server M1 merge, max 2,000
updates, four generations/group, LR 5e-6, KL beta 0.04, 3% warmup, prompt/code
limits 1,536/768, sampling T=0.8/top-p=0.95/top-k=50. Dev scoring uses 60 fixed
distinct problems every 200 updates. At least 800 updates precede plateau-based
early stopping. A dev-triggered alternative correctness-reward trial is queued
if the dense trial fails to improve over its parent on dev.

[`evaluation/reward_v2.py`](../../evaluation/reward_v2.py) calculates:

```text
dense = 0.85 * passed_fraction + 0.10 * normal_exit_fraction + 0.05 * all_pass
correctness alternative = passed_fraction + 0.05 * all_pass
```

Invalid code receives zero. Partial credit survives failures on other tests.
The new variance check had 20/30 mixed groups; sampling, precision and task
selection also changed, so this is not a controlled reward-only improvement.

Current dev scoring is **synchronous on the assigned remote GPU**. The async
prototype was deferred to avoid interrupting active experiments. The exact trainer
source loaded by the running GRPO process was reconstructed and verified against
its metadata hash; see [the archived source](sources/round2_train_executed_grpo.py).
Do not assume the latest on-disk trainer produced every active run.

[`scripts/round2_finish.py`](../../scripts/round2_finish.py) screens snapshots by dev,
reranks the top three on all 147 distinct dev problems at FP16 inference precision,
and only then evaluates the selected adapter. This is model selection, not an
assertion that the selected model beats M1. Check live status for completion.

### DPO stop decision (2026-11-11)

The user stopped the round-2 DPO queue near the end of its first configuration. Optimizer checkpoints through step 396 are preserved; further learning-rate and RPO configurations were cancelled. The saved dev-screen winner is step 66 (21/60), compared with the M1 parent at 20/60. Later screened scores were 20/60 at steps 198 and 264, and 19/60 at step 330. This is late dev degradation, consistent with overfitting or preference/execution objective mismatch; the cause has not been established, and this does not prove DPO generally worsens repair.

`scripts/round2_eval_selected_dpo.py` freezes that existing 60-task-dev selection and evaluates it on the two held-out benchmarks under both existing controllers. The originally planned top-three reranking on all 147 dev problems is skipped for this user-requested evaluation. Its selection metadata discloses this change. No held-out result is used to pick the snapshot. Evaluation runs on remote GPU 0 under `repairrl-dpo-eval`; the training service is stopped with autostart disabled. GRPO continues on GPU 1. Step 66 is an inference adapter; the later optimizer checkpoints contain resumable trainer states. Do not conflate their resume capabilities.
