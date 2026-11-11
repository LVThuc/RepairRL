# RepairRL — detailed project overview

RepairRL is a practical experiment in **post-training a small language model to
repair Python programs**, followed by test-guided retries at inference time.
Its main question is whether supervised learning, preference optimization or
execution-reward RL improves real repair success, and whether a debugging loop
adds further gains. The initial experiments are complete and gave mixed results;
stronger follow-up experiments are running.

This overview describes the actual implementation and measurements. More detailed
code references, figures and seminar notes are linked at the end.

## 1. What the system does

Input is a task description and a buggy Python program. The model returns the
**complete corrected program**, not a patch or an explanation. A trusted grader
executes it against externally maintained tests. If it fails, the agent can use
sanitized feedback to generate another program, with at most three code attempts.

The main training/evaluation domain is standalone competitive-programming
solutions using stdin/stdout. HumanEvalPack supplies a separate function-repair
evaluation. The system does not navigate arbitrary repositories or solve
multi-file software-engineering issues.

There are two distinct interventions:

- **Post-training changes weights:** SFT, DPO and GRPO produce different adapters.
- **Agent control changes inference:** feedback, diagnosis, candidate selection
  and retries operate without updating weights.

## 2. Model architecture and experimental branches

The base is `Qwen/Qwen2.5-Coder-1.5B-Instruct`, pinned at revision
`2e1fd397ee46e1388853d2af2c993145b0f1098a`.

```text
M0: untouched Qwen 1.5B
 └── SFT → M1
            ├── fresh DPO adapter → M2
            └── fresh GRPO adapter → M3
```

DPO and GRPO are **siblings rooted at SFT**. GRPO does not continue from DPO.
M1 is merged into the original FP16 base to create a frozen parent for the two
branches. Disabling a newly attached PEFT adapter exposes that M1 reference.

The model has 28 layers, hidden size 1,536, 12 attention heads and two KV heads.
LoRA targets the attention `q/k/v/o` and MLP `gate/up/down` projections. Rank 16
and alpha 32 yield **18,464,768 trainable parameters**, approximately 1.2% of the
base model size.

Original training used NF4 4-bit QLoRA, double quantization, FP16 compute and
gradient checkpointing. The original machine was an RTX 3060 12 GB. Current
follow-up work uses two independent RTX 5070 12 GB workers on a remote server,
with the 3060 used for controller development/evaluation.

## 3. Dataset and preparation

Training examples come from **published RunBugRun v2 buggy/fixed submissions**,
joined with problem descriptions and active tests. We did not invent toy bugs.
The downloaded archive was 120.5 MB; the restored SQLite database was about 2.3 GB.

The preparation pipeline is:

1. Extract short active Python examples from source splits 0/1/2.
2. Apply statement and actual Qwen-token context checks.
3. Choose 2–10 deterministic active tests per problem.
4. Require the published correction to pass all selected tests.
5. Require the original bug to fail at least one selected test.
6. Deduplicate exact ASTs and near-duplicate token sequences.
7. Split by **problem ID**, not individual submission pair.

| Stage | Pairs |
|---|---:|
| Published candidates extracted | 4,000 |
| Passed context/statement checks | 3,913 |
| Passed execution and deduplication | 3,044 |
| Selected train / dev / test | 2,000 / 200 / 200 |
| Verified but unused after caps | 644 |

The final splits contain 1,219 train, 147 dev and 144 held-out problem IDs, with
no problem-ID overlap. This matters because the source's predefined splits shared
some problems. Multiple submission pairs can still share a problem within a split.

The training corpus is approximately 1.56 million tokens: an average of 623 prompt
tokens and 155 response tokens per pair. Its longest full sequence is 2,024 tokens.
The original training context was 2,048. Twenty-six gold responses exceed the
original evaluation's 512-token generation cap, which is a recorded limitation.

HumanEvalPack's **164 Python tasks are evaluation-only**. All 164 published gold
programs passed the adapter sanity check, and all buggy programs failed it.
Code-overlap checks found no exact/near training matches, but semantic overlap
and foundation-model pretraining contamination remain unresolved.

## 4. Prompt and output contract

Every original arm uses the same initial chat prompt:

````text
SYSTEM: You are a Python code repair assistant. Return ONLY the full corrected Python 3 program.
USER:
Fix the provided program so it satisfies the problem specification.
Problem:
{statement}
Buggy program:
```python
{buggy_code}
```
Return corrected Python code, not explanations.
````

Qwen's tokenizer supplies its chat template. Gold code and scoring-only tests
are excluded from the prompt. Retry feedback contains counts/categories and
allowlisted diagnostics, not actual hidden inputs, expected outputs or raw stderr.

## 5. SFT: learning published corrections

SFT maps the original repair prompt to its corresponding published correction:

```python
prompt = chat_template(problem_statement, buggy_code)
completion = fixed_code + EOS
```

TRL `SFTTrainer` applies teacher-forced causal cross-entropy **only to response
tokens**. Prompt and padding labels are masked. Packing is disabled, and full
targets are retained rather than truncated.

Actual run: 2,000 pairs, one epoch, microbatch one, gradient accumulation eight,
250 optimizer updates, learning rate `1e-4`, linear decay and no warmup. It took
**26 minutes**, with **6.46 GB peak allocated GPU memory**.

Mean training loss was **0.0653**; logged token accuracy rose from approximately
96.4% to 98.3%. This is not repair accuracy: most output tokens may be unchanged
code, and teacher forcing supplies the preceding correct tokens.

Implementation: [`training/train_sft.py`](../../training/train_sft.py),
[`training/common.py`](../../training/common.py),
[`configs/full-sft.yaml`](../../configs/full-sft.yaml).

## 6. DPO: learning preferences between programs

The original DPO dataset used:

```text
prompt   = problem + original buggy program
chosen   = published fixed program
rejected = original buggy program
```

The policy learns to prefer the correction over the rejection relative to frozen
M1. This is a **weak-negative construction**: rejecting an unchanged input is an
easier distinction than identifying a plausible but incorrect repair.

Actual original run: 1,000 pairs from 1,000 problems, one epoch, 125 updates,
effective batch eight, learning rate `5e-6`, beta `0.1`. Reference log probabilities
were precomputed and cached. The successful run took **36.7 minutes** including
reference computation and peaked at **10.53 GB allocated**.

Final logged training preference accuracy reached 100%, while executable repair
results remained mixed. An earlier OOM was preserved and recovered using an
expandable CUDA allocator; the successful run retained context 2,048.

The current **DPO-hard** work replaces unchanged-bug rejections with actual M1
repair attempts. Four candidates are generated per training row. Retained
negatives must parse, finish normally, fail output comparison and differ from
the original bug/gold AST. Gold is reverified. Same-prompt IDs, seeds and grades
are recorded. The source pool has 2,000 rows. Collection is now scheduled to stop after 1,000 processed source examples, preserving the verified bank for three DPO experiments rather than waiting for all 2,000. Pair yield and quality remain measured outcomes, not a guaranteed count.
Multiple learning-rate/epoch checkpoints are selected using dev repair success.

Implementation: [`training/train_dpo.py`](../../training/train_dpo.py),
[`scripts/train_dpo_cached.py`](../../scripts/train_dpo_cached.py),
[`scripts/round2_fast_rollouts.py`](../../scripts/round2_fast_rollouts.py).

## 7. GRPO: learning from program execution

GRPO consumes prompts and trusted task references, samples four programs per
group, executes them and computes advantages relative to the group's rewards.
A KL term keeps the policy near the frozen SFT reference.

Original reward:

```text
invalid code / timeout / runtime failure / output-limit failure → 0
otherwise → passed_tests / total_tests + 0.05 if all-pass
```

The original run used 200 updates, 800 sampled completions, microbatch one,
accumulation four, learning rate `5e-6`, KL beta `0.04`, and 512 generated tokens.
Sampling used temperature 0.8, top-p 0.8 and top-k 20. It took **55.8 minutes** and
peaked at **3.97 GB allocated**. Those 200 updates were about 0.1 epoch, not one
complete pass over all 2,000 prompts.

A variance pilot found mixed rewards in 13/30 groups. During main training only
86/200 groups had reward variation. If every candidate receives the same reward,
the relative execution signal cannot rank them. KL can still contribute, so this
does not imply every such update has zero total gradient.

The active **GRPO-long** branch uses FP16 LoRA, up to 2,000 updates, checkpoint/dev
checks every 200 updates and early stopping after a sufficiently long plateau.
Its dense reward retains useful credit when other tests fail:

```text
0.85 × passed_test_fraction
+ 0.10 × normal_exit_fraction
+ 0.05 × all_pass
```

Its preflight had mixed rewards in 20/30 groups. This is not a controlled
reward-only improvement: precision, sampling and selected tasks also changed.
Intermediate **dev repair accuracy**, not training reward alone, selects the model.

Implementation: [`training/train_grpo.py`](../../training/train_grpo.py),
[`evaluation/reward.py`](../../evaluation/reward.py),
[`evaluation/reward_v2.py`](../../evaluation/reward_v2.py),
[`scripts/round2_train.py`](../../scripts/round2_train.py).

## 8. The agent and safe execution

V1 simply sends the latest program and aggregate failure summary back for another
repair. V3/V3.1 add a bounded structured diagnosis:

```text
failed candidate → fault/evidence/proposed change → repaired program → test
```

V3.1 retains the best tested candidate, preserves the plan while compacting
prompts, records actual code changes and feeds failed edit outcomes into the next
diagnosis. AST fingerprints detect repeated patches. Repeats still consume one
of the three code-generation calls, but are not executed again. There are at
most two diagnosis calls; they do not update model weights.

The candidate process runs in Bubblewrap with no network, read-only code/runtime
mounts, an unprivileged user, 3-second per-test limits, 512 MB address space and
bounded output/process/file resources. Hidden tests stay in the trusted grader.
The remote container cannot provide the required namespaces, so remote jobs call
the same local grader through a private SSH tunnel. Function-repair assertions
use a JSON call bridge and remain outside the candidate environment.

Implementation: [`agent/repair.py`](../../agent/repair.py),
[`agent/repair_v31.py`](../../agent/repair_v31.py),
[`agent/sandbox.py`](../../agent/sandbox.py),
[`evaluation/grading_service.py`](../../evaluation/grading_service.py).

## 9. Completed measurements

Original common protocol: NF4 inference, 2,048 context, 512 generated tokens,
greedy decoding, same tasks/tests, V1 controller, maximum three sequential attempts.

| Model | RunBugRun pass@1 | RunBugRun agent@3 | HumanEvalPack pass@1 | HumanEvalPack agent@3 |
|---|---:|---:|---:|---:|
| M0 | 58/200, 29.0% | 64/200, 32.0% | 91/164, 55.5% | 102/164, 62.2% |
| M1 SFT | 66/200, 33.0% | 78/200, 39.0% | 88/164, 53.7% | 104/164, 63.4% |
| M2 weak DPO | 59/200, 29.5% | 69/200, 34.5% | 87/164, 53.0% | 110/164, 67.1% |
| M3 GRPO | 65/200, 32.5% | 74/200, 37.0% | 87/164, 53.0% | 101/164, 61.6% |

SFT gave the strongest RunBugRun result. Weak DPO gave the strongest HumanEvalPack
agent result but hurt RunBugRun. Original GRPO did not beat SFT. These are
single-run point estimates, not significance claims.

The new controller dev ablation is also complete: on 40 distinct problems,
V1 solved 15/40 by attempt three, V3.1 without diagnosis solved 14/40, and full
V3.1 solved 14/40. All first-attempt counts were 13/40. **Diagnosis did not improve
success in this ablation**, despite more explicit state handling.

Agent@3 is not independent-sampling pass@3. Different test-suite strengths make
partial fractions diagnostic, and shared problem IDs require clustered uncertainty.
Original evaluation consumed nearly six hours across four models and two benchmarks,
compared with about two hours of main training.

## 10. Research connection and practical lessons

[ReCode](06_RECODE_CONNECTION.md) is the research connection: our measured tied
reward groups illustrate a limitation of outcome-only optimization. RepairRL
does **not** train or use a learned reasoning-process reward model, and does not
reproduce its consistency-gated algorithm. Inference-time diagnosis is not a
learned process reward. The paper's results must not be presented as ours.

The project demonstrates that practical post-training depends on verified data,
the intended preference distinction, useful reward variation, reference-policy
identity, bounded execution and fixed evaluation. Low loss, perfect preference
accuracy, or a successful training exit is not the final objective.

Limitations include the small model, one original seed, selected 2–10-test suites,
imperfect statement heuristics, possible semantic/pretraining contamination,
reused benchmarks, token caps and merge/precision changes. New experiments must
be compared with new common-protocol baselines rather than cherry-picked old scores.

## 11. Current work and where to read next

Remote GPU 0 generates hard negatives and then runs queued DPO tuning. GPU 1
trains longer GRPO and selects intermediate checkpoints on dev. The local 3060
handles controller/evaluation work. Existing checkpoints/results are preserved;
current status is in [LIVE_STATUS.md](generated/LIVE_STATUS.md). New held-out
improvements remain pending until full evaluations complete.

- [Training details and code snippets](03_TRAINING_IMPLEMENTATION.md).
- [Experiments, curves, failures and limitations](05_EXPERIMENTS_FINDINGS.md).
- [Slide-ready figures](assets/) and [60-minute seminar plan](07_TALK_60_MIN.md).
- [Five measured replay cases](08_DEMO_RUNBOOK.md).
- [Resume and artifact instructions](09_REPRODUCE_RESUME_PRESERVE.md).

Large artifacts are retained locally under
`/home/thao/Thuc/RepairRL-preservation/`. Complete trainer checkpoints include
optimizer/scheduler/RNG state; final adapters alone are inference or warm-start
artifacts. Cloud uploads have been stopped.
