# Full SFT run specification — 2026-11-10

This document describes the full run requested after the eight-example smoke
test. The workflow is M0 pre-SFT evaluation, fresh M1 training, reload check,
and M1 post-SFT evaluation. It does not train DPO or GRPO. See
`results/active-sft.json` for the live status-file path. Baseline evaluation is
required to finish successfully before the supervisor starts optimization.

## Data and preparation

Source: published RunBugRun v2 SQLite release, active Python (`language=5`)
records from original splits 0/1/2, joined with problem descriptions and active
stdin/stdout tests. No synthetic bugs, generated fixes or preference pairs are
used for SFT. HumanEvalPack is evaluation only.

| Stage | Pairs |
|---|---:|
| Extracted published candidates | 4,000 |
| Passed context and statement heuristic | 3,913 |
| Passed execution verification and deduplication | 3,044 |
| Retained for training | 2,000 |
| Retained for dev | 200 |
| Retained for held-out evaluation | 200 |
| Verified but unused because of split row caps | 644 |

The initial context/statement exclusions were 50 oversized prompts, 5 oversized
prompt/response sequences, and 32 statements requiring language/completeness
review. The execution/dedup exclusions were 243 gold failures, 564 bugs passing
the selected tests, 55 exact code duplicates and 7 near duplicates. These are
local filter outcomes, not claims that the original dataset is incorrect.

Source `token_count <= 512` is only a preliminary length filter. Actual limits
use the pinned Qwen tokenizer: prompt <=1,536 tokens and full prompt plus either
source response <=2,048 tokens. The statement heuristic requires >=40 whitespace
words and at least three English cue words; six sampled statements were also
inspected. This is not formal language or semantic-completeness validation.

Each accepted pair has >=2 active tests, up to 10 deterministic test IDs per
problem. The original fixed program must pass all selected tests; the original
bug must fail >=1. Gold and candidates use the same isolated Python executor and
output matcher. Exact deduplication uses Python AST hashes; near deduplication
uses Python token 5-gram Jaccard similarity >=0.9 on both code variants.

| Split | Repair pairs | Distinct problem IDs | Unique selected test cases | JSONL bytes |
|---|---:|---:|---:|---:|
| Train | 2,000 | 1,219 | 11,230 | 5,189,639 |
| Dev | 200 | 147 | 1,370 | 552,988 |
| Test | 200 | 144 | 1,323 | 492,760 |

Problem IDs are disjoint across splits. New groups use seed 42; existing pilot
problem assignments remain locked to their original splits, including prior
dev/test groups. Row caps are applied after group splitting. There are 13,923
unique selected tests across the retained splits; `tests.json` is 1,563,502 bytes.
The 200 held-out repair pairs require 1,861 test executions for one complete
attempt on every case (some share a problem's suite).

The full 2,000-row training set has no detected AST-exact/token-near code matches
with the 164 HumanEvalPack Python tasks. Semantic duplicates and foundation-model
pretraining contamination are not proven absent.

## Training examples and loss

Input: frozen system instruction + task description + original buggy full
program, formatted with Qwen's tokenizer chat template and an assistant prefix.
Target: the corresponding original fixed full Python program plus EOS.

```text
prompt = chat_template(system_instruction, statement, buggy_code)
completion = fixed_code + EOS
```

The model is teacher-forced on the full target program with causal next-token
cross-entropy. `completion_only_loss=True` masks system/user prompt tokens;
padding is masked too. Targets are not truncated. Sequence packing is disabled.
Tests are used for data validation and execution evaluation, not as an SFT reward.
Neither hidden test sources nor expected outputs are included in the prompt.

Tokenizer-estimated training corpus sizes:

| Tokens | Total | Mean per pair | Median | 95th percentile | Maximum |
|---|---:|---:|---:|---:|---:|
| Prompt | 1,246,960 | 623.48 | 587.5 | 1,099 | 1,535 |
| Response, including EOS | 310,459 | 155.23 | 116 | 410 | 689 |
| Full sequence | 1,557,419 | 778.71 | 718 | 1,455 | 2,024 |

Response counts are estimated from joint token length minus prompt length.
All 2,000 rows fit the training context. Generation is capped at 512 tokens for
both evaluation arms; 26 training gold programs exceed that cap (1.3%, maximum
689). Full gold targets are retained for SFT. Long generated programs may be
truncated and fail evaluation; this limitation is shared across arms. Prompts
that overflow the inference context fail explicitly rather than being truncated.

## Model, precision and adapter

Base: `Qwen/Qwen2.5-Coder-1.5B-Instruct`, pinned revision
`2e1fd397ee46e1388853d2af2c993145b0f1098a`. Full M1 starts from the untouched
base, not the pilot adapter. Architecture: 28 layers, hidden size 1,536,
intermediate size 8,960, 12 attention heads, 2 KV heads, vocabulary 151,936,
and tied input/output embeddings.

- QLoRA: 4-bit NF4 linear weights, double quantization, float16 compute.
- Base parameters are frozen; embeddings/norms may retain higher precision.
- LoRA rank 16, alpha 32, dropout 0.05, bias disabled.
- Target projections: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`,
  `up_proj`, `down_proj`.
- 18,464,768 trainable adapter parameters, approximately 1.2% of base size.
- Float16 mixed-precision training; bfloat16 disabled.
- SDPA attention, gradient checkpointing with `use_reentrant=False`;
  training KV cache disabled.
- One RTX 3060 desktop GPU, nominal 12 GB VRAM. Gradio is stopped.

## Optimization and checkpoints

| Setting | Value |
|---|---|
| Trainer | TRL SFTTrainer 0.23.1 |
| Epochs | 1 |
| Training examples seen | 2,000 |
| GPU microbatch | 1 |
| Gradient accumulation | 8 |
| Effective batch | 8 examples, one GPU |
| Expected optimizer updates | 250 |
| Maximum sequence | 2,048 tokens |
| Optimizer | torch AdamW |
| Initial learning rate | 1e-4 |
| Schedule | Linear decay to zero |
| Warmup | 0 steps / 0 ratio |
| Adam betas / epsilon | 0.9, 0.999 / 1e-8 |
| Weight decay | 0.0 |
| Gradient clipping | Maximum norm 1.0 |
| Model/data seed | 42 / 42 |
| Logging | Every 5 optimizer updates |
| Step checkpoints | Every 50 updates, keep latest 2 |
| Training-time dev evaluation | Disabled; no per-epoch dev-loss selection |
| Final adapter | `checkpoints/M1` |

The standard trainer shuffles training records with the frozen seed. CUDA
bitwise-deterministic algorithms are not forced. No learning-rate tuning or
checkpoint selection uses held-out/HumanEvalPack results. The final one-epoch
checkpoint is evaluated. The post-save smoke check uses one dev record; it is
not aggregate evaluation.

Software: Python 3.11.16; torch 2.8.0; Transformers 4.56.2; TRL 0.23.1;
PEFT 0.17.1; bitsandbytes 0.47.0; Accelerate 1.10.1; datasets 4.1.1.
The complete environment is recorded in `requirements.lock` and `uv.lock`.

## Pre- and post-SFT evaluation

The supervisor runs these stages sequentially:

1. Untouched M0 on the frozen 200 RunBugRun held-out pairs.
2. Untouched M0 on all 164 HumanEvalPack Python tasks.
3. Fresh SFT on the 2,000 training pairs, one epoch.
4. Save and dev-set reload/valid-code check.
5. M1 on the same 200 RunBugRun pairs.
6. M1 on the same 164 HumanEvalPack tasks.
7. Validate protocol compatibility and write separate comparison tables.

Both arms use greedy decoding (`do_sample=False`, temperature 0), seed 42,
512 new tokens, 2,048-token inference context, the same prompt template,
Qwen generation defaults including repetition penalty 1.1, and up to 3
sequential feedback-guided attempts with early stopping on all-pass.
Agent@3 is not independent-sampling pass@3. Feedback contains pass counts and
failure categories, never hidden test inputs, expected values or test sources.

Report first-attempt all-pass repair success, agent@3, validity, partial test
fractions, failure categories, generation tokens, generation/execution time,
attempts and test executions. HumanEvalPack is a separate table; its assertion
suite is scored as one pass/fail unit, without partial-assertion claims.

Candidate code runs under Bubblewrap with no network, UID/GID 65534, read-only
source, isolated PID/user namespaces, disabled nested user namespaces, no forks,
512 MB address-space limit, bounded file/output/tmpfs sizes and 3-second per-test
wall/CPU limits. Candidate Python is `/usr/bin/python3` (3.14.4 locally).
RunBugRun scores stdin/stdout; HumanEvalPack uses JSON function-call transport,
keeping assertions and expected values in the trusted grader. All 164 published
gold HumanEvalPack programs passed the adapter sanity check. HumanEval suites
also have an aggregate 30-second budget checked between function calls.

## Provenance and live artifacts

- `configs/full-sft.yaml`: full-run input configuration.
- `data/prepared/scale-verified/gate.json`: verification counts, split/group counts and SHA256 hashes.
- `results/scale-overlap-audit.json`: full train/external code-overlap audit.
- `results/full-sft-data-profile.json`: corpus byte/token statistics.
- `results/full-sft-trainer-profile.json`: resolved optimizer/trainer settings.
- `results/active-sft.json`: pointer to the live run's `status.json`.
- The run directory contains `config.resolved.yaml`, separate M0/M1 benchmark
  logs and raw generations, `train.log`, `reload.json`, and final comparison tables.
- The supervisor freezes dataset, experiment-source and environment hashes.
  If a baseline fails, training is not started. If training/reload fails,
  post-SFT scoring is stopped for inspection.
- Final `checkpoints/M1/repairrl-training.json` records actual optimizer logs,
  runtime, peak allocated VRAM, software versions and training row IDs.

The older eight-pair/two-step pilot is separate. Its results must not be mixed
with this 2,000-pair full run or the new held-out benchmark.
