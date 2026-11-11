# 07 — Suggested 60-minute presentation and speaker notes

## Audience and learning objectives

Assume familiarity with token prediction and basic fine-tuning. Explain just
enough SFT/DPO/GRPO theory to understand the actual data contracts and failure
modes. By the end, the audience should be able to describe the implemented
pipeline, read its metrics critically, identify a valid training comparison, and
distinguish a model adapter from a resumable trainer checkpoint.

The schedule below is **55 minutes of presentation/demo plus 5 minutes Q&A**.
Use about 26 slides. Each row gives an actual artifact or code reference so a
slide-producing AI can create the deck without inventing a pipeline.

| Slide | Minutes | Title / artifact | Main message |
|---:|---|---|---|
| 1 | 0–1 | LLM Post-training in Practice | A small repair model with honest mixed results |
| 2 | 1–4 | Why repair is a useful laboratory | Published bugs, executable outcomes, controllable scope |
| 3 | 4–6 | M0 → M1 → sibling branches | Architecture diagram from document 01 |
| 4 | 6–7 | Weights versus controller | Training and inference-time debugging are different interventions |
| 5 | 7–8 | One task and one output contract | Specification + buggy code → full corrected program |
| 6 | 8–10 | 4,000 candidates to 2,000 train pairs | Data funnel from document 02 |
| 7 | 10–12 | The split can leak | Group by problem ID; 200 pairs are 144 problems |
| 8 | 12–14 | Gold-pass / bug-fail gate | Show real verification code rather than a trainer screenshot |
| 9 | 14–17 | The grader is a product component | Trust-boundary diagram; candidates never receive expected values |
| 10 | 17–20 | SFT example and loss mask | Prompt/completion snippet, response-only labels |
| 11 | 20–22 | Fitting 1.5B into 12 GB | NF4 + rank-16 LoRA; show actual memory/parameter counts |
| 12 | 22–24 | SFT curve versus repair accuracy | 98% token accuracy does not mean 98% repairs |
| 13 | 24–26 | The first DPO dataset was weak | Chosen gold versus rejected input bug |
| 14 | 26–28 | Reference policy and OOM recovery | Frozen M1, reference cache, allocator issue |
| 15 | 28–30 | Hard negatives from actual M1 repairs | Current bank criteria and status, not a fabricated final size |
| 16 | 30–33 | GRPO reward callable | Show the original executable reward function |
| 17 | 33–36 | No reward variation, little ranking signal | 86/200 mixed groups; four equal scores example |
| 18 | 36–38 | What the short RL run achieved | 200 updates, 800 completions; held-out result did not beat SFT |
| 19 | 38–40 | Longer training with dev monitoring | Active 2,000-step ceiling, saved states, selection criteria |
| 20 | 40–43 | Diagnosis → change → repair → test | V3.1 sequence diagram; bounded calls and best candidate |
| 21 | 43–45 | The controller ablation | Completed 40-problem comparison, including the negative result |
| 22 | 45–47 | The matched results table | Separate RBR/HEP, pass@1/agent@3 and protocol labels |
| 23 | 47–50 | One success and one failure replay | Measured curated cases from document 08; no new GPU job needed |
| 24 | 50–54 | ReCode as the research connection | Use document 06; name the missing component accurately |
| 25 | 54–55 | Lessons and next experiment | Data quality, reward signal, evaluation, reproducibility |
| 26 | 55–60 | Questions | Keep backup technical slides available |

## Slide-by-slide notes

### 1–2: set the tone

Say that RepairRL is an executable learning experiment, not a polished claim that
every post-training objective improves every benchmark. State the target model,
the repair task and the scope before discussing algorithms. Competitive-programming
solutions make full-program execution feasible; arbitrary repository repair would
require a substantially different environment.

### 3–4: draw the genealogy correctly

M2 and M3 are separate children of the same SFT parent. A sequential M0→SFT→DPO→GRPO
arrow would misrepresent the experiment. Then draw the controller outside all
checkpoints: the same model can be run with V1 or V3.1, and the same controller can
be run with different models. This is the basis of attribution.

### 5: show the interface before the loss

Use the prompt template with placeholders. The response is complete code, not an
explanation or patch hunk. Explain why this simplifies parsing. Mention that actual
hidden tests are not in the prompt; public examples in the specification are normal.

### 6–8: demonstrate that dataset work is part of post-training

Show the real funnel counts and exclusions. Explain that a source gold failure
does not automatically mean the dataset is wrong: Python version, dependencies,
time limits and selected tests matter. Show the problem-ID overlap finding and
why regrouping was necessary. On the gate slide, demonstrate `gold.success and
not buggy.success`, not a toy mutation generator.

### 9: make execution concrete

Use the trust-boundary diagram. Hidden expected values remain in the grader;
the candidate gets only its input or function arguments. Explain the local
Bubblewrap and remote private-RPC arrangement. Avoid a prolonged security lecture:
the practical point is reproducible, bounded execution that cannot read the grader.

### 10–12: teach SFT through one batch

Show `prompt`, `completion=fixed_code+EOS` and `labels=-100` over prompt tokens.
Then show rank-16 adapters on attention/MLP projections. Use actual training
counts: 2,000 pairs, one epoch, 250 updates, 26 minutes, peak 6.46 GB allocated.
Display the loss/token-accuracy chart next to held-out repair success. The gap is
the lesson: predictable unchanged tokens dominate a code-repair target.

### 13–15: preferences encode a particular distinction

The weak negative is the exact buggy input, not a plausible alternate repair.
Show how easy it is to learn that preference while failing new tasks. Explain
why DPO's reference is frozen M1. Describe the actual memory-recovery event and
preserved failed run. For hard negatives, show the selection checklist and the
timestamped bank status; never relabel the target size as a completed dataset.

### 16–19: reward engineering and monitoring

Show 10–15 lines from `outcome_reward`, then a group of four equal scores. Explain
group-relative advantages in one equation, with a warning that the clip epsilon
and numerical standardization epsilon are different concepts. Report the measured
zero-variance fraction and negative held-out result. Explain that the new dense
reward preserves useful partial credit but may reward ordinary wrong termination.
Intermediate dev checks, not reward alone, decide which checkpoint survives.

### 20–21: distinguish an implemented loop from a demonstrated gain

Show the JSON plan and the actual edit/outcome record. A plan must survive prompt
compaction and a regressed candidate must not erase the best one. Repeated
generation still consumes an attempt. Then show the completed new dev ablation:
V1 15/40 agent successes, no-diagnosis 14/40, V3.1 14/40. Say explicitly that the
new controller did not add measured solves. Do not turn a cleaner architecture
into an unsupported quality claim.

### 22–23: results and replay

Use the original common-protocol table unless new final evaluations have fully
completed. State denominators and show both benchmarks: M1 is strongest on RBR,
while M2's HEP agent result is a limited positive finding. Replay a curated win
and a counterexample from saved traces. Label them as demonstrations selected by
observed outcomes; they do not replace aggregate evaluation.

### 24: the research connection

Use the concise verified facts and reward equation in document 06. Spend most of
the slide on our implementation boundary and the all-correct reward-tie example.
Do not imply that inference-time diagnosis trains a process evaluator. Do not
borrow the paper's benchmark improvement as a RepairRL result.

### 25: practical takeaways

Post-training is an entire measurement system: verified examples, the intended
preference/reward distinction, bounded execution, checkpoint lineage, fixed
evaluation and preserved state. More epochs or a successful trainer exit is not
the final metric. A negative result with a useful diagnosis is a valid seminar
outcome.

## Backup slides and anticipated questions

- **Is 2,000 SFT pairs enough?** Enough for this pilot to measure an effect; not a
  universal sample-complexity claim. Learning curves over data scale/multiple seeds
  would be needed to answer more generally.
- **Why preference accuracy 100% but weak repair results?** It measures fit to the
  training preferences. Weak rejection construction and distribution differences
  are plausible hypotheses, not separately proven causes.
- **Why did GRPO not win?** We measured many tied groups and a short run. Capacity,
  test strength, reward design and optimization are additional hypotheses.
- **Does the agent think?** It emits a structured hypothesis and tests a changed
  program. No ground-truth reasoning trace or learned process score is present.
- **Can I resume from an adapter?** It is a weight warm start, not optimizer-state
  continuation. Show the complete `checkpoint-*` file list.
- **Why not DDP on both 5070s?** Independent experiments fit each GPU and reduce
  wall-clock contention. Their VRAM is not automatically pooled.
- **Are the benchmarks clean?** Problem grouping and code overlap checks help;
  semantic/pretraining contamination and repeated benchmark reuse remain limitations.

## Deck-production rules

Keep the first three result charts unchanged except for styling. Use matching
colors for M0–M3 and separate controller panels. Put “completed,” “dev only,” or
“running” on every experiment slide. Prefer a few readable code lines over full
files. Leave the artifact/reproduction appendix accessible via the README link.
If late results arrive, regenerate evidence and replace the entire matched table,
not only the rows that improved.
