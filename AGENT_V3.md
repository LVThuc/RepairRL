# Bounded diagnosis and repair

`agent/repair_v3.py` implements the requested inference controller:

- At most **three code-generation calls** and **two diagnosis calls**.
- Each diagnosis is capped at **96 tokens**, with exactly three JSON string
  fields: `fault`, `evidence`, `change`. Invalid JSON is ignored; no extra call
  is made to repair the diagnosis.
- The first code prompt is identical to v1. Later prompts use redacted feedback,
  a short attempt history and the diagnosis as an unverified hint.
- The next repair starts from the **best tested candidate**. A lower test score
  cannot overwrite that candidate. Ties keep the earlier best candidate.
- An AST-equivalent repeat of a failed generated patch is **rejected before
  execution** and still consumes one code call. Its trace has `tested=false`,
  `duplicate_patch=true`, zero tests run and category `duplicate_patch`.
- A stuck retry can sample at temperature 0.3 with a task/attempt-specific seed.
  The controller does not promise that the model will generate a distinct or
  correct patch; a repeated attempt is rejected within the same hard budget.
- The returned program is the best feedback-tested candidate, with its score
  and source attempt recorded. Diagnoses are never treated as verified facts.

Runtime feedback contains only allowlisted exception names and bounded lines
in the candidate. It excludes raw stderr messages, hidden inputs, expected
outputs and private assertion source. History is pruned when context is tight;
a diagnosis that cannot fit is skipped. The model's original context/output
limits are preserved.

The code-only SFT/DPO/GRPO training data does not specifically train this
workflow. The comparison is an inference-controller ablation on fixed M1, not
three separately trained checkpoints.

## Dev comparison

```bash
python -m scripts.compare_agents --adapter checkpoints/M1 --tasks 5 --output results/agent-controller-dev
```

A detached five-problem check is queued after the main overnight jobs. Its
status is `results/agent-controller-dev.queue.json`, with a log at
`results/agent-controller-dev.log`. It selects one case per dev problem using
seed 42 and runs v1, v2 and v3 on exactly the same tasks. It records raw repairs,
diagnoses, duplicate rejection, best-attempt tracking, extra repairs after the
first attempt, code/diagnosis tokens and total model time.

This is a small smoke ablation. It cannot establish a reliable controller
ranking with five problems. No v3 improvement is claimed until measured. A
larger dev comparison should precede any selected-controller benchmark; the
same chosen controller must be used across model checkpoints. Existing v1
model comparisons remain unchanged.
