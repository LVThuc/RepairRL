# Retry controller v2

The original v1 benchmark, its M0/M1 comparison, and the DPO handoff remain
frozen. V2 is an opt-in controller in `agent/repair_v2.py`; no checkpoint
retraining or extra planning-model calls are needed to exercise it.

The first attempt uses exactly the v1 prompt and greedy decoding. On failures:

1. Add the trusted test count/category, an allowlisted exception type and
   bounded candidate-source line when available. Raw stderr messages, test
   inputs, expected outputs and assertion sources are excluded.
2. Keep a short score history and detect repeated failed programs with AST
   fingerprints, including changes that affect only formatting/comments.
3. Explicitly request a changed correction. If the final retry is stuck on a
   repeated failure, sample at temperature 0.3 using a seed derived from the
   task ID, attempt number and seed 42. There are still at most three code
   generations and no additional diagnosis calls. The model can still fail or
   repeat: this policy does not guarantee a different or correct program.

Runtime observations are untrusted because candidate code can print fake
tracebacks. The controller extracts only static exception names and candidate
line numbers. It does not forward arbitrary stderr instructions or values.
When context is tight, it drops history before diagnostics and can fall back
to v1's compact summary; feedback actually sent is saved in each trace.

A visible trace includes:

```text
attempt=2 score=8/10 repeated=True sampled=False
feedback: ... same failed program was submitted again ...
attempt=3 score=... repeated=... sampled=True
```

These are illustrative fields, not claimed model results. Unit tests verify
budget/early-stop behavior, the repeated-patch branch, task-specific seeds,
real sandbox error extraction, and hidden-value redaction.

Run a small dev-only comparison when the GPU is idle:

```bash
python -m scripts.compare_agent_v2 --adapter checkpoints/M1 --tasks 5 --output results/agent-v2-dev
```

The command selects one case per dev problem using seed 42, runs both v1 and
v2 on the same fixed cases, and writes raw traces plus a summary. It reports
repair success, repetition and sampled-retry counts. The first queued check is
only five problems so it adds a bounded smoke check after the existing DPO job.
Its queue state is `results/agent-v2-dev.queue.json`. A larger 40–50-problem dev
ablation should follow only if the small check justifies more GPU time.

This v2 ablation combines changed feedback with a stuck-retry sampling policy;
it cannot isolate which component causes any difference. It does not replace
the frozen v1 test metrics. New results must identify controller version, and
any full v2 comparison must apply the same policy to every checkpoint. No
measured quality improvement is claimed before the dev check finishes.
