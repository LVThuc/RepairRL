# 08 — Demonstration runbook

## Default: replay existing measured cases

Use the [curated case index](generated/DEMO_CASES.md). Five examples are selected
by observed outcomes and explicitly labelled as such: an SFT rescue, a feedback
recovery, a GRPO regression, an external-benchmark DPO success, and a persistent
failure. They demonstrate behavior; they do not estimate benchmark accuracy.

No hidden test inputs, expected outputs, assertion source or raw stderr belong
on a slide. Full source examples and sanitized recorded generations are in the
**private local** preservation packet, not ordinary Git. Raw scoring traces remain
in the complete private results backup.

After restoring the handoff archive:

```bash
# Lists cases; needs neither CUDA nor Hugging Face nor API credentials.
python -m scripts.seminar_demo --packet /path/to/preservation/demo-packet.json

# Replay aggregate feedback and the recorded M1 program for the recovery case.
python -m scripts.seminar_demo --packet /path/to/preservation/demo-packet.json \
  --case 2 --arm M1 --show-code

# Contrast a measured SFT success with the corresponding GRPO failure.
python -m scripts.seminar_demo --packet /path/to/preservation/demo-packet.json \
  --case 3 --arm M3 --show-code
```

These commands **replay**, not rerun, a model. Say so in the seminar. The script
prints counts/categories and optional generated code, never test values.

## Suggested three-minute sequence

1. Show the task interface and original buggy program from the private packet.
2. Replay case 2: first candidate fails; the next feedback-guided attempt succeeds.
3. Show the attempt count and all-pass result, not private test cases.
4. Replay case 3 or case 5 as a counterexample. A completed RL run or a planning
   call does not guarantee success.
5. Return to the full aggregate table and disclose outcome-based curation.

For a code-level slide, use a short difference between the recorded candidate
and the source correction, with the case ID/provenance. Do not replace the
model's output with a hand-corrected version and present it as a measured repair.
Review source terms before redistributing full third-party examples.

## Re-exporting the packet

If historical raw results and held-out manifests are restored:

```bash
python -m scripts.seminar_demo \
  --export-private /path/outside/git/demo-packet.json
```

This regenerates the public outcome-only index and a private packet. Keep the
private file out of Git. The preserved packet contains source code for presenter
inspection, but excludes scoring-only test payloads.

## Live model demonstration, only after current jobs are idle

The original Gradio entrypoint is `app.py`. It supports saved original adapters
and the V1 controller; it is not evidence that round-2 adapters/V3.1 have already
been integrated into the UI. For a safe original-model UI demonstration:

```bash
python app.py --data data/prepared/scale-verified/test.jsonl \
  --tests data/prepared/scale-verified/tests.json --checkpoint-root checkpoints
```

It binds locally by default. Do not launch another model onto a GPU occupied by
training. Do not change candidate isolation merely for a faster stage demo.
Expect live outputs to vary with model precision, hardware and decoding settings.
Keep the recorded replay ready as a backup and label any new live run separately
from the frozen benchmark.

## Failure fallback

- Missing CUDA/model: replay the preserved trace.
- Missing raw datasets: restore the small private demo packet from the local archive.
- Missing private packet: show the public case index and aggregate tables; do not invent code.
- Slow sandbox/model: switch to replay rather than increasing attempt budgets.
- Late new experiment results: show them only if the complete matched evaluation
  and its protocol are available; otherwise use the completed original table.
