# Untouched pretrained comparator (PT-base)

Requested 2026-11-11; queued separately from the original M0–M3 experiment.

- Model: https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B
- Pinned revision: `df3ce67c0e24480f20468b6ef2894622d69eb73b`.
- Official model card identifies training stage as **Pretraining**. No RepairRL adapter, SFT, DPO or GRPO is applied.
- Existing M0 remains `Qwen/Qwen2.5-Coder-1.5B-Instruct`; PT-base is an additional comparator.

`scripts/round2_pretrained_baseline.py` downloads weights immediately and records file checksums. It waits for all four common-parent M1 evaluations on the local GPU to finish, then evaluates PT-base on RunBugRun held-out 200 tasks and HumanEvalPack 164 tasks, with both V1 and V3.1. Remote DPO and GRPO continue independently.

The evaluator, code parser, grader, data splits, FP16 precision, seed 42, chat prompt contract, context 4096, code limit 768, and three code attempts match the frozen round-2 protocol. This is an evaluation of the untouched pretrained model under our existing repair/agent interface. Chat-format compliance may be weaker without instruction tuning. It does not isolate general coding ability or measure the effect of training RepairRL from a pretrained initialization. No prompt tuning on held-out tasks is performed.

Runtime artifacts remain private under `results/round2/pretrained-base/`: `status.json`, `queue.log`, `model-inventory.json`, per-benchmark JSONL traces, protocol manifests and summary JSON. The immutable model snapshot is in the local Hugging Face cache. Completed outputs will be documented separately; no result is available at queue creation.

Run or resume using `.venv/bin/python -u -m scripts.round2_pretrained_baseline` from the project root. Existing task traces resume by ordered task IDs; completed summaries are skipped. A failed status must be inspected before relaunching. The job uses the same local grading service as other round-2 evaluations.
