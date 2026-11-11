# Expanded SFT experiment (cancelled 2026-11-11)

**Current status:** Cancelled by the user during local CPU data preparation. The preparation process group was terminated and the remote waiting service stopped with autostart disabled. 2,103 accepted pairs (2,000 existing + 103 newly verified) remain preserved. No 10k SFT training started. The specification below records the cancelled setup.

User requested a fresh SFT run from **M0 Instruct**, targeting 10,000 pairs, after selected DPO evaluation finishes. This does not train from the untouched pretrained PT-base comparator or continue M1.

## Data and gates

`scripts.prepare_sft10k` extracts up to 60,000 published RunBugRun candidates in problem-diverse order (Python, token_count <=512, 2–10 active tests). It retains the existing 2,000 verified training examples and attempts to add 8,000. All original dev/test problem IDs are excluded. New examples must fit the existing chat prompt plus response in 2,048 tokens, pass every selected test for gold, and fail at least one for the buggy code. Verification uses the existing local Bubblewrap runner. AST exact deduplication and token 5-gram Jaccard >=0.9 deduplication compare new code against accepted training, fixed dev/test and HumanEvalPack code. No held-out performance is used in preparation. Existing dev/test manifests are copied byte for byte.

The preparation is resumable from accepted rows and verification logs. If fewer than 10,000 examples survive, it fails visibly and remote training remains blocked. The 10,000 count is a target until `data/prepared/sft10k/gate.json` passes. The gate hashes train/dev/test/test-suites; `training.common.check_data` rechecks them and problem disjointness before training.

## Training and evaluation

`configs/sft10k.yaml` inherits the original SFT settings: untouched pinned `Qwen/Qwen2.5-Coder-1.5B-Instruct`, NF4 QLoRA, rank16/alpha32/dropout0.05 on attention and MLP projections, max length2048, response-only loss, microbatch1/accumulation8, lr1e-4, one epoch, seed42. With exactly10,000 pairs this is1,250 optimizer updates. Save interval is250 updates. Latest optimizer states are retained by the Trainer and mirrored locally; separate inference snapshots preserve every dev-scored checkpoint.

`training.train_sft10k` monitors a fixed60-problem dev subset at step0 and each save, using the live NF4 model. This is **NF4 screening**, whereas the current DPO/GRPO screens use FP16. Do not compare those screening scores as matched model results. Select by dev pass@1, then partial fraction, then earlier step. After training, `scripts.sft10k_queue` evaluates that selected adapter at FP16 with the frozen V1 protocol on RunBugRun200 and HumanEvalPack164. The evaluator/parser/grader and held-out source files are unchanged. M0 V1 held-out evaluations already exist. No new SFT result is available yet.

## Scheduling and preservation

- Local CPU: detached `scripts.sft10k_prepare_upload`, log/status under `results/round2/sft10k-preparation/`. Upload publishes the successful gate last.
- Remote GPU0: Supervisor `repairrl-sft10k` waits for both the verified data gate and `dpo-selected/selection.json` state completed. It then trains and evaluates automatically. Remote GPU1 continues GRPO.
- Fresh model path: `results/round2/sft10k-model/`; evaluations/status: `results/round2/sft10k/`. Both fall inside the existing full-state remote-to-local preservation mirror.
- Scripts/configs are in Git; raw data and weights remain private. Original checkpoints/results are preserved.
- Interrupted training resumes from the last complete Trainer checkpoint with unchanged dataset/config hashes. Dev snapshots alone are inference adapters. Runtime GPU training/resume for this new experiment remains unverified until the queued job actually starts.
