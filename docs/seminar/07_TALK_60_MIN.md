# Suggested 60-minute seminar

Target: 55 minutes of presentation/demo plus 5 minutes for questions. Emphasize what was implemented and measured.

| Minutes | Topic | Visual / demonstration |
|---|---|---|
| 0–5 | Problem: repair a Python program and verify it | One buggy program and its specification |
| 5–10 | Pretraining, upstream instruction tuning, and our extra repair specialization | [Pipeline](figures/01_overall_pipeline.png) |
| 10–18 | Dataset acquisition, executable filtering, problem-group splits | [Filtering funnel](figures/02_dataset_filtering_funnel.png) |
| 18–25 | SFT implementation: NF4 QLoRA, targets, response masking | [Three workflows](figures/03_sft_dpo_grpo_workflow.png) |
| 25–32 | DPO: actual negative generation, verification, diversity limits | [DPO dev curve](figures/04_dpo_dev_progression.png) |
| 32–40 | GRPO: four samples, execution rewards, variance and checkpoint selection | [GRPO dev curve](figures/05_grpo_dev_progression.png) |
| 40–46 | Matched evaluation, transfer, uncertainty and failure interpretation | [Pass@1](figures/06_model_pass1_comparison.png), [agent@3](figures/07_agent3_comparison.png) |
| 46–52 | Live model switching and repair | [Demo guide](DEMO_QUICKSTART.md), [cup-swap case](figures/08_actual_repair_case.png) |
| 52–55 | ReCode connection, limitations and lessons | [Research connection](06_RECODE_CONNECTION.md) |
| 55–60 | Questions | Keep full technical report available |

## Speaker notes

- M0 is already instruction-tuned. Our SFT specializes it further. DPO and GRPO are sibling branches from M1.
- Training uses RunBugRun only. HumanEvalPack is evaluation-only. A large source collection does not imply that all its records were trained on.
- Show exact verification exclusions and distinguish examples from problem groups.
- High token accuracy or preference accuracy does not establish repair success.
- The DPO bank has 1,062 pairs but only 503 distinct problems; almost half the negatives pass no selected tests. Discuss the data, not merely the optimizer name.
- A useful GRPO reward needs candidate diversity and reward variance. Training longer does not guarantee monotonic gains.
- The main table is completed FP16/V1 evaluation of DPO step 66 and interim GRPO step 800. Dev curves and older NF4 results are separate.
- GRPO's RunBugRun gain does not establish HumanEvalPack transfer. Objective mismatch, specialization and forgetting are hypotheses of different evidential strength.
- Curated demos illustrate behavior; use the complete benchmark table for performance claims. Include both a DPO success and a DPO regression.
- Explain precisely which ReCode components were not implemented.

Use the [technical report](TECHNICAL_REPORT.md) as the authoritative numerical narrative and [figure captions](figures/README.md) for per-slide caveats.
