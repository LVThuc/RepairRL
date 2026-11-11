# Seminar material

A practical case study of Python code-repair post-training with Qwen2.5-Coder-1.5B.

| Material | Contents |
|---|---|
| [Technical report](TECHNICAL_REPORT.md) | Context, verified data, training, completed V1 results, limitations and three case studies |
| [Live demo guide](DEMO_QUICKSTART.md) | Editable code, model switching, real execution, and SSH forwarding |
| [Figures](figures/README.md) | Eight slide figures in PNG/SVG, plus a combined PDF |
| [Talk outline](07_TALK_60_MIN.md) | Suggested 60-minute flow and speaker notes |
| [ReCode connection](06_RECODE_CONNECTION.md) | What the paper contributes and what RepairRL implements |
| [Execution notes](SERVER_GRADER_MIGRATION.md) | Candidate isolation, runtime validation and grading migration |
| [Measured aggregates](report/results.json) | Completed matched V1 metrics with trace/summary checksums |

Use the completed **FP16 / V1** table as the main result. The GRPO step-800 evaluation is an interim dev-selected checkpoint. The original NF4/512-token experiment remains a historical appendix in the technical report.

Raw datasets, hidden tests, model weights and private deployment records are outside this public source tree.
