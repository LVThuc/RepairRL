# Data and model artifacts

The public repository contains implementation, configurations, measured aggregates, figures and split IDs. Raw benchmark data, trained weights, hidden tests, private response traces and full optimizer checkpoints are stored separately.

| Artifact | Needed for |
|---|---|
| Pinned pretrained/Instruct checkpoints | Base-model inference and fresh SFT |
| Prepared RunBugRun train/dev/test + trusted test suites + gate | Training and matched repair evaluation |
| HumanEvalPack prepared Python tasks | External evaluation |
| SFT adapter and exact merged M1 parent | Parent inference and independent DPO/GRPO branches |
| DPO step-66 and GRPO step-800 adapters | Published comparison and live demo |
| Full Trainer checkpoints | Resume an interrupted optimizer trajectory |
| Raw generation/test traces | Task-level analysis and figure-data regeneration |

Exact RunBugRun split membership and prepared-file hashes are public in [the split manifest](../data/manifests/runbugrun-splits.json). Model revisions and reported adapter stages are documented in the [technical report](seminar/TECHNICAL_REPORT.md). Obtain datasets/models from their upstream sources and the experiment-specific prepared artifacts/adapters from the owner. Upstream availability does not imply all derived adapters have been publicly hosted.

An inference adapter lacks optimizer momentum, scheduler position and RNG state. A resumable Trainer checkpoint needs adapter files, optimizer/scheduler/RNG state, trainer progress and applicable scaler state. Only load trusted states. `training/resume.py` validates hashes and checkpoint completeness before delegating restoration to the pinned Trainer implementation.

For continuation, preserve the matching dataset/configuration, parent merge identity, environment, source hashes and execution metadata. Do not substitute a different M1 merge silently: small numerical differences can change deterministic generations. Keep interrupted or historical artifacts separate from new output directories.
