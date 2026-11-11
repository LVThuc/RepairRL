# LLM Post-training in Practice — RepairRL seminar handoff

**Slide figures:** [Eight PNG/SVG figures and combined PDF](figures/README.md).

**Full technical report:** [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md) — implemented data pipeline, training settings, matched V1 results, uncertainty, and three curated cases.

**Want one document? Read [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md).** It covers
the complete implemented project, actual settings, measured results and current work.

This is the entrypoint for a roughly 60-minute seminar grounded in the **implemented
RepairRL project**, rather than its original proposal. The case study is deliberately
small enough to inspect: a 1.5B Python repair model, paired executable examples,
three post-training objectives, and an inference-time debugging controller.

## Read this first

- **Completed:** the original M0/M1/M2/M3 experiment, including both evaluation
  benchmarks, saved adapters, trainer checkpoints, and a five-problem controller check.
- **Running:** the second-server rollout bank and longer GRPO experiment, the local
  40-problem controller ablation, and automatic DPO/final-evaluation queues.
- **Not deployed:** the asynchronous dev-evaluation prototype in `experimental/async_dev/`.
- **Research relationship:** RepairRL uses execution/outcome rewards. It does not
  implement ReCode's learned reasoning-process reward model or reproduce CG-GRPO.
- **No universal win:** SFT was strongest on the original RunBugRun test set; weak
  DPO helped HumanEvalPack agent@3 but hurt RunBugRun; original GRPO did not beat SFT.

For the latest timestamp, read [LIVE_STATUS.md](generated/LIVE_STATUS.md). A partial
dev run is not a completed result. The original numerical evidence is in
[COMPLETED_RESULTS.md](generated/COMPLETED_RESULTS.md), with machine-readable
[CSV](generated/round1-results.csv) and [JSON](generated/round1-results.json).

## Suggested reading order

| Document | What the presenter can obtain |
|---|---|
| [01 — Context and architecture](01_CONTEXT_ARCHITECTURE.md) | Narrative, boundaries, diagrams, checkpoint genealogy |
| [02 — Data and verification](02_DATA_AND_VERIFICATION.md) | Published source, actual filtering, split sizes, provenance |
| [03 — Training implementation](03_TRAINING_IMPLEMENTATION.md) | Exact objectives, hyperparameters, API snippets and references |
| [04 — Agent, execution and evaluation](04_AGENT_EXECUTION_EVALUATION.md) | V1/V3/V3.1, trust boundaries, feedback redaction, metric definitions |
| [05 — Experiments and findings](05_EXPERIMENTS_FINDINGS.md) | Completed results, costs, curves, failed runs and limitations |
| [06 — ReCode connection](06_RECODE_CONNECTION.md) | Verified paper relationship and explicit non-reproduction boundary |
| [07 — 60-minute presentation](07_TALK_60_MIN.md) | Timed slide plan, speaker notes, figures and transitions |
| [08 — Demo runbook](08_DEMO_RUNBOOK.md) | Replay/live procedures and curated measured cases |
| [09 — Reproduce, resume and preserve](09_REPRODUCE_RESUME_PRESERVE.md) | Restore instructions, resume commands, artifact semantics |
| [10 — Ownership and operations](10_OPERATIONS_HANDOFF.md) | Active services, dependency graph, update procedure and blockers |
| [Implementation index](generated/CODE_INDEX.md) | Function-level links and source SHA256 hashes |

The documents are self-contained for slide production. Use the actual code links
when showing implementation; use generated data files when drawing charts.

## Slide-ready assets

- [Repair success, SVG](assets/round1-repair-success.svg) / [PNG](assets/round1-repair-success.png).
- [Training curves, SVG](assets/round1-training-curves.svg) / [PNG](assets/round1-training-curves.png).
- [Training costs, SVG](assets/round1-training-cost.svg) / [PNG](assets/round1-training-cost.png).
- Mermaid diagrams in documents 01 and 04: model genealogy, trusted grading and agent sequence.
- Source snippets in document 03; measured case identifiers in document 08.

Captions matter: token accuracy is teacher-forced, DPO preference accuracy is a
training metric, GRPO rollout success is on training prompts, and none equals
held-out repair accuracy. Original and second-round protocols must stay in
separate tables.

## Reproducibility and storage

Source/documentation: `https://github.com/LVThuc/RepairRL`.

Large artifacts are deliberately excluded from ordinary Git. See
[`artifacts/manifests/storage-index.json`](../../artifacts/manifests/storage-index.json)
for the local backup index, byte counts, expected hashes and retrieval locations.
The preservation root is `/home/thao/Thuc/RepairRL-preservation/`. Per the owner's
instruction, local storage is the authoritative artifact store. Cloud uploads
were stopped; the abandoned partial upload is not a restoration source.

The inventory distinguishes full trainer states from inference-only adapters.
Do not destroy the remote instance merely because a final adapter exists. Ensure
the complete checkpoint files have been mirrored and verified first.

## Updating the handoff

```bash
# Refresh aggregate results, run status and function/source index; no GPU required.
python -m scripts.seminar_snapshot

# Figures use an isolated docs environment; training dependencies stay unchanged.
uv venv /tmp/repairrl-docs-env --python 3.11
uv pip install --python /tmp/repairrl-docs-env/bin/python matplotlib==3.10.7
/tmp/repairrl-docs-env/bin/python scripts/seminar_figures.py
```

The active refresh watcher updates generated evidence as experiments finish.
GitHub publication and local archive refresh are separate operations: a new local
file is not evidence that it has already been pushed or included in an archive. Consult document
10 and the artifact index for the latest verified publication/backup state.

## Instructions for another AI producing slides

1. Keep the completed original experiment as the main numerical case study until
   new held-out evaluations finish.
2. Build the deck from the timed plan in document 07; spend most time on actual
   data, code, debugging and evaluation decisions.
3. Preserve denominators, protocols, hardware and status labels on result slides.
4. Do not invent new measurements, learning curves, diagnosis successes or causal explanations.
5. Label selected demonstrations as curated and avoid exposing scoring-only tests.
6. Describe ReCode using document 06's verified distinction, not the obsolete draft's ambitions.
7. Retain negative results and operational failures: they are central to what the
   project teaches about post-training in practice.
