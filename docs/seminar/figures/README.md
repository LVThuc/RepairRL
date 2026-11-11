# Eight figures for the RepairRL seminar

All figures use16:9 layouts. PNG files are3200×1800; SVG files preserve editable text/vector shapes. The combined PDF contains all eight figures in order. Text in the figures is English for the seminar. Use SVG for resizing in slide editors and PNG when SVG import is unreliable.

| # | Figure | PNG | SVG |
|---|---|---|---|
|1|Overall pipeline|[PNG](01_overall_pipeline.png)|[SVG](01_overall_pipeline.svg)|
|2|Dataset filtering funnel|[PNG](02_dataset_filtering_funnel.png)|[SVG](02_dataset_filtering_funnel.svg)|
|3|SFT / DPO / GRPO workflow|[PNG](03_sft_dpo_grpo_workflow.png)|[SVG](03_sft_dpo_grpo_workflow.svg)|
|4|DPO dev progression|[PNG](04_dpo_dev_progression.png)|[SVG](04_dpo_dev_progression.svg)|
|5|GRPO dev progression|[PNG](05_grpo_dev_progression.png)|[SVG](05_grpo_dev_progression.svg)|
|6|Model comparison: pass@1|[PNG](06_model_pass1_comparison.png)|[SVG](06_model_pass1_comparison.svg)|
|7|First attempt versus agent@3|[PNG](07_agent3_comparison.png)|[SVG](07_agent3_comparison.svg)|
|8|Actual cup-swap repair|[PNG](08_actual_repair_case.png)|[SVG](08_actual_repair_case.svg)|

[Combined eight-page PDF](RepairRL_seminar_figures.pdf).

## Captions and speaker notes

1. **Pipeline.** M0 Instruct is specialized with repair SFT. DPO and GRPO independently start from M1. All four arms receive the same evaluation; the pretrained model is an additional comparator. The inference agent does not update weights.
2. **Filtering.** 4,000→3,913→3,044, then2,000train/200dev/200test. Explicitly account for644 accepted but unused examples. Train/dev/test problem groups are disjoint; examples are not independent problem counts. Top widths are proportional to the corresponding stage counts; split cards are categorical.
3. **Learning signals.** Same repair-prompt contract; different targets/objectives. SFT learns gold responses, DPO prefers gold over verified failed M1 repairs, and GRPO samples four current-policy completions and executes them. C1–C4 are schematic, not measured example programs or invented reward values.
4. **DPO dev.** This is repair pass@1 on60 fixed dev problems, not DPO preference classification accuracy. Counts:20,21,21,20,20,19 at steps0,66,132,198,264,330. Step66 is selected by dev, tied with132 and winning the earlier-step tiebreak. The curve does not prove a specific cause of degradation.
5. **GRPO dev.** Counts:20,21,22,22,25,24,24 at steps0,200,400,600,800,1000,1200. These are completed historical dev checks frozen through1200, not later/final results. Held-out comparison uses the dev-selected step800.
6. **Pass@1.** Completed FP16/V1 results: RunBugRun29.0/34.0/33.5/38.0%, HumanEvalPack57.3/61.6/57.3/60.4% for M0/M1/DPO66/GRPO800. Compare methods within each benchmark; benchmarks differ in format/distribution. Pretrained is omitted here to keep the requested four-arm comparison clear; its results remain in the technical report.
7. **Agent@3.** Same model, same tasks, first attempt versus success within at most three sequential attempts with feedback. It is not independent-sample pass@3. RunBugRun agent@3:30.0/40.0/38.0/42.5%; HumanEvalPack64.6/65.9/63.4/65.9%.
8. **Case.** Curated held-out RunBugRun21346 (p00047). Two independent `if` statements can undo a cup swap. First-attempt results for pretrained/M0/M1/DPO/GRPO are2/3,0/3,1/3,3/3,3/3 tests. M0 assumes a nonexistent leading count; M1 changes comparison operators incorrectly; DPO/GRPO use mutually exclusive branches. Displayed excerpts summarize recorded full programs. No hidden inputs or expected outputs appear. This illustrates a DPO success despite its weaker aggregate score. Full private case code is in the owner-only section9 package.

## Sources and regeneration

Measured inputs are exported under `../report/`: `results.json`, `case-studies.json`, `dpo-dev.json`, `grpo-dev.json`. `sources.json` records their SHA256 hashes. These exports contain aggregate metrics and public case metadata; raw hidden tests and private response traces are excluded.

From the project root, using the pinned figure environment:

```bash
python -m scripts.build_seminar_figures
```

The dependencies are listed in `../requirements-figures.lock`. Generation uses Matplotlib3.10.7 and NumPy2.4.6. Figures1–3 are implementation diagrams,4–7 are measured charts, and8 is a curated measured case. Main comparison uses the current FP16/768-token V1 protocol; do not combine it with the original NF4/512-token results.
