# Controller decision: V1 only

On 2026-11-11 the user dropped V3 from remaining evaluations. Existing V3 traces and results remain historical artifacts. Pretrained, selected DPO, selected GRPO, and final comparison jobs now use V1 only. The frozen evaluator and generation/test settings remain unchanged; required headline coverage is now 8 results (4 models × 2 benchmarks). PT-base is an additional comparator.
