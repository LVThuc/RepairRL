# Live experiment status

Snapshot: 2026-11-11T07:22:29.598361+00:00

This is a point-in-time view, not completed held-out evidence. A running service may be waiting, not optimizing.

Latest GRPO optimizer step observed in mirrored log: 437.

## Remote queues

```json
{
  "rollouts-fp16/status.json": {
    "state": "completed",
    "rows_completed": 1000,
    "rows_total": 1000,
    "pairs": 1062,
    "distinct_problems": 503,
    "wave_seconds": 9360.008119283011,
    "batch_prompts": 8,
    "peak_allocated_gb": 8.943492608,
    "categories": {
      "pass": 1516,
      "runtime_error": 573,
      "incorrect_output": 1857,
      "invalid_code": 40,
      "timeout": 13,
      "output_limit": 1
    },
    "original_source_budget": 2000,
    "cutoff_target": 1000,
    "pairs_sha256": "f086c4d01d66f6e67ed0e4fdc83e4870bebb634f03cf321a58559da1bc19db92",
    "rollouts_sha256": "c89c98fce4dcf82836c9918847b445bc22937d4dc4880edb8831d0689edccd8c",
    "physical_rows_completed": 1005,
    "overshoot_preserved": 5
  },
  "grpo-dense/selection.json": {
    "state": "training",
    "latest_dev_step": 400,
    "selected_dev": {
      "n": 60,
      "pass@1": 0.36666666666666664,
      "agent@3": 0.36666666666666664,
      "avg_attempts": 1.0,
      "validity": 1.0,
      "partial_fraction": 0.5716666666666667,
      "avg_generation_tokens": 160.38333333333333,
      "avg_generation_seconds": 11.301377837334197,
      "avg_patch_seconds": 11.515811503833296,
      "tests_run": 543,
      "categories": {
        "incorrect_output": 32,
        "pass": 22,
        "runtime_error": 6
      },
      "diagnosis_calls": 0,
      "diagnosis_tokens": 0,
      "diagnosis_seconds": 0,
      "duplicate_generations": 0,
      "extra_repairs": 0
    },
    "parent_dev": {
      "n": 60,
      "pass@1": 0.3333333333333333,
      "agent@3": 0.3333333333333333,
      "avg_attempts": 1.0,
      "validity": 1.0,
      "partial_fraction": 0.5183333333333333,
      "avg_generation_tokens": 160.63333333333333,
      "avg_generation_seconds": 11.242273636583317,
      "avg_patch_seconds": 11.458520036417031,
      "tests_run": 543,
      "categories": {
        "incorrect_output": 33,
        "pass": 20,
        "runtime_error": 7
      },
      "diagnosis_calls": 0,
      "diagnosis_tokens": 0,
      "diagnosis_seconds": 0,
      "duplicate_generations": 0,
      "extra_repairs": 0
    },
    "improved_over_parent": true
  },
  "grpo-dense/variance.json": {
    "passed": true,
    "mixed_group_fraction": 0.6666666666666666,
    "reward_version": "dense"
  },
  "dpo-queue.json": {
    "state": "training",
    "pid": 8645,
    "updated_at": 1794381708.7807548,
    "trial": "dpo-lr3e6"
  },
  "grpo-followup-queue.json": {
    "state": "waiting_for_grpo_dense",
    "pid": 2610,
    "updated_at": 1794381731.8641381
  }
}
```

## Local controller ablation

```json
{
  "v1": {
    "status": "completed",
    "metrics": {
      "n": 40,
      "pass@1": 0.325,
      "agent@3": 0.375,
      "avg_attempts": 2.3,
      "validity": 0.9782608695652174,
      "partial_fraction": 0.44565217391304346,
      "avg_generation_tokens": 164.15217391304347,
      "avg_generation_seconds": 6.156517970858539,
      "avg_patch_seconds": 6.34946941418441,
      "tests_run": 824,
      "categories": {
        "incorrect_output": 65,
        "pass": 15,
        "runtime_error": 10,
        "invalid_code": 2
      },
      "diagnosis_calls": 0,
      "diagnosis_tokens": 0,
      "diagnosis_seconds": 0,
      "duplicate_generations": 0,
      "extra_repairs": 2
    },
    "n": 40
  },
  "no-diagnosis": {
    "status": "completed",
    "metrics": {
      "n": 40,
      "pass@1": 0.325,
      "agent@3": 0.35,
      "avg_attempts": 2.325,
      "validity": 1.0,
      "partial_fraction": 0.26344086021505375,
      "avg_generation_tokens": 162.67741935483872,
      "avg_generation_seconds": 6.272700881365635,
      "avg_patch_seconds": 6.368620962774354,
      "tests_run": 416,
      "categories": {
        "incorrect_output": 27,
        "duplicate_patch": 48,
        "pass": 14,
        "runtime_error": 4
      },
      "diagnosis_calls": 0,
      "diagnosis_tokens": 0,
      "diagnosis_seconds": 0,
      "duplicate_generations": 48,
      "extra_repairs": 1
    },
    "n": 40
  },
  "v31": {
    "status": "completed",
    "metrics": {
      "n": 40,
      "pass@1": 0.325,
      "agent@3": 0.35,
      "avg_attempts": 2.325,
      "validity": 1.0,
      "partial_fraction": 0.26774193548387093,
      "avg_generation_tokens": 163.93548387096774,
      "avg_generation_seconds": 6.343787977398093,
      "avg_patch_seconds": 6.451235183407784,
      "tests_run": 450,
      "categories": {
        "incorrect_output": 32,
        "duplicate_patch": 43,
        "pass": 14,
        "runtime_error": 4
      },
      "diagnosis_calls": 53,
      "diagnosis_tokens": 3811,
      "diagnosis_seconds": 153.4496400200078,
      "duplicate_generations": 43,
      "extra_repairs": 1
    },
    "n": 40
  }
}
```

Async dev-scoring code under experimental/ is deferred and was not deployed. Current remote dev scoring is synchronous.
