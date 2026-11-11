# Second-round evaluation progress

Completed summary files: 8/16.

Separate FP16/768-token protocol. Missing rows remain pending. This is not a complete comparison until all required rows exist.

| Model | Controller | Benchmark | n | pass@1 | agent@3 |
|---|---|---|---:|---:|---:|
| M0 | v1 | humanevalpack | 164 | 57.3% | 64.6% |
| M0 | v1 | runbugrun | 200 | 29.0% | 30.0% |
| M0 | v31 | humanevalpack | 164 | 57.3% | 68.9% |
| M0 | v31 | runbugrun | 200 | 29.0% | 30.5% |
| M1 | v1 | humanevalpack | 164 | 61.6% | 65.9% |
| M1 | v1 | runbugrun | 200 | 34.0% | 40.0% |
| M1 | v31 | humanevalpack | 164 | 61.6% | 65.9% |
| M1 | v31 | runbugrun | 200 | 34.0% | 38.0% |
