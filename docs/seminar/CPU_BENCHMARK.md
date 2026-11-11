# CPU verification benchmark (2026-11-11)

Full expanded SFT preparation remains cancelled. This bounded benchmark runs 12 published source pairs (four each from prior accepted, buggy-passes and gold-fails categories) against their full selected suites. It does not prepare the 10k dataset.

| Machine | Workers | Seconds for 12 pairs |
|---|---:|---:|
| local | 1 | 5.93 |
| local | 4 | 1.81 |
| server | 1 | 7.86 |
| server | 4 | 2.29 |

Per-test grading outcomes agree across all benchmark runs: True.

Local uses the existing Bubblewrap runner (Python3.14); server uses an experimental minimal chroot with Python3.12, UID65534, seccomp denying network and process creation, bounded CPU/RAM/output and trusted grader outside the jail. Isolation smoke checks pass. Different runtimes/executors prevent treating these timings as a controlled CPU-only hardware comparison. Server CPU is Xeon E5-2630v4 (20 logical CPUs), local CPU Ryzen5700G (16 logical CPUs).

The sample is small and excludes extraction, tokenization, near deduplication and rejected-candidate search. It cannot establish a 10k preparation ETA or acceptance rate. Private per-test records remain under results/round2/cpu-benchmark/.
