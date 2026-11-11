# Deferred asynchronous dev-scoring prototype

These files were drafted locally while round-2 experiments were already running.
They were **not deployed or launched**. The user then requested preservation without
interrupting those experiments. Canonical `scripts/round2_train.py` and
`evaluation/round2.py` have been aligned with the remote on-disk implementation.

Do not present this prototype as a completed feature or use it as a resume recipe.
It needs integration tests, path/package cleanup, an isolated smoke run, and an
explicit new experiment directory before future deployment. It moves periodic dev
scoring to the local GPU and batches dev-only inference. Existing active experiments
still evaluate dev synchronously on their assigned remote GPU.
