# Evaluation protocol

## Published comparison

- Models: pretrained Qwen2.5-Coder-1.5B; M0 Instruct; M1 repair SFT; DPO-hard step 66; GRPO step 800.
- Checkpoints are selected on dev. GRPO step 800 is an interim selection; later checkpoints are separate results.
- RunBugRun: 200 held-out examples from 144 problem groups. Training, dev and test problem IDs are disjoint.
- HumanEvalPack Python: 164 evaluation-only tasks. Never used for training or checkpoint selection.
- FP16 inference, fixed repair prompt, seed 42, greedy decoding, context 4,096, maximum 768 generated code tokens.
- V1: at most three sequential attempts, stopping on success; retries receive the current candidate and sanitized test-count/category feedback.

## Metrics

**Pass@1:** fraction of tasks passing all selected tests on the first attempt.

**Agent@3:** fraction solved within up to three feedback-guided attempts. This differs from independent-sample pass@3.

**Partial test fraction:** report per-task means at attempt 1 or the best observed attempt when comparing models. The existing `partial_fraction` summary field instead averages over all generated attempts, which weights models differently when their retry counts differ. HumanEvalPack is recorded as a binary whole-task outcome, not an individual-assertion fraction.

**Validity:** parseable candidate code. Validity does not imply execution correctness.

Store raw generations, traces, model identity, data/test hashes, decoding settings, runtime, latency and attempts. Use matched task IDs for paired comparisons. RunBugRun uncertainty estimates cluster by problem ID.

## Execution

Candidate code runs outside the training process. RunBugRun uses stdin/stdout and benchmark-style numeric/whitespace matching. HumanEvalPack uses isolated function-call RPC; trusted assertions stay in the grader. Timeout: 3 seconds per execution; RAM: 512 MB; output: 65,536 bytes.

The main completed comparison was graded with the local Bubblewrap runner and candidate Python 3.14.4. A later server migration preserved that runtime but changed isolation to chroot/seccomp. Runs crossing migration require their execution ledger and validation metadata; do not claim identical backend timing. See [execution notes](docs/seminar/SERVER_GRADER_MIGRATION.md).

## Reporting limits

Keep the original NF4/512-token results separate from this FP16/768-token protocol. Report curated demonstrations as curated. Do not use interim held-out inspection to tune continuing training. Account for limited selected test suites, single-seed training, possible pretraining contamination and unresolved semantic overlap.
