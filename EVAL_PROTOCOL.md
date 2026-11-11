# RepairRL evaluation protocol v1

No improvement is assumed. This is a standalone Python repair experiment, not a
Repair-R1 or ReCode reproduction. There are no hard-negative preference pairs.

M0 is Qwen/Qwen2.5-Coder-1.5B-Instruct at revision
`2e1fd397ee46e1388853d2af2c993145b0f1098a`. M1 is an SFT adapter. Merge M1
once at fp16 from the unquantized base on CPU, then independently load that
immutable merge for M2 weak-negative DPO and M3 execution-reward GRPO. Both new
adapters use disabled-adapter M1 as their reference; neither branch starts from
the other. Merge metadata records the M1 adapter checksum. Quantization/merge
rounding can slightly change numerical behavior; measure this when comparing.

Freeze data manifests, test suites, prompt version, tokenizer, model revision,
seed, greedy decoding, 512 new tokens, 2048 total tokens, 3-second per-call/test
sandbox timeout and three total attempts before comparing arms. Record the
resolved model revision, manifest/test SHA256, task IDs and all raw generations.
Overlength prompts fail explicitly rather than silently truncating the problem.
`evaluation.compare` rejects mismatched protocols or task sets. Report shorter
pilot subsets separately from full benchmark runs.

The prompt contains specification and original buggy code; retries contain the
latest candidate and a count/category-only test summary. No hidden expected
outputs, inputs, test sources or gold solution enter model prompts. Parse one
full Python program (optionally one surrounding code fence); explanations and
invalid syntax fail. Return the latest candidate after three failed attempts.

RunBugRun: select active Python records from predefined splits 0/1/2, then
construct NEW deterministic 80/10/10 problem-group splits. Audit original split
overlap. The pilot samples one variant per problem before taking later variants.
Gold must pass every selected active test and the bug must fail at least one.
Require two tests; cap at ten deterministic test IDs. Metrics refer to these
frozen selected tests, not every test in the full database. Deduplicate both
code variants with AST hashes and token 5-gram Jaccard >=0.9. This does not prove
absence of semantic duplicates or cross-benchmark problem overlap. Inspect
statements for language/completeness; token limits exclude oversized records.

The output matcher ports RunBugRun's per-line numeric-token comparison and
problem-specific absolute tolerances from upstream commit
`374251a9d65410f37e1136049cb7ff5dcca3d0ae`. Nonzero exit, timeout, or output cap
always fails even if a prefix matches. This exit-status policy is stricter than
upstream's output-first status precedence. Both gold and candidates use it.

HumanEvalPack: evaluation only, all 164 Python tasks. Use published declaration
plus buggy body as input and full function code as output. Published assertions
and helper definitions stay in the trusted grader. Candidate function calls use
JSON-only argument/result transport into fresh Bubblewrap processes. No pickle
or generated code runs in the host grader. Stateless RPC differs from a single
shared Python process; all 164 gold solutions must pass before scoring. No
partial assertion pass fraction is claimed: each assertion suite is one unit.
Each candidate suite has a 30-second aggregate budget checked between calls
(plus at most one in-flight per-call timeout). Function inputs necessarily reach the candidate, while assertion source and
expected values remain outside. Potential pretraining contamination remains.

Headline metrics are first-attempt all-pass success and sequential feedback-guided
agent@3, with early stopping. Agent@3 is NOT independent-sampling pass@3. Also
record validity (syntactically parseable code), failure categories, generation
tokens, generation/patch time, attempts, and tests run. Partial fractions are
only diagnostic. Report RunBugRun and HumanEvalPack separately. Do not choose
hyperparameters using HumanEvalPack. Disclose cherry-picked demonstration cases.

Reward: zero for invalid programs, timeout, output cap, or runtime failure;
otherwise passed/total plus 0.05 for all-pass. This is an outcome reward, not a
learned process reward. The variance gate requires >=30 distinct training
problems with four/eight samples each, and >=20% of groups with nonzero reward
variance. This operational threshold is a project choice requiring inspection,
not evidence of learning. GRPO defaults to 20 pilot optimizer steps; scaling
requires successful save/reload, evaluator and memory/time checks.

Candidate execution requires Linux user/network/PID namespaces, Bubblewrap,
and /usr/bin/python3. It drops capabilities, UID/GID 65534, disallows child
processes and nested user namespaces, limits address space to 512 MB, CPU and
wall time, output, file size and open files, and bounds temporary tmpfs to 16 MB.
Only /usr, minimal /dev, isolated /proc and read-only candidate source are
mounted. The sandbox environment is cleared; tests, model files, home, GPU,
workspace and host secrets are not mounted. Never add an unsandboxed fallback.
This is kernel isolation, not a formally verified hostile-code security boundary.
