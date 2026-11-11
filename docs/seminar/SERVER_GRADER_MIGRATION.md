# Candidate execution and server grading

The main published V1 comparison uses the original Bubblewrap runner with candidate Python 3.14.4. Candidates run as an unprivileged user with bounded CPU, memory, processes and output; hidden tests stay in the trusted grader.

A later deployment moved grading onto the GPU server because its container disallowed the user namespaces required by Bubblewrap. The alternative runner uses a minimal chroot, UID/GID 65534, seccomp restrictions and the preserved candidate Python 3.14.4 interpreter/standard library/shared libraries. Network access and process creation are blocked. Trusted tests are outside the jail.

Validation covered 20 RunBugRun tasks and all 164 HumanEvalPack tasks: gold passed and buggy programs failed for all 184. This is a runtime sanity check, not proof of identical behavior for every generated program.

## Source

- `agent/sandbox.py`: original local Bubblewrap executor.
- `agent/server_sandbox.py`: optional chroot/seccomp server executor.
- `evaluation/grading_service.py`: original trusted grader/client.
- `evaluation/server_grading_service.py`: server grader.
- `evaluation/server_grading_proxy.py`: server-local compatibility endpoint.
- `scripts/build_grader_runtime.py`: preserve candidate runtime and checksums.
- `scripts/validate_server_grader.py`: gold/bug validation.

The server runner requires a configured runtime tree and suitable Linux privileges; it is not a silent fallback. The live demo uses the original local Bubblewrap runner.

## Provenance

Isolation and timing changed during migration. Runs crossing the cutover require an execution ledger recording the change and the evaluation prefix already completed. Protocol files describe the grader at startup; the migration ledger supplies subsequent changes. Matching Python/test hashes plus validation supports a disclosed compatibility decision, not a claim of identical backend performance.

Retain runtime checksums, test hashes, grader identity and migration records alongside private experiment artifacts. Do not compare local and remote latency as a controlled hardware benchmark.
