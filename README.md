> **Seminar and preservation handoff:** start at [docs/seminar/README.md](docs/seminar/README.md).
> It documents actual code, completed measurements, active experiments, reproduction,
> and inference-only versus resumable artifacts. Large model/data/state files remain
> in the local preservation store; see [the inventory](artifacts/manifests/storage-index.json).
> Historical setup notes below can refer to earlier pilot defaults.

# RepairRL

Execution-gated Python repair on Qwen2.5-Coder-1.5B-Instruct. M0 is untouched;
M1 is SFT; M2 is M1 + **weak-negative DPO**; M3 is independently M1 + execution
GRPO. Full replacement programs and at most three inference-only repair attempts.
No improvement, completed training experiment, or <40-hour runtime is promised.
See [setup status](results/SETUP_STATUS.md) for actual local checks and remaining gates.

## Environment

Linux with NVIDIA CUDA support and working unprivileged Bubblewrap namespaces.
Tested here on RTX 3060 12 GB. Python 3.11 is used for training; candidate Python
is `/usr/bin/python3` (3.14.4 on this machine), recorded by the doctor. Source
examples incompatible with that candidate interpreter fail gold verification.

```bash
cd /home/thao/Thuc/nlp/RepairRL
source .venv/bin/activate
python -m scripts.doctor
python -m pytest -q
```

For a fresh clone, install [uv](https://docs.astral.sh/uv/), then run
`uv sync --frozen --extra dev` and `source .venv/bin/activate`. `uv.lock` pins the
complete resolution; `requirements.lock` records the installed Linux environment
for `uv pip install -r requirements.lock` followed by `uv pip install -e . --no-deps`.
System requirements are `bubblewrap`, `lrzip`, and Git. No sudo access was needed
here: Bubblewrap was installed already; an official Ubuntu lrzip package was
extracted to ignored `vendor/lrzip/`. On another machine install these through
your package manager. Never bypass the sandbox if its check fails.

## Published data and pilot gate

The current local `data/raw/runbugrun.db` was restored from official release v2.
The following commands document the complete path for another machine:

```bash
python -m scripts.fetch_runbugrun
lrunzip -p 2 -o data/raw/runbugrun.sql data/raw/runbugrun.sql.lrz
python -m scripts.restore_db data/raw/runbugrun.sql data/raw/runbugrun.db
python -m data.extract_runbugrun --db data/raw/runbugrun.db --output data/prepared/pilot --limit 100
python -m data.filter_and_split --input data/prepared/pilot --output data/prepared/verified
python -m data.build_dpo_weak_pairs --input data/prepared/verified/train.jsonl --output data/prepared/verified/dpo-weak.jsonl
```

Locally, substitute `vendor/lrzip/usr/bin/lrunzip` for `lrunzip` if needed.
Restore refuses to overwrite an existing database. Verification requires >=20
execution-verified source pairs and nonempty train/dev/test problem groups.
Inspect statement quality before scaling. Run `python -m data.audit_overlap`
after preparing HumanEvalPack to audit exact/near code overlap; semantic overlap
and pretraining contamination still require review. Original dataset problem IDs overlap
across predefined splits: this project makes fresh problem-group splits.

Pilot manifests are small integration data, not the promised 1,000–2,000 training
examples. To scale, extract e.g. `--limit 3000` into `data/prepared/scale-source`
and verify into `data/prepared/scale-verified`, then update configs to those
manifests and tests BEFORE freezing your experiment. Report actual accepted
counts; filters may exclude many rows. Never replace splits between arms.

## Starting full SFT

See [SFT_DETAILS.md](SFT_DETAILS.md) for the exact 2,000-pair dataset, training
settings, token counts, and mandatory M0-before/M1-after evaluation workflow.

The full-run configuration is `configs/full-sft.yaml`. It requires at least
1,000 verified training rows, uses one epoch, and starts a fresh adapter on the
pinned untouched Qwen base. It does not continue the small SFT pilot.

The scale preparation used for this run is:

```bash
python -m data.extract_runbugrun --db data/raw/runbugrun.db --output data/prepared/scale-source --limit 4000
python -m data.filter_context --input data/prepared/scale-source --output data/prepared/scale-context
python -m data.filter_and_split --input data/prepared/scale-context --output data/prepared/scale-verified --workers 4 --preserve-splits data/prepared/verified --min-verified 1300 --max-train 2000 --max-dev 200 --max-test 200
python -m data.audit_overlap --train data/prepared/scale-verified/train.jsonl --output results/scale-overlap-audit.json
python -m scripts.run_sft_job --config configs/full-sft.yaml --run-dir results/full-sft-RUN_ID
```

Use a fresh run directory and checkpoint destination when repeating training.
The supervisor freezes the resolved configuration and first evaluates untouched
M0 on the 200 held-out RunBugRun pairs and all 164 HumanEvalPack tasks. It then
trains SFT, performs a dev-set adapter reload check, and evaluates M1 on those
same benchmarks. Process IDs, dataset/code hashes and stage transitions are
recorded in `status.json`. `results/active-sft.json` points to the current run. During
training, monitor its `train.log`; a completed training/reload job still requires
independent execution evaluation on the frozen held-out set. Keep the Gradio
demo stopped while training so it does not retain a second model on the GPU.

The context filter uses the pinned Qwen tokenizer, reserves 512 inference tokens,
and requires the full prompt plus either source response to fit 2,048 tokens.
Its English/completeness cue filter is a heuristic; sample statements are
inspected separately. Existing pilot problem assignments are preserved so prior
dev/test problems cannot move into full-run training. Verification with multiple
workers preserves source order for deterministic deduplication and splitting.

## Baseline and training

Run from repository root. All training entry points check hashed execution gates.
The baseline and evaluator use the same prompt, budget, greedy decoding and tests.

```bash
python -m evaluation.benchmark --arm M0 --output results/M0.jsonl
python -m training.train_sft --validate-only
python -m training.train_sft
python -m evaluation.benchmark --arm M1 --adapter checkpoints/M1 --output results/M1.jsonl
python -m scripts.merge_sft
python -m training.train_dpo --validate-only
python -m training.train_dpo
python -m evaluation.benchmark --arm M2 --model checkpoints/M1-merged --adapter checkpoints/M2 --output results/M2.jsonl
```

SFT uses response-only loss, NF4 QLoRA r16/alpha32, batch 1, accumulation 8 and
2048 tokens. Rows exceeding the prompt+response budget are dropped and counted;
no target is truncated. DPO constructs chosen/rejected directly from the same
verified row. Its rejected answer is the original bug, not a plausible attempted
patch. DPO logs chosen/rejected log probabilities in trainer metadata.

`merge_sft` merges M1 at fp16 on CPU using the unquantized base, avoiding a second
GPU copy. Each sibling training arm loads this same merge and attaches a fresh
adapter. Disabling that new adapter yields frozen M1 for the reference. M2 and
M3 remain separate. Merge/adapter rounding is a numerical limitation to inspect.

A short integration smoke can use `--max-steps 2 --limit 8 --output
checkpoints/pilot/M1`; these flags do not constitute a completed SFT experiment.
Existing local integration checkpoints live under `checkpoints/pilot/`.
`configs/pilot-sft.yaml`, `configs/pilot-dpo.yaml`, and `configs/pilot-grpo.yaml`
keep these separate from full-run outputs. The GRPO pilot limits completions to
256 tokens; held-out evaluation remains 512 tokens for all arms. To reproduce
that gate use `python -m evaluation.variance --config configs/pilot-grpo.yaml
--output results/pilot-variance.json`, then `python -m training.train_grpo
--config configs/pilot-grpo.yaml`. Use new output paths when rerunning.

Use a separate output and config for each pilot. Training refuses to overwrite
an existing saved adapter. Config paths/data paths resolve from repository root.

## GRPO gates

```bash
python -m evaluation.variance --tasks 30 --generations 4
python -m training.train_grpo --validate-only
python -m training.train_grpo
python -m evaluation.benchmark --arm M3 --model checkpoints/M1-merged --adapter checkpoints/M3 --output results/M3.jsonl
```

The variance check uses distinct TRAIN problems and M1 stochastic generations;
>=20% of 30+ groups must have mixed rewards. A failed gate stops GRPO. The default
GRPO run is a 20-step integration pilot; successful training/save/reload and
bounded VRAM/time must be verified before scaling. Batch 1 × accumulation 4
supports four generations; generation count is not microbatch size. No vLLM or
hard-negative generation is required. Reward traces retain completions and
execution results for inspection. No learned process-reward model is used.

## External evaluation and local demo

```bash
python -m evaluation.humanevalpack_adapter --sanity
python -m evaluation.benchmark --benchmark humanevalpack --data data/external/humanevalpack.jsonl --arm M0 --output results/HEP-M0.jsonl
python -m evaluation.compare results/M0.summary.json results/M1.summary.json results/M2.summary.json results/M3.summary.json
python app.py --port 7860
```

Repeat external evaluation separately for each saved arm, using its model/adapter
arguments as above. HumanEvalPack is evaluation-only and must not select training
hyperparameters. The adapter checks all 164 gold programs before scoring; hidden
assertions stay in the grader and candidate function calls use JSON transport.
Possible model pretraining contamination is unresolved. Cross-source semantic
overlap is not proven absent; do not claim uncontaminated benchmark generalization.

For the integration adapters, use `python app.py --checkpoint-root checkpoints/pilot`.
Gradio listens on localhost with no public sharing. It shows available full-run
checkpoints and held-out tasks only. Individual examples are demonstrations,
not aggregate evidence. Final reports must distinguish first-attempt repair
success from sequential feedback-guided agent@3. See [EVAL_PROTOCOL.md](EVAL_PROTOCOL.md).

## Sources and scope

- [RunBugRun](https://github.com/giganticode/run_bug_run), official v2 paired submissions/tests.
- [Qwen model](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct), pinned commit in `configs/base.yaml`.
- [TRL](https://huggingface.co/docs/trl/), locally pinned to 0.23.1 and inspected APIs.
- [PEFT](https://huggingface.co/docs/peft/) for LoRA/QLoRA.
- [HumanEvalPack](https://huggingface.co/datasets/bigcode/humanevalpack), evaluation only.

This is not a Repair-R1/ReCode reproduction. DebugBench and hard-negative DPO
are excluded. Tests in `tests/` use small harness fixtures, never synthetic
experimental training data. Dataset/model artifacts remain local and ignored.

## Automatic DPO after M1 evaluation

The current DPO supervisor is queued behind successful completion of both M1
benchmarks. `results/active-dpo.json` points to its live status. It merges the
full M1 checkpoint, trains a fresh M2 adapter using 1,000 weak gold-vs-original-bug
preferences for one epoch (125 optimizer updates), checks reload, and evaluates
M2 on the same frozen RunBugRun/HumanEvalPack sets. The configuration is
`configs/full-dpo.yaml`; outputs are in `checkpoints/M2` and its run directory.
A failed upstream stage stops the handoff for inspection. GRPO is not queued by
this supervisor.

## Overnight automation and controller comparison

[OVERNIGHT_RUN.md](OVERNIGHT_RUN.md) describes the live DPO recovery, independent
M1-rooted GRPO gates/pilot/200-step run, evaluations, status pointers and bounded
failure handling. [AGENT_V3.md](AGENT_V3.md) describes the separate three-code-call,
two-diagnosis-call controller and queued v1/v2/v3 dev comparison. The original
v1/v2-only probe was superseded before it used the GPU.
