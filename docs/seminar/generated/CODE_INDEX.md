# Implementation index

Line numbers refer to this checkout; hashes identify exact files.

## [repairrl/common.py](../../../repairrl/common.py)

SHA256: `87f615de62bf7f0d599cfa4c36c4cc9f636ce2dd190feae5d93252cc7dfabd0e`

- [`messages`](../../../repairrl/common.py#L11) — line 11
- [`read_jsonl`](../../../repairrl/common.py#L20) — line 20
- [`write_jsonl`](../../../repairrl/common.py#L25) — line 25
- [`digest`](../../../repairrl/common.py#L33) — line 33
- [`load_config`](../../../repairrl/common.py#L41) — line 41

## [repairrl/model.py](../../../repairrl/model.py)

SHA256: `85bdf70aee568f95500bae45acb2540aa0398819ffb42620e6a7f4964b960b8a`

- [`load_model`](../../../repairrl/model.py#L6) — line 6
- [`Generator`](../../../repairrl/model.py#L27) — line 27

## [data/extract_runbugrun.py](../../../data/extract_runbugrun.py)

SHA256: `571ef0bf6aadecfa3ffe277405c35cf506835002c8412c100a3311986795fe84`

- [`extract`](../../../data/extract_runbugrun.py#L10) — line 10
- [`main`](../../../data/extract_runbugrun.py#L68) — line 68

## [data/filter_and_split.py](../../../data/filter_and_split.py)

SHA256: `55c391c2dff544e20aa5271e63739d49102680d4b286f795da27eba8fa18cc2e`

- [`shingles`](../../../data/filter_and_split.py#L17) — line 17
- [`code_key`](../../../data/filter_and_split.py#L27) — line 27
- [`split_rows`](../../../data/filter_and_split.py#L35) — line 35
- [`verify_pair`](../../../data/filter_and_split.py#L53) — line 53
- [`main`](../../../data/filter_and_split.py#L61) — line 61

## [data/filter_context.py](../../../data/filter_context.py)

SHA256: `11a5e7fbdd85294407460cdfe04ad123fe4cd0678f6f6c3f4eda47664fc8b722`

- [`main`](../../../data/filter_context.py#L10) — line 10

## [training/common.py](../../../training/common.py)

SHA256: `55ad618e59ac1b05e882c6f31ca38641485bfe6755fd009e2576e8b03eb124d7`

- [`arguments`](../../../training/common.py#L11) — line 11
- [`check_data`](../../../training/common.py#L30) — line 30
- [`lora_config`](../../../training/common.py#L61) — line 61
- [`setup_model`](../../../training/common.py#L68) — line 68
- [`format_rows`](../../../training/common.py#L80) — line 80
- [`trainer_options`](../../../training/common.py#L113) — line 113
- [`run_and_save`](../../../training/common.py#L124) — line 124

## [training/train_sft.py](../../../training/train_sft.py)

SHA256: `2f56807d0b4419fe3bcee5e2b840c6bf72931b0913723ee8d42166ee02e5a214`

- [`main`](../../../training/train_sft.py#L4) — line 4

## [training/train_dpo.py](../../../training/train_dpo.py)

SHA256: `b012c507359812a1e51a8cb875348913a8ab38912cfe3c30dff461fa6e809df1`

- [`main`](../../../training/train_dpo.py#L5) — line 5

## [training/train_grpo.py](../../../training/train_grpo.py)

SHA256: `c1de4153bc1483285ab9e32b9f172e9fa1bb41ec522490619d44ffca0e74bc85`

- [`main`](../../../training/train_grpo.py#L8) — line 8

## [training/resume.py](../../../training/resume.py)

SHA256: `1f6f328d1f306dbec23fe23027f14b3d842c77cfb81e462a321a57e128c9f848`

- [`prepare`](../../../training/resume.py#L16) — line 16
- [`main`](../../../training/resume.py#L54) — line 54

## [evaluation/runner.py](../../../evaluation/runner.py)

SHA256: `9825c3ff1f81fbfd9abe57826e413b0a52a069a11b37a8f8256d829467471020`

- [`parsed_lines`](../../../evaluation/runner.py#L15) — line 15
- [`output_matches`](../../../evaluation/runner.py#L34) — line 34
- [`parse_program`](../../../evaluation/runner.py#L47) — line 47
- [`Grade`](../../../evaluation/runner.py#L66) — line 66
- [`grade_program`](../../../evaluation/runner.py#L89) — line 89

## [evaluation/reward.py](../../../evaluation/reward.py)

SHA256: `6f6ae502047925445884bea477d18e716628662d7e4c324b96f20482a400afc8`

- [`outcome_reward`](../../../evaluation/reward.py#L9) — line 9
- [`ExecutionReward`](../../../evaluation/reward.py#L15) — line 15

## [evaluation/reward_v2.py](../../../evaluation/reward_v2.py)

SHA256: `84e01fa90a7af873de269a9414807971dbe5dfdfd7022c400d6e106796412ff4`

- [`dense_reward`](../../../evaluation/reward_v2.py#L8) — line 8
- [`RemoteExecutionReward`](../../../evaluation/reward_v2.py#L17) — line 17

## [evaluation/benchmark.py](../../../evaluation/benchmark.py)

SHA256: `972405666159fe805edad48013daf45b1ab09bcd42bfb39ba864f6651321a740`

- [`summarize`](../../../evaluation/benchmark.py#L12) — line 12
- [`main`](../../../evaluation/benchmark.py#L28) — line 28

## [evaluation/round2.py](../../../evaluation/round2.py)

SHA256: `0327448d18541a59e60b57ca372d2634b852bb9dec859b5ddf699b5403dea3db`

- [`dev_rows`](../../../evaluation/round2.py#L14) — line 14
- [`attached_generator`](../../../evaluation/round2.py#L23) — line 23
- [`evaluate`](../../../evaluation/round2.py#L31) — line 31
- [`main`](../../../evaluation/round2.py#L65) — line 65

## [agent/sandbox.py](../../../agent/sandbox.py)

SHA256: `aec586e7a1edd507723eeffd32b2e0fb0572f885171f61e6f2e5a6544be86471`

- [`SandboxUnavailable`](../../../agent/sandbox.py#L15) — line 15
- [`Execution`](../../../agent/sandbox.py#L20) — line 20
- [`Sandbox`](../../../agent/sandbox.py#L28) — line 28

## [agent/repair.py](../../../agent/repair.py)

SHA256: `419b33f21dd981a8ea5f175405b3cd8e515956018c126692697640c20c838d64`

- [`repair_task`](../../../agent/repair.py#L5) — line 5

## [agent/repair_v3.py](../../../agent/repair_v3.py)

SHA256: `b0df02063e224c8bc991d7d4ac98efa4f078f337ae50f3b83f95b89650170516`

- [`DiagnosisGenerator`](../../../agent/repair_v3.py#L10) — line 10
- [`parse_diagnosis`](../../../agent/repair_v3.py#L22) — line 22
- [`history_text`](../../../agent/repair_v3.py#L37) — line 37
- [`fits`](../../../agent/repair_v3.py#L44) — line 44
- [`repair_task_v3`](../../../agent/repair_v3.py#L52) — line 52

## [agent/repair_v31.py](../../../agent/repair_v31.py)

SHA256: `4fbafd1098ba6d3115cf6b51d0341e4a3066e4128f66d6b4ef5e05a6a49ae2e1`

- [`edit_summary`](../../../agent/repair_v31.py#L12) — line 12
- [`planned_prompt`](../../../agent/repair_v31.py#L16) — line 16
- [`repair_task_v31`](../../../agent/repair_v31.py#L30) — line 30

## [scripts/round2_train.py](../../../scripts/round2_train.py)

SHA256: `4b33d6ff7cf4321e00e0090f256449c93bc37ab6c4d308cf4aaa4160122f272f`

- [`atomic_json`](../../../scripts/round2_train.py#L12) — line 12
- [`hard_rows`](../../../scripts/round2_train.py#L20) — line 20
- [`format_dataset`](../../../scripts/round2_train.py#L50) — line 50
- [`variance_probe`](../../../scripts/round2_train.py#L72) — line 72
- [`main`](../../../scripts/round2_train.py#L112) — line 112

## [scripts/round2_fast_rollouts.py](../../../scripts/round2_fast_rollouts.py)

SHA256: `0c9a62d398df7310771ace01e5a21ccfe77f05098c79d580c3e526d21fa1d1e9`

- [`main`](../../../scripts/round2_fast_rollouts.py#L16) — line 16

## [scripts/round2_jobs.py](../../../scripts/round2_jobs.py)

SHA256: `e911bafa0d64ed0f56b0435448a5149949bc6435960180ae4492e20d61167c3f`

- [`main`](../../../scripts/round2_jobs.py#L12) — line 12

## [scripts/round2_finish.py](../../../scripts/round2_finish.py)

SHA256: `1e03802b469fd32bdea36a6485bbe177fc842f59062f5d95fd5ad5cb76f5ee24`

- [`main`](../../../scripts/round2_finish.py#L12) — line 12
