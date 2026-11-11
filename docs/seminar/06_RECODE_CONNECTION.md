# 06 — ReCode: the research connection and its boundary

## Verified paper reference

Fan, Zhang, Chen and Liu, ACL 2026: [ReCode: Reinforcing Code Generation with
Reasoning-Process Rewards](https://aclanthology.org/2026.acl-long.2031/).
The [authoritative PDF](https://aclanthology.org/2026.acl-long.2031.pdf), especially
§§2.1–2.2, defines the method. The [official implementation](https://github.com/ZJU-CTAG/ReCode)
provides separate reward-model training and RL paths; RepairRL did not run them.

### Paper facts for the presenter

ReCode trains a learned evaluator of reasoning using ordered improved/original/
degraded reasoning variants. It then combines format and execution signals with
that evaluator, allowing the reasoning score to contribute only when execution
succeeds. This guards against optimizing an attractive reasoning score while
producing incorrect code. The additional signal can distinguish successful
solutions whose execution rewards would otherwise tie. The paper's responses
contain separate reasoning and answer sections. Its 7B policy result averages
50.4% to 58.5% across the reported generation benchmarks: +8.1 percentage points,
or 16.1% relative, not +16.1 points. LCB-RB evaluates reasoning preferences; it is
not a repair-patch hard-negative dataset. These are paper results, not RepairRL
measurements. [Source: paper §§2–5](https://aclanthology.org/2026.acl-long.2031.pdf).

The paper's gated reward is:

```text
R = R_format + R_outcome + I(R_outcome == 1) * R_process
```

## What the repository actually implements

RepairRL's [`evaluation/reward.py`](../../evaluation/reward.py) and
[`evaluation/reward_v2.py`](../../evaluation/reward_v2.py) calculate rewards from
test outcomes and execution categories. No neural network scores a reasoning
trace. Neither original nor active GRPO training asks for `<think>`/`<answer>`
sections: the training response is a full replacement program.

| Feature | RepairRL implementation status |
|---|---|
| CRPL preference synthesis over reasoning | Not implemented; DPO pairs rank complete programs |
| Learned reasoning-process reward model | Not trained, loaded, queried or distilled |
| CG-GRPO process-score gate | Not implemented; there is no process score to gate |
| Execution-based feedback | Implemented with the external trusted grader |
| Group-relative policy optimization | Implemented through TRL GRPOTrainer |
| Separate diagnosis/planning at inference | Implemented in V3/V3.1, outside training |
| Reasoning benchmark evaluation | LCB-RB not used; repair evaluation uses RunBugRun and HumanEvalPack |

This is why “ReCode-inspired outcome-reward repair experiment” is defensible,
while “ReCode reproduction,” “CG-GRPO implemented,” or “learned process rewards”
would be inaccurate descriptions of the project.

## The practical bridge: why variance matters

In the original RepairRL run, 114 of 200 training groups had identical rewards.
Suppose four programs all pass the selected suite. Our original reward assigns
each 1.05; the dense second-round reward assigns each 1.0. Relative advantages
cannot rank those programs by their different reasoning quality. Adding a
normal-exit term does not solve that particular tie.

Likewise, four normally terminating wrong programs can all receive the same
partial score. A binary or coarse outcome signal may provide little useful
within-group discrimination. That is a measured optimization issue in our own
run, even though it does not identify the sole cause of GRPO's weak results.

Do not confuse three different objects:

1. **Execution reward:** an external measurement of program behavior on selected tests.
2. **Diagnosis text:** an inference-time hypothesis about a fault and proposed change.
3. **Learned process reward:** an absent component that would assign a learned score
   to reasoning and influence training gradients.

Generating an articulate diagnosis does not create the third object. Our GRPO
reward callable never consumes the diagnosis JSON, and our policy training does
not optimize its logical quality.

## How to discuss future work accurately

A future process-reward extension would require new data contracts, a trained
scorer, response parsing that separates reasoning from executable code, and
explicit checks for score exploitation. It would also need ablations against
the existing outcome-only baselines, not just a new experiment name.

That extension is not present in any preserved checkpoint. The active longer
GRPO run changes outcome reward, precision, sampling and update budget. It should
be evaluated on those terms before attributing improvements to a research
mechanism that was never implemented.

For slides, spend roughly four minutes here after showing our real variance and
repair results. The strongest connection is a concrete missing learning signal,
not a claim that the project matches the paper's model size, datasets or scores.
