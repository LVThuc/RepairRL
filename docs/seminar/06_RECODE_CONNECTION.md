# Connection to ReCode

Primary source: [ReCode, ACL 2026](https://aclanthology.org/2026.acl-long.2031/).

ReCode connects program repair with execution correctness and learned reasoning-process rewards. Its approach includes contrastive reasoning preference learning, a reasoning-process reward scorer, and correctness-gated use of that process signal during policy optimization.

RepairRL takes inspiration from execution-aware post-training, but implements a smaller experiment:

| Component | RepairRL implementation |
|---|---|
| Correctness signal | Execute generated programs against trusted tests |
| Policy optimization | TRL GRPO with an outcome reward |
| Dense signal | Test-pass fraction, clean-exit fraction, all-pass bonus |
| Learned process reward | Not implemented |
| Contrastive reasoning preference training | Not implemented |
| Explicit reasoning-trace targets | Not used; model outputs code |
| Full consistency-gated method | Not reproduced |

Our DPO preferences compare complete gold programs against verified failed M1 repair candidates. They are not ReCode's reasoning preferences. Likewise, the former diagnosis agent is an inference controller, not a learned process-reward model.

For the seminar, say: **“RepairRL studies a practical outcome-reward variant motivated by execution-aware repair research.”** Keep the paper's measured results separate from RepairRL's results. Neither our small experiment nor its current scores establish reproduction or an algorithm-wide ranking.

See the [technical report](TECHNICAL_REPORT.md) for actual settings, completed evaluations and limitations.
