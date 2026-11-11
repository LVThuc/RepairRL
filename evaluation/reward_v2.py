"""Auditable execution rewards retaining useful partial credit on runtime failures."""
import json
from pathlib import Path
import statistics
from evaluation.grading_service import GradingClient


def dense_reward(grade, variant="dense"):
    if not grade.valid or grade.total <= 0:
        return 0.0
    if variant == "correctness":
        return grade.fraction + 0.05 * grade.success
    clean = sum(t["category"] in {"pass", "incorrect_output"} for t in grade.tests) / grade.total
    return 0.85 * grade.fraction + 0.10 * clean + 0.05 * grade.success


class RemoteExecutionReward:
    def __init__(self, output, variant="dense"):
        self.grader, self.output, self.variant = GradingClient(), Path(output), variant
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.__name__ = "execution_" + variant

    def __call__(self, completions, problem_id, row_id, **kwargs):
        texts = [c[0]["content"] if isinstance(c, list) else c for c in completions]
        grades = self.grader.batch(list(zip(texts, problem_id, strict=True)))
        values = [dense_reward(g, self.variant) for g in grades]
        groups = {}
        with self.output.open("a") as f:
            for text, pid, rid, grade, reward in zip(texts, problem_id, row_id, grades, values, strict=True):
                groups.setdefault(str(rid), []).append(reward)
                f.write(json.dumps({"row_id": rid, "problem_id": pid, "completion": text, "reward": reward,
                                    "grade": grade.to_dict(), "reward_version": self.variant}) + "\n")
        stats = {"n": len(grades), "all_pass": sum(g.success for g in grades) / len(grades),
                 "validity": sum(g.valid for g in grades) / len(grades),
                 "partial_fraction": statistics.mean(g.fraction for g in grades),
                 "reward_mean": statistics.mean(values),
                 "mixed_groups": sum(statistics.pvariance(v) > 1e-12 for v in groups.values()),
                 "groups": len(groups), "group_variances": {k: statistics.pvariance(v) for k, v in groups.items()}}
        with self.output.with_suffix(".stats.jsonl").open("a") as f:
            f.write(json.dumps(stats) + "\n")
        return values
