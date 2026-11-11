"""Outcome-only reward; not a learned process reward model."""
import json
import statistics
from collections import defaultdict
from evaluation.runner import grade_program
from agent.sandbox import Sandbox


def outcome_reward(grade, all_pass_bonus=0.05):
    if not grade.valid or any(t["category"] in {"timeout", "runtime_error", "output_limit"} for t in grade.tests):
        return 0.0
    return grade.fraction + (all_pass_bonus if grade.success else 0.0)


class ExecutionReward:
    def __init__(self, suites, log_path, timeout=3.0):
        self.suites = suites
        self.sandbox = Sandbox(timeout=timeout)
        self.sandbox.check()
        self.log_path = log_path
        self.__name__ = "execution_reward"

    def __call__(self, completions, problem_id, **kwargs):
        rewards, grades = [], []
        grouped = defaultdict(list)
        with open(self.log_path, "a") as f:
            for completion, pid in zip(completions, problem_id, strict=True):
                raw = completion[0]["content"] if isinstance(completion, list) else completion
                grade = grade_program(raw, self.suites[str(pid)], pid, self.sandbox)
                value = outcome_reward(grade)
                rewards.append(value)
                grades.append(grade)
                grouped[str(pid)].append(value)
                f.write(json.dumps({"problem_id": pid, "completion": raw, "reward": value,
                                    "grade": grade.to_dict()}) + "\n")
        with open(self.log_path + ".stats.jsonl", "a") as stats:
            stats.write(json.dumps({"n": len(grades),
                "validity": sum(g.valid for g in grades) / len(grades),
                "all_pass_rate": sum(g.success for g in grades) / len(grades),
                "mean_test_fraction": sum(g.fraction for g in grades) / len(grades),
                "group_variances": {pid: statistics.pvariance(values) for pid, values in grouped.items()}}) + "\n")
        return rewards
