"""Measure reward diversity before GRPO; never use held-out or HumanEval tasks."""
import argparse
import json
from pathlib import Path
import random
import statistics
from repairrl.common import load_config, digest, messages
from evaluation.runner import grade_program
from evaluation.reward import outcome_reward
from agent.sandbox import Sandbox


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/grpo.yaml")
    p.add_argument("--tasks", type=int, default=30)
    p.add_argument("--generations", type=int, default=4, choices=[4, 8])
    p.add_argument("--output", default="results/variance.json")
    a = p.parse_args()
    if a.tasks < 30:
        p.error("Variance gate requires at least 30 distinct train tasks")
    cfg = load_config(a.config)
    from training.common import check_data
    rows, _ = check_data(cfg)
    # One row per problem avoids treating variants as independent tasks.
    unique = {str(r["problem_id"]): r for r in rows}
    rows = list(unique.values())
    random.Random(cfg["seed"]).shuffle(rows)
    if len(rows) < a.tasks:
        p.error(f"Need {a.tasks} distinct train problems; only {len(rows)} available. Scale verified data first.")
    suites = json.loads(Path(cfg["tests"]).read_text())
    sandbox = Sandbox(timeout=cfg["sandbox_timeout"])
    sandbox.check()
    from repairrl.model import Generator
    generate = Generator(cfg["model"], max_new_tokens=cfg["max_completion_length"],
                         max_context=cfg["max_length"], temperature=0.8, seed=cfg["seed"])
    rows = [r for r in rows if len(generate.tokenizer.apply_chat_template(
        messages(r["statement"], r["buggy_code"]), tokenize=True, add_generation_prompt=True)) <= cfg["max_prompt_length"]]
    if len(rows) < a.tasks:
        p.error(f"Only {len(rows)} distinct train tasks fit the GRPO prompt budget")
    groups = []
    output = Path(a.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix(".rollouts.jsonl").open("w") as f:
        for row in rows[:a.tasks]:
            rewards = []
            for _ in range(a.generations):
                result = generate(messages(row["statement"], row["buggy_code"]))
                grade = grade_program(result["text"], suites[str(row["problem_id"])], row["problem_id"], sandbox)
                reward = outcome_reward(grade)
                rewards.append(reward)
                f.write(json.dumps({"id": row["id"], "generation": result, "grade": grade.to_dict(), "reward": reward}) + "\n")
                f.flush()
            groups.append({"id": row["id"], "rewards": rewards, "variance": statistics.pvariance(rewards)})
            print(f"{len(groups)}/{a.tasks} rewards={rewards}", flush=True)
    mixed = sum(g["variance"] > 1e-12 for g in groups) / len(groups)
    report = {"passed": mixed >= 0.2, "mixed_group_fraction": mixed, "threshold": 0.2,
              "rollout_config": {k: cfg[k] for k in ("seed", "max_completion_length", "max_prompt_length", "sandbox_timeout")},
              "generations": a.generations, "groups": groups, "train_sha256": digest(cfg["train_data"]), "tests_sha256": digest(cfg["tests"]),
              "parent_sha256": digest(Path(cfg["model"]) / "repairrl-merge.json"), "seed": cfg["seed"]}
    output.write_text(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit("NO-GO: fewer than 20% of groups have mixed rewards")


if __name__ == "__main__":
    main()
