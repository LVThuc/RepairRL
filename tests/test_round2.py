"""Scientific contracts: no hidden feedback, no prompt substitution, bounded calls."""
import json
from agent.repair_v31 import repair_task_v31
from evaluation.runner import Grade
from evaluation.reward_v2 import dense_reward
from scripts.round2_rollouts import select_negative


def test_dense_reward_preserves_partial_credit_and_allpass_dominates():
    partial = Grade(1, 2, "runtime_error", True, [{"category": "pass"}, {"category": "runtime_error"}])
    wrong = Grade(0, 2, "incorrect_output", True, [{"category": "incorrect_output"}] * 2)
    full = Grade(2, 2, "pass", True, [{"category": "pass"}] * 2)
    assert dense_reward(full) == 1.0
    assert dense_reward(full) > dense_reward(partial) > dense_reward(wrong) > 0
    assert dense_reward(Grade(0, 2, "invalid_code", False, [])) == 0
    assert dense_reward(partial, "correctness") == 0.5


def test_hard_negative_rejects_unchanged_bug_gold_runtime_and_duplicates():
    row = {"buggy_code": "print(1)", "fixed_code": "print(2)"}
    def sample(code, fraction, category="incorrect_output", index=0):
        return {"sample_index": index, "generation": {"text": code}, "grade":
                Grade(int(fraction * 10), 10, category, True, []).to_dict()}
    samples = [sample("print(1)", .8), sample("# copy\nprint(2)", .9),
               sample("print(3)", .5, index=2), sample("print(4)", .7, index=3),
               sample("print( 3 )", .5), sample("1/0", 0, "runtime_error")]
    assert [s["generation"]["text"] for s in select_negative(row, samples)] == ["print(4)", "print(3)"]


def test_v31_carries_plan_measured_edit_and_redacts_feedback():
    task = {"id": 1, "problem_id": 1, "statement": "Compute the requested number", "buggy_code": "print(0)"}
    plan = {"fault": "wrong operation", "evidence": "incorrect output", "change": "replace addition with multiplication"}
    class Generator:
        def __init__(self):
            self.codes = iter(["print(1)", "print(2)", "print(3)"])
            self.prompts, self.plans = [], []
        def __call__(self, prompt):
            self.prompts.append(prompt)
            return {"text": next(self.codes), "tokens": 4, "seconds": 0}
        def sample_retry(self, prompt, seed, temperature):
            assert temperature == .4
            return self(prompt)
        def diagnose(self, prompt, seed, max_tokens):
            self.plans.append(prompt)
            return {"text": json.dumps(plan), "tokens": 20, "seconds": 0}
    g = Generator()
    scores = iter([8, 1, 9])
    def grader(raw):
        return Grade(next(scores), 10, "incorrect_output", True,
                     [{"category": "incorrect_output", "stdout": "PRIVATE_INPUT", "stderr": "PRIVATE_EXPECTED"}])
    result = repair_task_v31(task, g, grader)
    assert result["attempts"] == 3 and result["diagnosis_calls"] == 2
    assert result["candidate"] == "print(3)"
    assert all(t["plan_in_repair_prompt"] for t in result["trace"][1:])
    assert "print(1)" in g.prompts[2][1]["content"]
    assert "Previous diagnosis and edit outcome" in g.plans[1][1]["content"]
    assert "PRIVATE_INPUT" not in str(g.prompts + g.plans)
    assert "PRIVATE_EXPECTED" not in str(g.prompts + g.plans)
