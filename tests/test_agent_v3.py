import json
import sys
from types import SimpleNamespace
import pytest
from agent.repair_v3 import DiagnosisGenerator, parse_diagnosis, repair_task_v3
from evaluation.runner import Grade
from repairrl.common import messages

TASK = {"id": "v3-unit", "problem_id": "unit", "statement": "Print the required result", "buggy_code": "print(0)"}
PLAN = json.dumps({"fault": "Wrong expression", "evidence": "Tests fail", "change": "Correct the expression"})


class Generator:
    def __init__(self, codes, plan=PLAN):
        self.codes = iter(codes)
        self.prompts, self.planner_calls, self.sampled_calls = [], [], []
        self.plan = plan
    def __call__(self, prompt):
        self.prompts.append(prompt)
        return {"text": next(self.codes), "tokens": 4, "seconds": 0.1}
    def diagnose(self, prompt, seed, max_tokens):
        self.planner_calls.append((prompt, seed, max_tokens))
        return {"text": self.plan, "tokens": 20, "seconds": 0.1}
    def sample_retry(self, prompt, seed):
        self.sampled_calls.append(seed)
        return self(prompt)


def test_first_success_has_identical_prompt_and_no_diagnosis():
    g = Generator(["print(1)"])
    r = repair_task_v3(TASK, g, lambda _: Grade(1, 1, "pass", True, []))
    assert g.prompts == [messages(TASK["statement"], TASK["buggy_code"])]
    assert r["code_generation_calls"] == 1 and r["diagnosis_calls"] == 0
    assert not g.planner_calls and r["agent_at_3"]


def test_regression_does_not_replace_best_or_become_next_start():
    g = Generator(["print(1)", "print(2)", "print(3)"])
    scores = iter([8, 2, 4])
    r = repair_task_v3(TASK, g, lambda _: Grade(next(scores), 10, "incorrect_output", True, []))
    assert r["candidate"] == "print(1)" and r["best_attempt"] == 1
    assert r["best_grade"]["fraction"] == 0.8
    assert "print(1)" in g.prompts[2][1]["content"]
    assert "print(2)" not in g.prompts[2][1]["content"]
    assert r["code_generation_calls"] == 3 and r["diagnosis_calls"] == 2
    assert all(call[2] == 96 for call in g.planner_calls)


def test_duplicate_is_rejected_within_budget_and_not_tested_again():
    g = Generator(["print(1)", "# cosmetic\nprint( 1 )", "print(2)"])
    tested = []
    def grade(raw):
        tested.append(raw)
        good = raw == "print(2)"
        return Grade(int(good), 1, "pass" if good else "incorrect_output", True, [])
    r = repair_task_v3(TASK, g, grade)
    assert len(tested) == 2 and len(g.prompts) == 3
    assert r["trace"][1]["duplicate_patch"] and not r["trace"][1]["tested"]
    assert r["trace"][1]["grade"]["total"] == 0
    assert len(g.sampled_calls) == 1 and r["agent_at_3"]
    assert "different fault hypothesis" in g.planner_calls[1][0][1]["content"]


def test_invalid_diagnosis_does_not_create_unbounded_retry_calls():
    g = Generator(["print(1)", "print(2)", "print(3)"], plan="not json")
    r = repair_task_v3(TASK, g, lambda _: Grade(0, 1, "incorrect_output", True, []))
    assert r["code_generation_calls"] == 3 and r["diagnosis_calls"] == 2
    assert all(d["error"] == "invalid_diagnosis_json" for d in r["diagnoses"])


def test_early_recovery_uses_only_one_diagnosis():
    g = Generator(["print(1)", "print(2)"])
    r = repair_task_v3(TASK, g, lambda raw: Grade(int(raw == "print(2)"), 1,
        "pass" if raw == "print(2)" else "incorrect_output", True, []))
    assert r["code_generation_calls"] == 2 and r["diagnosis_calls"] == 1


def test_diagnosis_prompt_never_contains_raw_stderr_or_private_values():
    g = Generator(["print(1)", "print(2)"])
    grade = Grade(0, 1, "runtime_error", True, [{"category": "runtime_error", "stderr":
        'File "/work/candidate.py", line 1\nValueError: PRIVATE_INPUT hidden expected SECRET_ANSWER\n'}])
    repair_task_v3(TASK, g, lambda _: grade, max_attempts=2)
    text = str(g.planner_calls)
    assert "ValueError" in text
    assert "PRIVATE_INPUT" not in text and "SECRET_ANSWER" not in text


@pytest.mark.parametrize("kwargs", [{"max_attempts": 4}, {"max_diagnoses": 3}, {"diagnosis_tokens": 129}])
def test_budget_limits_are_hard(kwargs):
    with pytest.raises(ValueError):
        repair_task_v3(TASK, None, None, **kwargs)


def test_diagnosis_parser_rejects_extra_fields_and_oversized_text():
    assert parse_diagnosis(PLAN)
    assert parse_diagnosis(json.dumps({"fault": "x", "evidence": "x", "change": "x", "code": "print(1)"})) is None
    assert parse_diagnosis(json.dumps({"fault": "x"*241, "evidence": "x", "change": "x"})) is None


def test_diagnosis_wrapper_restores_code_budget_even_on_failure(monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(set_seed=lambda _: None))
    class Failed:
        temperature = 0.3
        max_new_tokens = 512
        def __call__(self, prompt):
            assert self.max_new_tokens == 96 and self.temperature == 0.0
            raise RuntimeError("unit failure")
    base = Failed()
    with pytest.raises(RuntimeError):
        DiagnosisGenerator(base).diagnose([], 42, 96)
    assert base.max_new_tokens == 512 and base.temperature == 0.3
