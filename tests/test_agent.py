from agent.repair import repair_task
from evaluation.runner import Grade
from evaluation.benchmark import summarize


def test_agent_stops_on_success_and_retries_current_candidate():
    prompts = []
    responses = iter(["print(1)", "print(2)"])
    def generate(prompt):
        prompts.append(prompt)
        return {"text": next(responses), "tokens": 4, "seconds": 0.1}
    def grader(code):
        good = "2" in code
        return Grade(int(good), 1, "pass" if good else "incorrect_output", True, [])
    task = {"id": "u", "problem_id": "u", "statement": "Print 2", "buggy_code": "print(0)"}
    row = repair_task(task, generate, grader)
    assert row["attempts"] == 2 and row["agent_at_3"] and not row["pass_at_1"]
    assert "print(1)" in prompts[1][1]["content"]
    assert "0/1 tests passed" in prompts[1][1]["content"]
    metrics = summarize([row])
    assert metrics["pass@1"] == 0 and metrics["agent@3"] == 1
