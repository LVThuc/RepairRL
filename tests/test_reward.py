from evaluation.runner import Grade
from evaluation.reward import outcome_reward


def test_reward_ordering_and_failure_gates():
    partial = Grade(9, 10, "incorrect_output", True, [{"category": "incorrect_output"}])
    full = Grade(10, 10, "pass", True, [{"category": "pass"}])
    assert outcome_reward(full) > outcome_reward(partial) > 0
    assert outcome_reward(Grade(0, 10, "invalid_code", False, [])) == 0
    assert outcome_reward(Grade(9, 10, "timeout", True, [{"category": "timeout"}])) == 0
