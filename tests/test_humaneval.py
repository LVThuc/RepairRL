from evaluation.humanevalpack_adapter import grade_humaneval
from agent.sandbox import Sandbox
from repairrl.codec import encode, decode


def test_function_transport_and_hidden_assertions():
    task = {"id": "unit", "entry_point": "double", "test_setup": "",
            "test": "def check(candidate):\n    assert candidate(2) == 4\n    assert candidate(3) == 6\n"}
    box = Sandbox(timeout=1)
    assert grade_humaneval("def double(x): return x*2", task, box).success
    assert not grade_humaneval("def double(x): return x+2", task, box).success
    assert not grade_humaneval("def double(x): raise ValueError()", task, box).success
    attack = "def double(x):\n    import pathlib\n    assert ('candidate'+'('+str(3)+')') not in pathlib.Path('/work/candidate.py').read_text()\n    return x*2"
    assert grade_humaneval(attack, task, box).success


def test_codec_preserves_builtin_containers():
    value = {1: (True, {2, 3}), "x": [None, 2.5]}
    assert decode(encode(value)) == value


def test_wrong_candidate_return_type_is_a_failure_not_a_grader_crash():
    task = {"id": "unit", "entry_point": "numbers", "test_setup": "",
            "test": "def check(candidate):\n    assert sum(candidate()) == 3\n"}
    grade = grade_humaneval("def numbers(): return None", task, Sandbox(timeout=1))
    assert not grade.success and grade.category == "incorrect_output"


def test_humaneval_does_not_hide_sandbox_infrastructure_errors():
    import pytest
    from agent.sandbox import SandboxUnavailable
    class Unavailable:
        def run(self, *args):
            raise SandboxUnavailable("unit infrastructure failure")
    task = {"id": "unit", "entry_point": "numbers", "test_setup": "",
            "test": "def check(candidate):\n    assert candidate() == 3\n"}
    with pytest.raises(SandboxUnavailable):
        grade_humaneval("def numbers(): return 3", task, Unavailable())
