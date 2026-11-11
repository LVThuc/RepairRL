import pytest
from agent.sandbox import Sandbox
from evaluation.runner import grade_program, output_matches, parse_program


@pytest.fixture(scope="module")
def sandbox():
    box = Sandbox(timeout=0.8, output_bytes=4096)
    box.check()
    return box


def test_output_normalization():
    assert output_matches("1  2.0\nword", "1.00001\t2\nword\n", "unknown")
    assert not output_matches("1\n2", "1 2", "unknown")
    assert not output_matches("1.00000000", "1.00000002", "p03882")
    assert not output_matches("yes", "YES", "unknown")


def test_gold_bug_and_empty_suite(sandbox):
    suite = [{"id": 1, "input": "2\n", "output": "4"}, {"id": 2, "input": "3", "output": "6"}]
    assert grade_program("print(int(input())*2)", suite, "unit", sandbox).success
    failed = grade_program("print(int(input())+2)", suite, "unit", sandbox)
    assert failed.passed == 1 and not failed.success
    with pytest.raises(ValueError):
        grade_program("print(1)", [], "unit", sandbox)


def test_bad_syntax_and_fences(sandbox):
    assert parse_program("```python\nprint(1)\n```") == "print(1)"
    with pytest.raises(ValueError):
        parse_program("explanation\n```python\nprint(1)\n```")
    grade = grade_program("def broken(", [{"id": 0, "input": "", "output": ""}], "unit", sandbox)
    assert grade.category == "invalid_code" and not grade.valid


def test_timeout_output_and_runtime(sandbox):
    assert sandbox.run("while True: pass").status == "timeout"
    assert sandbox.run("while True: print('x'*1000)").status == "output_limit"
    assert sandbox.run("raise RuntimeError('bad')").status == "runtime_error"


def test_sandbox_hides_host_and_secrets(sandbox, tmp_path, monkeypatch):
    secret = tmp_path / "secret"
    secret.write_text("private")
    monkeypatch.setenv("REPAIRRL_TEST_SECRET", "must-not-leak")
    run = sandbox.run(f"import os; print(os.path.exists({str(secret)!r})); print(os.getenv('REPAIRRL_TEST_SECRET')); print(os.getuid())")
    assert run.status == "ok" and run.stdout.splitlines() == ["False", "None", "65534"]
    assert sandbox.run("open('/work/candidate.py','w').write('changed')").status == "runtime_error"


def test_sandbox_no_network_no_forks(sandbox):
    code = "import socket\ns=socket.socket()\ns.settimeout(.1)\ntry: s.connect(('1.1.1.1',80)); print('connected')\nexcept OSError: print('blocked')"
    assert sandbox.run(code).stdout.strip() == "blocked"
    run = sandbox.run("import os\ntry: os.fork(); print('forked')\nexcept OSError: print('blocked')")
    assert run.stdout.strip() == "blocked"


def test_feedback_does_not_echo_program_output(sandbox):
    grade = grade_program("print('attacker-controlled secret')", [{"id": 1, "input": "hidden", "output": "hidden-answer"}], "unit", sandbox)
    assert "attacker" not in grade.feedback() and "hidden" not in grade.feedback()


def test_candidate_cannot_spoof_sandbox_control(sandbox):
    run = sandbox.run("import sys; print('bwrap: spoofed setup error', file=sys.stderr); print('done')")
    assert run.status == "ok" and run.stdout.strip() == "done"
