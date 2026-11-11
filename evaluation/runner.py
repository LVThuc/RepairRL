"""Trusted stdin/stdout grading; tests never enter the candidate filesystem."""
import ast
from dataclasses import dataclass, asdict
from decimal import Decimal
import json
from pathlib import Path
import re
from agent.sandbox import Sandbox

# Ported from RunBugRun ProgramOutputMatcher at 374251a9d65410f37e1136049cb7ff5dcca3d0ae.
FLOAT_EPS = json.loads(Path(__file__).with_name("float_eps.json").read_text())
NUMBER = re.compile(r"-?\d+(?:\.\d+)?", flags=re.ASCII)


def parsed_lines(s):
    result = []
    for line in s.split("\n"):
        values, pos = [], 0
        while pos < len(line):
            if line[pos] in " \t\r\f\v":
                pos += 1
                continue
            match = NUMBER.match(line, pos)
            if match:
                values.append(Decimal(match[0]))
            else:
                match = re.compile(r"\S+").match(line, pos)
                values.append(match[0])
            pos = match.end()
        result.append(values)
    return result


def output_matches(expected, actual, problem_id):
    expected, actual = expected.strip(), actual.replace("\x00", "").strip()
    if expected == actual:
        return True
    left, right = parsed_lines(expected), parsed_lines(actual)
    if len(left) != len(right):
        return False
    eps = Decimal(FLOAT_EPS.get(str(problem_id), "1e-4"))
    return all(len(a) == len(b) and all(
        abs(x - y) <= eps if isinstance(x, Decimal) and isinstance(y, Decimal) else x == y
        for x, y in zip(a, b)) for a, b in zip(left, right))


def parse_program(raw):
    if not isinstance(raw, str) or len(raw.encode()) > 131072:
        raise ValueError("Missing or oversized program")
    code = raw.strip()
    if code.startswith("```"):
        match = re.fullmatch(r"```(?:python|py|python3)?\s*\n(.*?)\n```", code, re.DOTALL)
        if not match:
            raise ValueError("Expected one complete Python program")
        code = match[1].strip()
    if not code:
        raise ValueError("Empty program")
    try:
        ast.parse(code)
    except (SyntaxError, ValueError, RecursionError) as e:
        raise ValueError("Invalid Python syntax") from e
    return code


@dataclass
class Grade:
    passed: int
    total: int
    category: str
    valid: bool
    tests: list

    @property
    def success(self):
        return self.total > 0 and self.passed == self.total and self.category == "pass"

    @property
    def fraction(self):
        return self.passed / self.total if self.total else 0.0

    def feedback(self):
        # No expected values, test inputs, traceback snippets or attacker-controlled strings.
        return f"{self.passed}/{self.total} tests passed. Result: {self.category}."

    def to_dict(self):
        return asdict(self) | {"success": self.success, "fraction": self.fraction}


def grade_program(raw, tests, problem_id, sandbox=None):
    if not tests:
        raise ValueError("Cannot grade an empty suite")
    try:
        code = parse_program(raw)
    except ValueError:
        return Grade(0, len(tests), "invalid_code", False, [])
    sandbox = sandbox or Sandbox()
    trace, passed = [], 0
    categories = []
    for test in tests:
        run = sandbox.run(code, test["input"])
        category = run.status
        if category == "ok":
            category = "pass" if output_matches(test["output"], run.stdout, problem_id) else "incorrect_output"
        passed += category == "pass"
        categories.append(category)
        trace.append(asdict(run) | {"test_id": test["id"], "category": category})
    category = next((c for c in ("timeout", "output_limit", "runtime_error", "incorrect_output") if c in categories), "pass")
    return Grade(passed, len(tests), category, True, trace)
