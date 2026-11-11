"""Opt-in repair controller; frozen v1 model comparisons keep their original code."""
import ast
import hashlib
import re
from evaluation.runner import parse_program
from repairrl.common import messages

VERSION = "repairrl-agent-v2"
EXCEPTIONS = (
    "SyntaxError", "IndentationError", "TabError", "ZeroDivisionError", "IndexError",
    "KeyError", "ValueError", "TypeError", "NameError", "UnboundLocalError",
    "AttributeError", "RecursionError", "OverflowError", "MemoryError", "EOFError",
    "AssertionError", "StopIteration", "ImportError", "ModuleNotFoundError",
    "RuntimeError", "NotImplementedError", "FileNotFoundError", "PermissionError",
    "TimeoutError", "UnicodeDecodeError",
)
ERROR_TYPE = re.compile(r"^(" + "|".join(EXCEPTIONS) + r")(?:\s*:|\s*$)", re.MULTILINE)
CANDIDATE_LINE = re.compile(r'File "/work/(?:candidate|solution)\.py", line (\d+)')


def fingerprint(raw):
    try:
        value = ast.dump(ast.parse(parse_program(raw)), include_attributes=False)
    except (ValueError, SyntaxError, RecursionError):
        value = raw.strip()
    return hashlib.sha256(value.encode()).hexdigest()


def diagnostic(raw, grade):
    """Allowlisted type and bounded candidate line only: no messages or values."""
    if not grade.valid:
        try:
            parse_program(raw)
        except ValueError as error:
            cause = error.__cause__
            if isinstance(cause, SyntaxError):
                line = cause.lineno
                kind = type(cause).__name__
                return f"{kind} at candidate line {line}. Return a complete, valid Python program."
        return "Output was not a complete, parseable Python program."
    for test in grade.tests:
        if test.get("category") != "runtime_error":
            continue
        # stderr is candidate-controlled. Only static exception names and a line
        # belonging to candidate source may enter the prompt; never raw text.
        stderr = test.get("stderr", "")[:65536]
        errors = ERROR_TYPE.findall(stderr)
        if errors:
            lines = [int(n) for n in CANDIDATE_LINE.findall(stderr)]
            lines = [n for n in lines if 1 <= n <= len(raw.splitlines())]
            location = f" at candidate line {lines[-1]}" if lines else ""
            return f"Redacted runtime observation (untrusted): {errors[-1]}{location}."
    if grade.category == "runtime_error":
        return "The program terminated with a runtime error. Review input handling and operations that may raise exceptions."
    if grade.category == "timeout":
        return "Execution exceeded its time limit. Review termination and algorithmic complexity."
    if grade.category == "output_limit":
        return "Output exceeded its limit. Review printing and loop termination."
    return "Output failed comparison. Re-check the specification, input parsing and boundary cases."


def feedback(raw, grade, history, repeated, include_history=True):
    text = grade.feedback() + "\n" + diagnostic(raw, grade)
    if include_history:
        recent = history[-2:]
        text += "\nAttempts: " + "; ".join(
            f"{a['attempt']}={a['grade']['passed']}/{a['grade']['total']} {a['grade']['category']}"
            for a in recent)
    if repeated:
        text += "\nThe same failed program was submitted again; formatting-only changes do not repair it."
    text += "\nCorrect the remaining fault. Return a changed, complete Python program; no explanation."
    return text


class RetryGenerator:
    """Reuse the existing loader; sample only a stuck final retry, without extra calls."""
    def __init__(self, generator):
        self.generator = generator

    def __getattr__(self, key):
        return getattr(self.generator, key)

    def __call__(self, prompt):
        return self.generator(prompt) | {"temperature": self.generator.temperature}

    def sample_retry(self, prompt, seed, temperature=0.3):
        from transformers import set_seed
        previous = self.generator.temperature
        self.generator.temperature = temperature
        try:
            set_seed(seed)
            return self.generator(prompt) | {"temperature": temperature, "retry_seed": seed}
        finally:
            self.generator.temperature = previous


def retry_seed(task_id, attempt, seed):
    key = f"{seed}:{task_id}:{attempt}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "big")


def repair_task_v2(task, generate, grader, max_attempts=3, seed=42, on_event=None):
    if not 1 <= max_attempts <= 3:
        raise ValueError("Use one to three total attempts")
    code, last_grade, repeated, trace = task["buggy_code"], None, False, []
    seen = {fingerprint(code)}
    for attempt in range(1, max_attempts + 1):
        observation = None
        if last_grade:
            observation = feedback(code, last_grade, trace, repeated)
        prompt = messages(task["statement"], code, observation)
        # Drop history before diagnostics if the fixed context budget is tight.
        tokenizer = getattr(generate, "tokenizer", None)
        if tokenizer and last_grade:
            options = [observation, feedback(code, last_grade, trace, repeated, include_history=False),
                       last_grade.feedback() + "\n" + diagnostic(code, last_grade), last_grade.feedback()]
            for option in options:
                candidate_prompt = messages(task["statement"], code, option)
                n = len(tokenizer.apply_chat_template(candidate_prompt, tokenize=True, add_generation_prompt=True))
                if n + generate.max_new_tokens <= generate.max_context:
                    prompt, observation = candidate_prompt, option
                    break
        # Attempt 1 stays identical to v1. There are at most three code generations.
        sampled = attempt == 3 and repeated and hasattr(generate, "sample_retry")
        if sampled:
            generated = generate.sample_retry(prompt, retry_seed(task["id"], attempt, seed))
        else:
            generated = generate(prompt)
        raw = generated["text"]
        grade = grader(raw)
        key = fingerprint(raw)
        repeated = key in seen and not grade.success
        seen.add(key)
        record = {"attempt": attempt, "generation": generated, "grade": grade.to_dict(),
                  "feedback_sent": observation, "candidate_hash": key,
                  "repeated_failed_program": repeated, "sampled_retry": sampled}
        trace.append(record)
        if on_event:
            on_event(record)
        try:
            code = parse_program(raw)
        except ValueError:
            code = raw
        last_grade = grade
        if grade.success:
            break
    return {"id": task["id"], "problem_id": task["problem_id"], "controller": VERSION,
            "attempts": len(trace), "pass_at_1": trace[0]["grade"]["success"],
            "agent_at_3": trace[-1]["grade"]["success"], "candidate": code, "trace": trace}
