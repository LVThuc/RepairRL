from repairrl.common import messages
from evaluation.runner import parse_program


def repair_task(task, generate, grader, max_attempts=3):
    if not 1 <= max_attempts <= 3:
        raise ValueError("Use one to three total attempts")
    code, feedback, trace = task["buggy_code"], None, []
    for attempt in range(1, max_attempts + 1):
        generated = generate(messages(task["statement"], code, feedback))
        grade = grader(generated["text"])
        trace.append({"attempt": attempt, "generation": generated, "grade": grade.to_dict()})
        try:
            code = parse_program(generated["text"])
        except ValueError:
            code = generated["text"]
        if grade.success:
            break
        feedback = grade.feedback()
    return {"id": task["id"], "problem_id": task["problem_id"], "attempts": len(trace),
            "pass_at_1": trace[0]["grade"]["success"], "agent_at_3": trace[-1]["grade"]["success"],
            "candidate": code, "trace": trace}
