from demo_app import render_changes
from demo_problems import PROBLEMS


def test_model_text_cannot_inject_html_into_change_view():
    response = 'value = "<img src=x onerror=alert(1)>"'
    view = render_changes('value = "old"', response)
    assert "<img" not in view
    assert "&lt;img" in view
    assert "diff-added" in view and "diff-removed" in view


def test_invalid_response_is_not_presented_as_a_patch():
    view = render_changes("x = 1", "This is an explanation, not Python!")
    assert "not a valid Python program" in view
    assert "diff-added" not in view


def test_unchanged_response_does_not_claim_a_fix():
    assert "No code changes" in render_changes("x = 1\n", "x = 1\n")


def test_changes_are_against_submitted_code_not_a_gold_reference():
    view = render_changes("original = 99\nkeep = 1", "original = 100\nkeep = 1")
    assert "original = 99" in view and "original = 100" in view
    assert "+1 added" in view and "−1 removed" in view


def test_demo_has_six_unique_cases_and_valid_latex_commands():
    assert len(PROBLEMS) == 6
    assert len({(v["benchmark"], v["id"]) for v in PROBLEMS.values()}) == 6
    for problem in PROBLEMS.values():
        assert problem["description"].startswith("## ")
        assert "\\\\" not in problem["description"]
