import pytest
from data.build_dpo_weak_pairs import weak_pair
from data.filter_and_split import split_rows, code_key, shingles


def test_splits_are_group_disjoint_and_reproducible():
    rows = [{"id": i, "problem_id": f"problem-{i//3}"} for i in range(60)]
    parts = split_rows(rows)
    assert parts == split_rows(rows)
    groups = [{r["problem_id"] for r in part} for part in parts.values()]
    assert sum(map(len, parts.values())) == len(rows)
    assert all(groups[i].isdisjoint(groups[j]) for i, j in ((0, 1), (0, 2), (1, 2)))


def test_weak_pair_uses_same_record_and_no_gold_prompt():
    row = {"id": 1, "problem_id": "unit", "statement": "Return twice the number",
           "buggy_code": "print(int(input()) + 2)", "fixed_code": "print(int(input()) * 2)", "verified": True}
    pair = weak_pair(row)
    assert pair["chosen"][0]["content"] == row["fixed_code"]
    assert pair["rejected"][0]["content"] == row["buggy_code"]
    assert row["fixed_code"] not in str(pair["prompt"])
    with pytest.raises(ValueError):
        weak_pair(row | {"verified": False})


def test_whitespace_and_comment_dedup():
    assert code_key("print(1)") == code_key("# comment\nprint( 1 )")
    assert shingles("print(1)") == shingles("# comment\nprint( 1 )")


def test_existing_problem_assignments_are_preserved_when_scaling():
    rows = [{"id": i, "problem_id": f"problem-{i // 3}"} for i in range(60)]
    preserved = {"problem-0": "test", "problem-1": "dev", "problem-2": "train"}
    parts = split_rows(rows, preserved=preserved)
    for pid, name in preserved.items():
        assert any(r["problem_id"] == pid for r in parts[name])
        assert all(r["problem_id"] != pid for k, records in parts.items() if k != name for r in records)
