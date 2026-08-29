from __future__ import annotations

from guard_study.runner import parse_emitted_category, summarize_emitted_categories


class _CategoryTokenizer:
    """Decodes the shape Llama Guard actually emits: a verdict, then a policy code."""

    _VOCABULARY = {
        10: "\n",
        11: "S11",
        12: "S2",
        13: "safe",
        14: "GPS12",
        15: "unsafe",
        99: "<|eot_id|>",
    }

    @staticmethod
    def decode(token_ids, skip_special_tokens=False):
        del skip_special_tokens
        return "".join(_CategoryTokenizer._VOCABULARY[token_id] for token_id in token_ids)


def test_policy_code_is_read_from_the_tokens_after_the_verdict():
    assert parse_emitted_category(_CategoryTokenizer(), [10, 15, 10, 11, 99], 1) == "S11"
    assert parse_emitted_category(_CategoryTokenizer(), [10, 15, 10, 12, 99], 1) == "S2"


def test_a_safe_verdict_carries_no_policy_code():
    assert parse_emitted_category(_CategoryTokenizer(), [10, 13, 99], 1) is None


def test_absent_or_empty_completions_yield_no_code():
    assert parse_emitted_category(_CategoryTokenizer(), [10, 15], None) is None
    assert parse_emitted_category(_CategoryTokenizer(), [10, 15], 1) is None


def test_a_code_shaped_substring_of_another_token_is_not_a_code():
    # "GPS12" contains "S12" but is not a policy code.
    assert parse_emitted_category(_CategoryTokenizer(), [10, 15, 14, 99], 1) is None


def test_category_summary_reports_how_far_a_corpus_is_from_a_decomposition():
    rows = [
        {"pair_id": "alpha", "emitted_category": "S9"},
        {"pair_id": "beta", "emitted_category": "S9"},
        {"pair_id": "gamma", "emitted_category": "S2"},
        {"pair_id": "delta", "emitted_category": None},
    ]

    summary = summarize_emitted_categories(rows)

    assert summary["codes"] == ["S2", "S9"]
    assert summary["pairs_per_code"] == {"S2": 1, "S9": 2}
    assert summary["pair_ids"]["S9"] == ["alpha", "beta"]
    # A single-pair code cannot supply a category mean; the summary has to say so
    # rather than letting one pair stand in for a category.
    assert summary["min_pairs_per_code"] == 1
    assert summary["uncoded_rows"] == 1


def test_category_summary_is_empty_when_nothing_was_coded():
    summary = summarize_emitted_categories([{"pair_id": "alpha", "emitted_category": None}])

    assert summary["codes"] == []
    assert summary["min_pairs_per_code"] == 0
    assert summary["uncoded_rows"] == 1
