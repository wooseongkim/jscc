from __future__ import annotations

from speech_jscc.diagnostics.o5_protocol_audit import (
    compare_protocol_values,
    protocol_rows,
    scientific_comparability,
)


def test_unknown_historical_values_are_not_guessed() -> None:
    comparison = compare_protocol_values(None, "known", historical_available=False)
    assert comparison["classification"] == "unknown"
    assert comparison["original_o5"] is None


def test_hash_comparison_detects_intentional_difference() -> None:
    comparison = compare_protocol_values("abc", "def")
    assert comparison["classification"] == "different"


def test_batch_seed_difference_is_not_directly_comparable() -> None:
    rows = protocol_rows({"seed": 23, "train": {"learning_rate": 1e-3}})
    batch = next(row for row in rows if row["field"] == "fixed_batch_seed")
    assert batch["original_o5"] == 23003
    assert batch["new_c1"] == 23023
    assert batch["classification"] == "different"
    assert scientific_comparability(rows) == "different realization, not directly comparable"
