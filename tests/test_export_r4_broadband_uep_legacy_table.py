import importlib.util
from pathlib import Path


def _module():
    path = Path("scripts/export_r4_broadband_uep_legacy_profile_table.py")
    spec = importlib.util.spec_from_file_location("legacy_table", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_table_rows_exclude_minus_five_and_repeat_profile_definition():
    rows = [
        {"profile": "U0", "target_jsr_db": value, "mean": 0.0,
         "p5": 0.0, "relative_lt_minus_3": 0.0,
         "absolute_lt_minus_10": 0.0}
        for value in (None, -5.0, 0.0, 5.0, 10.0)
    ]
    profiles = {"U0": {"repetition": [3] * 8, "power_share": [0.125] * 8}}
    table = _module().table_rows(rows, profiles)
    assert [row["JSR (dB)"] for row in table] == ["no jammer", "0", "5", "10"]
    assert all("r=[3,3,3,3,3,3,3,3]" in row["Profile: r; p"] for row in table)
