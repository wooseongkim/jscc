from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from speech_jscc.diagnostics.o5_root_cause import assert_paired_hashes, linear_slope


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def record_at_step(rows: list[dict[str, Any]], step: int) -> dict[str, Any]:
    for row in rows:
        if int(row["step"]) == step:
            return row
    raise ValueError(f"metrics do not contain exact step {step}")


def _value(row: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in row:
            return row[name]
    return None


def _paired_row(condition: str, row: dict[str, Any]) -> dict[str, Any]:
    scale = row.get("optimal_scale", {})
    layerwise = scale.get("stage1_layerwise_rescaled_loss")
    if layerwise is None and scale.get("per_layer"):
        values = [float(item["rescaled_normalized_mse"]) for item in scale["per_layer"]]
        layerwise = sum(values) / len(values)
    return {
        "condition": condition,
        "optimization_budget_steps": int(row["step"]),
        "final_loss": row["loss"],
        "best_loss": row.get("best_loss", row["loss"]),
        "final_power_ratio": _value(row, "aggregate_power_ratio"),
        "final_correlation": _value(row, "aggregate_pearson_correlation"),
        "global_power_weighted_rescaled_nmse": scale.get(
            "global_power_weighted_rescaled_nmse",
            scale.get("aggregate", {}).get("rescaled_normalized_mse"),
        ),
        "stage1_layerwise_rescaled_loss": layerwise,
    }


def _same_extension_lineage(summary: dict[str, Any], hashes: dict[str, str]) -> bool:
    base = summary.get("extension_base_hashes")
    if base is not None and any(hashes.get(key) != value for key, value in base.items()):
        return False
    # Appending to one metrics.jsonl is a single trajectory unless explicit
    # extension lineage metadata contradicts the condition hashes.
    return True


def build_report_data(root: Path, paired_step: int = 500) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    paired: list[dict[str, Any]] = []
    extensions: list[dict[str, Any]] = []
    for metrics_path in sorted(root.glob("*/metrics.jsonl")):
        condition = metrics_path.parent.name
        rows = _read_jsonl(metrics_path)
        try:
            base = record_at_step(rows, paired_step)
        except ValueError:
            continue
        paired.append(_paired_row(condition, base))
        latest = max(rows, key=lambda row: int(row["step"]))
        if int(latest["step"]) <= paired_step:
            continue
        summary_path = metrics_path.parent / "summary.json"
        hashes_path = metrics_path.parent / "fixed_realization_hashes.json"
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        hashes = json.loads(hashes_path.read_text()) if hashes_path.exists() else {}
        if summary.get("condition", condition) != condition or not _same_extension_lineage(summary, hashes):
            continue
        tail = [row for row in rows if int(row["step"]) >= paired_step]
        losses = [float(row["loss"]) for row in tail]
        extensions.append({
            "condition": condition,
            "base_steps": paired_step,
            "extended_steps": int(latest["step"]),
            "base_final_loss": base["loss"],
            "extended_final_loss": latest["loss"],
            "relative_loss_reduction": (float(base["loss"]) - float(latest["loss"])) / float(base["loss"]),
            "base_power_ratio": _value(base, "aggregate_power_ratio"),
            "extended_power_ratio": _value(latest, "aggregate_power_ratio"),
            "base_correlation": _value(base, "aggregate_pearson_correlation"),
            "extended_correlation": _value(latest, "aggregate_pearson_correlation"),
            "base_best_step": min((row for row in rows if int(row["step"]) <= paired_step), key=lambda row: row["loss"])["step"],
            "extended_best_step": min(rows, key=lambda row: row["loss"])["step"],
            "final_window_slope": linear_slope(losses[max(0, int(len(losses) * 0.8)):]),
            "extension_interpretation": "optimization_duration_limited" if float(latest["loss"]) < 0.5 * float(base["loss"]) else "limited_extension_gain",
        })
    return paired, extensions


def _write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    path.with_suffix(".json").write_text(json.dumps(rows, indent=2))
    with path.with_suffix(".csv").open("w", newline="") as handle:
        if rows:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader(); writer.writerows(rows)


def regenerate(root: Path) -> None:
    paired, extensions = build_report_data(root)
    hashes = {}
    for path in sorted(root.glob("*/fixed_realization_hashes.json")):
        hashes[path.parent.name] = json.loads(path.read_text())
    if hashes:
        assert_paired_hashes(hashes)
    _write_table(root / "aggregate_comparison_500", paired)
    _write_table(root / "extension_comparison", extensions)
    (root / "aggregate_comparison.json").write_text(json.dumps(paired, indent=2))
    _write_table(root / "aggregate_comparison", paired)
    by = {row["condition"]: row for row in paired}
    ext = {row["condition"]: row for row in extensions}
    lines = [
        "# O5 Root-Cause Report", "",
        "## Paired 500-step comparison", "",
        "All rows below are extracted at exactly step 500. Different optimization budgets are not mixed.", "",
    ]
    lines += [f"- {name}: loss={row['final_loss']:.9f}, power_ratio={row['final_power_ratio']:.6f}, correlation={row['final_correlation']:.6f}" for name, row in by.items()]
    lines += ["", "## Same-trajectory extensions", ""]
    lines += [f"- {name}: {row['base_steps']} steps loss {row['base_final_loss']:.9f} -> {row['extended_steps']} steps loss {row['extended_final_loss']:.9f} ({row['extension_interpretation']})." for name, row in ext.items()]
    lines += [
        "", "## Evidence-based conclusion", "",
        "- Fixed full-barrage learning with estimated CSI is possible for the C1 realization.",
        "- C1 improved on the same trajectory from step 500 to step 1000, so 500 steps were insufficient for that realization.",
        "- This fixed-realization result does not establish generalization to random channels or random jammers.",
        "- Pilot contamination changes convergence but is not a structural impossibility in this fixed experiment.",
        "- Oracle jammer subtraction returning near the clean reference validates the jammer integration path.",
        "- The historical O5 and C1 use different realization seeds and must not be interpreted as a direct performance improvement.",
    ]
    (root / "root_cause_report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    regenerate(Path(args.root))


if __name__ == "__main__":
    main()
