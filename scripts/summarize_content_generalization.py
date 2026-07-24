from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from speech_jscc.diagnostics.content_generalization import CONTENT_STAGES, SUBSET_SIZES, ladder_decision


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--root", required=True); args = parser.parse_args(); root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    rows = []; first_failure = None; smallest_pass = {}
    for stage in CONTENT_STAGES:
        for subset in SUBSET_SIZES:
            path = root / stage / f"subset_{subset}" / "summary.json"
            if not path.exists(): continue
            summary = json.loads(path.read_text()); passed = bool(summary["gate"]["passed"])
            row = {"stage": stage, "subset_size": subset, "steps": summary["steps"], "passed": passed,
                   "best_step": summary["best_step"], "final_window_loss_slope": summary["final_window_loss_slope"]}
            for group, metrics in summary["validation"].items():
                prefix = group.replace("_utterance_unseen_channel", "").replace("_unseen_utterance", "")
                row[f"{prefix}_loss"] = metrics["aggregate"]["normalized_mse"]
                row[f"{prefix}_power_ratio"] = metrics["aggregate"]["power_ratio"]
                row[f"{prefix}_correlation"] = metrics["aggregate"]["pearson_correlation"]
                row[f"{prefix}_layer0_loss"] = metrics["layer0_summary"]["normalized_mse"]
            rows.append(row)
            decision = ladder_decision(stage, subset, passed)
            if passed and stage not in smallest_pass: smallest_pass[stage] = subset
            if decision == "stop_first_failing_stage" and first_failure is None: first_failure = stage
    (root / "aggregate_results.json").write_text(json.dumps(rows, indent=2))
    if rows:
        fields = sorted({key for row in rows for key in row})
        with (root / "aggregate_results.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    lines = ["# Stage-1 Content-Generalization Report", "", "## Prior evidence", "",
             "- O6: **FAIL**. V1 is only a weak pass; random-channel learning is not considered solved.",
             "- J1: **exploratory_failed_parent**. It is excluded from curriculum readiness and cannot parent J2.", "",
             "## G0–G3 results", ""]
    if not rows: lines.append("- No long external G-stage result has been executed.")
    for row in rows: lines.append(f"- {row['stage']} subset={row['subset_size']}: {'PASS' if row['passed'] else 'FAIL'}")
    lines += ["", f"- First failing content stage: {first_failure or 'not yet determined'}",
              f"- Smallest passing subsets: `{json.dumps(smallest_pass, sort_keys=True)}`", "",
              "## Next command", "", "`bash scripts/run_stage1_content_generalization_external.sh --device cuda`"]
    (root / "content_generalization_report.md").write_text("\n".join(lines) + "\n")
    (root / "experiment_manifest.json").write_text(json.dumps({"o6_status": "FAIL", "j1_status": "exploratory_failed_parent",
        "j1_may_parent_j2": False, "first_failing_stage": first_failure, "smallest_passing_subset": smallest_pass}, indent=2))
    print(json.dumps({"results": len(rows), "first_failing_stage": first_failure, "smallest_passing_subset": smallest_pass}, indent=2))


if __name__ == "__main__": main()
