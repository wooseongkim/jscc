from __future__ import annotations

import argparse
import json
from pathlib import Path

from speech_jscc.diagnostics.stage1_readiness import classify_distribution_evidence, evaluate_readiness


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--root", default="runs/stage1_random_distribution")
    parser.add_argument("--fixed-path-passed", action="store_true"); parser.add_argument("--tests-passed", action="store_true")
    args = parser.parse_args(); root = Path(args.root)
    expected = {}
    o6 = root / "o6_random_clean" / "summary.json"
    if o6.exists():
        provenance = json.loads(o6.read_text()).get("provenance", {})
        expected = {key: provenance.get(key) for key in ("manifest_hashes", "latent_cache_hash", "validation_suite_hash")}
    result = evaluate_readiness(root, expected=expected, fixed_path_passed=args.fixed_path_passed, tests_passed=args.tests_passed)
    def load_summary(name):
        path = root / name / "summary.json"
        return json.loads(path.read_text()) if path.exists() else None
    content_root = Path("runs/stage1_content_generalization")
    g3_candidates = sorted(content_root.glob("g3_random_clean/subset_*/summary.json"))
    g3 = json.loads(g3_candidates[-1].read_text()) if g3_candidates else None
    result["distribution_evidence"] = classify_distribution_evidence(
        load_summary("o6_random_clean"), load_summary("j1_weak_barrage"), g3_summary=g3
    )
    if not result["distribution_evidence"]["curriculum_resume_allowed"]:
        result["ready"] = False; result["uniform_training_command"] = None
        result["reasons"].append("content-generalization G3 and valid O6/J curriculum evidence are required")
    root.mkdir(parents=True, exist_ok=True)
    (root / "uniform_stage1_readiness.json").write_text(json.dumps(result, indent=2))
    lines = ["# Uniform Stage-1 Readiness", "", f"Ready: **{result['ready']}**", "", "## Blocking reasons", ""]
    lines += [f"- {reason}" for reason in result["reasons"]] or ["- None"]
    if result["uniform_training_command"]: lines += ["", "## Prepared command", "", f"`{result['uniform_training_command']}`"]
    (root / "uniform_stage1_readiness.md").write_text("\n".join(lines) + "\n")
    report = ["# Stage-1 Distribution Diagnostic Progress", "", "- Corrected O5: fixed C1 is learnable; its 500→1000 continuation shows optimization-duration limitation.",
              "- Historical O5 and C1 are different realizations and are not directly comparable.",
              f"- O6 evidence: {result['distribution_evidence']['o6_random_clean']}.",
              f"- J1 evidence: {result['distribution_evidence']['j1_weak_barrage']} (never a J2 parent)."]
    for stage in ("o6_random_clean", "j1_weak_barrage", "j2_moderate_barrage", "j3_strong_barrage", "j4_mixed_sparse", "j5_full_mixture"):
        item = result["stages"].get(stage); report.append(f"- {stage}: " + ("PASS" if item and item.get("gate", {}).get("passed") else "not completed or failed"))
    report += [f"- Full Uniform readiness: {result['ready']}", "", "## Exact next external command", "",
               "`bash scripts/run_o6_random_clean_external.sh --device cuda`" if "o6_random_clean" not in result["stages"] else "`bash scripts/run_stage1_jammer_curriculum_external.sh --stage j1_weak_barrage --device cuda`"]
    (root / "stage1_distribution_diagnostic_report.md").write_text("\n".join(report) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
