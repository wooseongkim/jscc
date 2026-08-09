#!/usr/bin/env python3
"""Render the Stage-3 JSR-dependent fixed-UEP comparison figure.

The input is the actual Stage-3 ``per_condition_summary.csv``.  Each JSR
point is the row-weighted mean across the evaluated SNR values; therefore no
legacy-final samples or synthetic scores enter this visualization.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


DEFAULT_ROOT = Path(
    "runs/waveform_aware_wireless/r4_broadband_uep_optimization/"
    "stage3_full_selection"
)
CODEBOOK_PROFILE = "bff35c50c8443a566baa"
HIGH_JSR_PROFILE = "eb31979bb7622b0ee2e2"


def _jsr_value(value: str) -> float | None:
    return None if value.strip() == "" else float(value)


def build_jsr_series(
    summary_path: Path,
    *,
    codebook_profile: str,
    high_jsr_profile: str,
    candidate_profiles: list[str],
) -> dict[str, list[float]]:
    """Return row-weighted JSR means for the three deployable profiles."""
    grouped: dict[tuple[str, float], list[tuple[float, int]]] = defaultdict(list)
    with summary_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            jsr = _jsr_value(row["target_jsr_db"])
            if jsr is None:
                continue
            grouped[(row["profile"], jsr)].append(
                (float(row["mean_delta_si_sdr_vs_u0_db"]), int(row["rows"]))
            )

    def mean(profile: str, jsr: float) -> float:
        values = grouped[(profile, jsr)]
        if not values:
            raise ValueError(f"missing {profile} result at JSR {jsr:g} dB")
        total_rows = sum(count for _, count in values)
        return sum(value * count for value, count in values) / total_rows

    jsr_values = sorted({jsr for _, jsr in grouped})
    # ``candidate_profiles`` is retained in the callable interface so callers
    # can validate the evaluated profile set; it is intentionally not used to
    # construct an oracle curve.
    for profile in ("U0", codebook_profile, high_jsr_profile, *candidate_profiles):
        missing = [jsr for jsr in jsr_values if (profile, jsr) not in grouped]
        if missing:
            raise ValueError(f"profile {profile} is incomplete at JSR {missing}")

    codebook = [mean(codebook_profile, jsr) for jsr in jsr_values]
    high_jsr = [mean(high_jsr_profile, jsr) for jsr in jsr_values]
    return {
        "jsr_db": jsr_values,
        "u0": [mean("U0", jsr) for jsr in jsr_values],
        "codebook_policy": codebook,
        "high_jsr_candidate": high_jsr,
    }


def _write_csv(path: Path, series: dict[str, list[float]]) -> None:
    fields = list(series)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(dict(zip(fields, values)) for values in zip(*(series[field] for field in fields)))


def _render(output_prefix: Path, series: dict[str, list[float]]) -> None:
    import matplotlib.pyplot as plt

    x = series["jsr_db"]
    fig, axis = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
    axis.axhline(0.0, color="0.45", linewidth=1.0, zorder=0)
    axis.plot(x, series["u0"], "o--", color="black", label="U0 (uniform)")
    axis.plot(x, series["codebook_policy"], "o-", color="#0072B2", label="Codebook policy (joint r+p)")
    axis.plot(x, series["high_jsr_candidate"], "s-", color="#D55E00", label="High-JSR candidate (joint r+p)")
    axis.set_xlabel("Broadband jammer-to-signal ratio (JSR, dB)")
    axis.set_ylabel("Mean ΔSI-SDR vs U0 (dB)")
    axis.set_title("JSR-dependent joint repetition–power UEP gain")
    axis.set_xticks(x)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False, loc="upper left")
    for suffix in (".png", ".pdf"):
        fig.savefig(output_prefix.with_suffix(suffix), dpi=300 if suffix == ".png" else None)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir or input_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((input_dir / "profile_manifest.json").read_text())
    candidates = sorted(profile for profile in manifest if profile != "U0")
    series = build_jsr_series(
        input_dir / "per_condition_summary.csv",
        codebook_profile=CODEBOOK_PROFILE,
        high_jsr_profile=HIGH_JSR_PROFILE,
        candidate_profiles=candidates,
    )
    _write_csv(output_dir / "figure3_jsr_dependent_delta_si_sdr.csv", series)
    _render(output_dir / "figure3_jsr_dependent_delta_si_sdr", series)
    (output_dir / "figure3_metadata.json").write_text(json.dumps({
        "input": str(input_dir / "per_condition_summary.csv"),
        "split": "expanded_selection",
        "codebook_policy": CODEBOOK_PROFILE,
        "high_jsr_candidate": HIGH_JSR_PROFILE,
        "repetition_vectors": {
            profile: manifest[profile]["repetition"] for profile in manifest
        },
        "series": series,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
