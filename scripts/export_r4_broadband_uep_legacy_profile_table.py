#!/usr/bin/env python3
"""Export the legacy-final UEP tail table as CSV and a multipage PDF."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


DEFAULT_ROOT = Path(
    "runs/waveform_aware_wireless/r4_broadband_uep_optimization/"
    "legacy_final_stage2_stage3_profiles"
)
INCLUDED_JSR = (None, 0.0, 5.0, 10.0)


def _condition_key(row: dict) -> tuple:
    return (row["utterance_id"], row["realization_index"], row["snr_db"], row["target_jsr_db"])


def _profile_text(profile: dict) -> str:
    repetition = ",".join(str(value) for value in profile["repetition"])
    shares = ",".join(f"{float(value):.3f}".lstrip("0") for value in profile["power_share"])
    return f"r=[{repetition}]; p=[{shares}]"


def _jsr_label(value: float | None) -> str:
    return "no jammer" if value is None else f"{value:g}"


def summarize_rows(per_sample_rows: list[dict]) -> list[dict]:
    """Produce utterance×SNR tail metrics after realization averaging."""
    base = {_condition_key(row): row for row in per_sample_rows if row["profile"] == "U0"}
    records = []
    for profile in sorted({row["profile"] for row in per_sample_rows}, key=lambda value: (value != "U0", value)):
        for jsr in INCLUDED_JSR:
            rows = [row for row in per_sample_rows if row["profile"] == profile and row["target_jsr_db"] == jsr]
            grouped: dict[tuple, list[tuple[float, float]]] = defaultdict(list)
            for row in rows:
                baseline = base[_condition_key(row)]
                grouped[(row["utterance_id"], row["snr_db"])].append(
                    (row["si_sdr_db"] - baseline["si_sdr_db"], row["si_sdr_db"])
                )
            deltas = sorted(sum(value[0] for value in group) / len(group) for group in grouped.values())
            si_sdrs = [sum(value[1] for value in group) / len(group) for group in grouped.values()]
            index = (len(deltas) - 1) * 0.05
            low, high = int(index), min(int(index) + 1, len(deltas) - 1)
            p5 = deltas[low] + (deltas[high] - deltas[low]) * (index - low)
            records.append({
                "profile": profile,
                "target_jsr_db": jsr,
                "mean": sum(deltas) / len(deltas),
                "p5": p5,
                "relative_lt_minus_3": sum(value < -3.0 for value in deltas) / len(deltas),
                "absolute_lt_minus_10": sum(value < -10.0 for value in si_sdrs) / len(si_sdrs),
            })
    return records


def table_rows(rows: list[dict], profiles: dict) -> list[dict]:
    return [{
        "Profile: r; p": _profile_text(profiles[row["profile"]]),
        "JSR (dB)": _jsr_label(row["target_jsr_db"]),
        "Mean Δ (dB)": f"{row['mean']:+.3f}",
        "p5 Δ (dB)": f"{row['p5']:+.3f}",
        "Δ < −3 dB": f"{row['relative_lt_minus_3']:.3f}",
        "SI-SDR < −10 dB fraction": f"{row['absolute_lt_minus_10']:.3f}",
    } for row in rows if row["target_jsr_db"] in INCLUDED_JSR]


def write_pdf(path: Path, rows: list[dict]) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    headers = list(rows[0])
    per_page = 34
    with PdfPages(path) as pdf:
        for start in range(0, len(rows), per_page):
            page = rows[start:start + per_page]
            fig, axis = plt.subplots(figsize=(16.5, 11.7))
            axis.axis("off")
            title = "Legacy-final broadband UEP profiles: paired tail statistics"
            subtitle = "64 utterances; 2 realizations averaged per utterance×SNR; SNR = 5/10/15 dB; JSR −5 dB omitted"
            axis.set_title(f"{title}\n{subtitle}\nPage {start // per_page + 1}", fontsize=13, pad=16)
            table = axis.table(
                cellText=[[row[column] for column in headers] for row in page],
                colLabels=headers,
                colWidths=[0.53, 0.08, 0.10, 0.10, 0.09, 0.13],
                loc="center",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(6.2)
            table.scale(1.0, 1.18)
            for (row, _), cell in table.get_celld().items():
                if row == 0:
                    cell.set_facecolor("#d9eaf7")
                    cell.set_text_props(weight="bold")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    output_dir = args.output_dir or args.input_dir / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    per_sample = [json.loads(line) for line in (args.input_dir / "per_sample_uep_results.jsonl").read_text().splitlines()]
    profiles = json.loads((args.input_dir / "profile_manifest.json").read_text())
    rows = table_rows(summarize_rows(per_sample), profiles)
    csv_path = output_dir / "legacy_profile_tail_table_no_minus5.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_pdf(output_dir / "legacy_profile_tail_table_no_minus5.pdf", rows)


if __name__ == "__main__":
    main()
