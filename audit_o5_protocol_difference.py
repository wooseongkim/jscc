from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import torch

from evaluation.paired import run_mode_on_paired_batch
from models.resource_allocator import allocate_resources
from speech_jscc.config import load_config, resolve_device
from speech_jscc.diagnostics.o5_protocol_audit import protocol_rows, scientific_comparability
from speech_jscc.diagnostics.o5_root_cause import condition_batch, stable_tensor_hash
from speech_jscc.experiment import build_components
from speech_jscc.models import SpeechJSCC
from train_latent_jscc import RepresentationSource
from train_stage1_fixed_tx import _make_batch


def _hash_step0(config: dict, *, protocol: str, device: torch.device) -> dict[str, str]:
    seed = int(config["seed"])
    torch.manual_seed(seed); random.seed(seed)
    codec, built_model = build_components(config, device); codec.eval()
    target, _ = RepresentationSource(config, codec, device, "train").next_batch(1)
    if protocol == "original_o5":
        torch.manual_seed(seed); random.seed(seed)
        model = SpeechJSCC(tuple(target.shape[1:]), config["model"]["channel_uses"], 8,
                           config["model"]["hidden_dim"], 1.0).to(device)
        batch_seed = 23003
    else:
        model = built_model
        batch_seed = seed + 23000
    batch = _make_batch(codec, model, config, target=target, waveform=None, snr_db=10.0,
                        jsr_db=0.0, jammer_type="barrage", seed=batch_seed, device=device)
    if protocol != "original_o5":
        batch = condition_batch(batch, "full_barrage_estimated_csi", 0.0)
    state = torch.zeros(1, model.encoder.channel_state_dim, device=device)
    gates = torch.ones(1, model.encoder.num_layers, device=device)
    result = run_mode_on_paired_batch(
        codec, model, batch, state, gates, equalizer="estimated", fading="multipath_block",
        channel_estimator="dft_tap_ls", estimator_num_taps=6, allocation_mode="uniform",
        resource_reliability=torch.ones_like(batch.noise.real), receiver_state_mode="observable_v1",
        decode_waveform=False,
    )
    parameters = torch.cat([parameter.detach().flatten().cpu() for parameter in model.parameters()])
    values = {
        "latent_target": target, "initial_model_parameters": parameters,
        "legitimate_channel": batch.signal_fading, "jammer_channel": batch.jammer_fading,
        "awgn": batch.noise, "jammer_waveform": batch.jammer, "jammer_mask": batch.jammer_mask,
        "pilot_mask": batch.pilot_mask, "transmitted_initial_data_symbols": result["data_symbols"],
        "receiver_state": result["decoder_state"], "decoder_input": result["decoder_input"],
    }
    return {key: stable_tensor_hash(value) for key, value in values.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--step0", action="store_true")
    parser.add_argument("--device")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.device: config["device"] = args.device
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    rows = protocol_rows(config)
    hashes: dict[str, dict[str, str] | None] = {"original_o5": None, "new_c1": None}
    if args.step0:
        device = resolve_device(config.get("device", "auto"))
        hashes = {name: _hash_step0(config, protocol=name, device=device)
                  for name in ("original_o5", "new_c1")}
        by = {row["field"]: row for row in rows}
        for key in hashes["original_o5"]:
            field = f"step0_hash.{key}"
            old, new = hashes["original_o5"][key], hashes["new_c1"][key]
            from speech_jscc.diagnostics.o5_protocol_audit import compare_protocol_values
            by[field] = {"field": field, **compare_protocol_values(old, new)}
        rows = list(by.values())
    with (out / "protocol_difference.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    (out / "step0_hash_comparison.json").write_text(json.dumps(hashes, indent=2))
    classification = scientific_comparability(rows)
    report = ["# O5 Protocol Difference Audit", "", f"**Classification:** {classification}", "",
              "The historical O5 used batch seed 23003; C1 used seed 23023. Their 500-step losses are not a direct performance comparison.", "",
              "| Field | Original O5 | New C1 | Classification |", "|---|---|---|---|"]
    report += [f"| {r['field']} | `{r['original_o5']}` | `{r['new_c1']}` | {r['classification']} |" for r in rows]
    (out / "protocol_difference_report.md").write_text("\n".join(report) + "\n")
    (out / "reproduction_commands.md").write_text(
        "# Reproduction commands\n\n```bash\npython audit_o5_protocol_difference.py --config "
        f"{args.config} --output_dir {args.output_dir} --step0\n```\n"
    )


if __name__ == "__main__": main()
