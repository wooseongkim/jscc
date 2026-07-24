from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import torch

from channels.jammer import compute_jsr, make_jammer
from channels.multipath import exponential_pdp
from channels.pilot import (
    csi_nmse,
    equalize_with_csi,
    estimate_channel_ls,
    insert_pilots,
    make_pilot_mask,
    pilot_evm,
)
from channels.rayleigh import compute_effective_sinr, post_channel_jsr, rayleigh_channel
from speech_jscc.config import load_config, resolve_device


def _tensor_mean(value: torch.Tensor) -> float:
    return float(value.detach().real.float().mean().cpu().item())


def _scenario_overrides(name: str) -> dict[str, Any]:
    if name == "clean_high_snr":
        return {"snr_db": 30.0, "jammer_type": "none"}
    if name == "noisy":
        return {"snr_db": 5.0, "jammer_type": "none"}
    if name == "narrowband_jammer":
        return {"snr_db": 10.0, "jsr_db": 0.0, "jammer_type": "narrowband"}
    if name == "pilot_jammer":
        return {"snr_db": 10.0, "jsr_db": 0.0, "jammer_type": "pilot"}
    raise ValueError(f"unsupported scenario: {name}")


def run_diagnostic(config_path: str | Path, *, scenario: str, output_dir: str | Path | None = None) -> dict[str, Any]:
    config = load_config(config_path)
    channel = dict(config["channel"])
    channel.update(_scenario_overrides(scenario))
    device = resolve_device(config.get("device", "auto"))
    batch_size = int(config.get("diagnostics", {}).get("batch_size", 64))
    output = Path(output_dir or config.get("diagnostics", {}).get("output_dir", "runs/channel_diagnostics/multipath_block")) / scenario
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib"))
    generator = torch.Generator(device=device).manual_seed(int(config.get("seed", 0)))
    subcarriers, symbols = [int(value) for value in channel.get("grid_shape", [32, 16])]
    target_power = float(channel.get("target_power", 1.0))
    transmitted = torch.full(
        (batch_size, subcarriers, symbols),
        complex(math.sqrt(target_power), 0.0),
        device=device,
        dtype=torch.complex64,
    )
    pilot_mask = make_pilot_mask(
        tuple(transmitted.shape),
        int(channel.get("pilot_spacing", 4)),
        time_spacing=channel.get("pilot_time_spacing", 4),
        device=device,
    )
    transmitted, pilots = insert_pilots(transmitted, pilot_mask)
    jammer_type = channel.get("jammer_type", "none")
    if jammer_type == "none":
        jammer = torch.zeros_like(transmitted)
        jammer_mask = torch.zeros_like(pilot_mask)
    else:
        jammer, jammer_mask = make_jammer(
            transmitted,
            float(channel.get("jsr_db", 0.0)),
            jammer_type,
            float(channel.get("jammed_fraction", 0.25)),
            pilot_mask=pilot_mask if jammer_type == "pilot" else None,
            pilot_spacing=int(channel.get("pilot_spacing", 4)),
            generator=generator,
        )
    result = rayleigh_channel(
        transmitted,
        jammer,
        float(channel.get("snr_db", 30.0)),
        fading=channel.get("fading", "multipath_block"),
        num_taps=int(channel.get("num_taps", 6)),
        pdp_decay=float(channel.get("pdp_decay", 0.7)),
        generator=generator,
    )
    estimated = estimate_channel_ls(
        result["received"],
        pilots,
        pilot_mask,
        fading=channel.get("fading", "multipath_block"),
        channel_estimator=channel.get("channel_estimator", "block_frequency_ls"),
        estimator_num_taps=channel.get("estimator_num_taps"),
        estimator_ridge_lambda=channel.get("estimator_ridge_lambda", 1e-6),
    )
    equalized = equalize_with_csi(result["received"], estimated)
    tap_power = result["signal_taps"].abs().square().mean(dim=0) if "signal_taps" in result else torch.empty(0)
    report = {
        "scenario": scenario,
        "fading_model": result.get("fading_model"),
        "grid_shape": [batch_size, subcarriers, symbols],
        "num_taps": int(channel.get("num_taps", 6)),
        "pdp": [float(value) for value in result.get("pdp", exponential_pdp(1)).detach().cpu().tolist()],
        "empirical_average_tap_powers": [float(value) for value in tap_power.detach().cpu().tolist()],
        "empirical_average_h_power": _tensor_mean(result["signal_fading"].abs().square()),
        "csi_nmse": _tensor_mean(csi_nmse(result["signal_fading"], estimated)),
        "pilot_evm": _tensor_mean(pilot_evm(result["received"], pilots, pilot_mask, estimated)),
        "channel_estimator": channel.get("channel_estimator", "block_frequency_ls"),
        "estimator_num_taps": channel.get("estimator_num_taps"),
        "estimator_ridge_lambda": channel.get("estimator_ridge_lambda", 1e-6),
        "requested_snr_db": float(channel.get("snr_db", 30.0)),
        "requested_jsr_db": None if jammer_type == "none" else float(channel.get("jsr_db", 0.0)),
        "measured_pre_channel_jsr_db": None
        if jammer_type == "none"
        else _tensor_mean(compute_jsr(transmitted, jammer, db=True)),
        "post_channel_jsr_db": None
        if jammer_type == "none"
        else _tensor_mean(post_channel_jsr(result["faded_signal"], result["faded_jammer"], db=True)),
        "post_equalization_effective_sinr_db": _tensor_mean(
            compute_effective_sinr(
                equalize_with_csi(result["faded_signal"], estimated),
                equalize_with_csi(result["faded_jammer"], estimated),
                equalize_with_csi(result["noise"], estimated),
                db=True,
            )
        ),
        "time_constancy_max_error": float(
            (result["signal_fading"][:, :, 0] - result["signal_fading"][:, :, -1])
            .abs()
            .max()
            .detach()
            .cpu()
            .item()
        ),
        "signal_jammer_channel_independence_mean_abs_diff": None
        if "jammer_taps" not in result
        else float(
            (result["signal_taps"] - result["jammer_taps"]).abs().mean().detach().cpu().item()
        ),
        "jammer_mask_ratio": float(jammer_mask.float().mean().detach().cpu().item()),
        "assume_ideal_cp": bool(result.get("assume_ideal_cp", False)),
        "block_fading_over_time": bool(result.get("block_fading_over_time", False)),
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose the multipath OFDM block-fading channel")
    parser.add_argument("--config", default="configs/eval_multipath_channel.yaml")
    parser.add_argument(
        "--scenario",
        choices=["clean_high_snr", "noisy", "narrowband_jammer", "pilot_jammer"],
        default="clean_high_snr",
    )
    parser.add_argument("--output_dir")
    args = parser.parse_args()
    run_diagnostic(args.config, scenario=args.scenario, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
