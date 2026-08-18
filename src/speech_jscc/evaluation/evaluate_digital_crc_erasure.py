"""Paired proposed-JSCC versus Digital-CRC-erasure evaluation for fixed R4."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import yaml

from channels.multipath import exponential_pdp
from channels.physical_ofdm import active_grid_masks, apply_tti_multipath, demodulate_tti, estimate_comb_dft_ls, insert_physical_pilots, modulate_tti
from channels.r4_uep_allocator import UEPProfile
from channels.temporal_multipath import correlated_tap_trajectory, jakes_slot_correlation
from speech_jscc.config import resolve_device
from speech_jscc.evaluation.digital_crc_erasure import DigitalCRCTransport
from speech_jscc.evaluation.r4_jammer_baseline import build_r4_jammer
from speech_jscc.evaluation.evaluate_r4_jammer_refiner_checkpoints import (
    ROOT, _PHYSICAL_JAMMER_TYPE, _tap_coefficients, build_fixed_condition_plan,
    condition_key, sha256,
)
from speech_jscc.experiment import build_components
from speech_jscc.training.r4_waveform_finetune import R4ForwardCondition, R4WaveformForward, freeze_codec_for_input_gradient
from speech_jscc.training.si_sdr_loss import si_sdr
from train_channel_free_conv_conformer import load_batch


def paired_condition_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["sample_id"], int(row["crop_offset"]), float(row["snr_db"]), row.get("jsr_db"),
        row["jammer_type"], int(row["realization_index"]), int(row["channel_seed"]),
        int(row["noise_seed"]), int(row["jammer_seed"]),
    )


def validate_csi_only_allocation(allocation: dict[str, Any]) -> None:
    if not bool(allocation.get("enabled")) or allocation.get("mode") != "csi_only":
        raise ValueError("Digital baseline requires CSI-only allocation; risk/interference modes are prohibited")
    forbidden = {"risk", "interference", "jammer", "oracle", "jsr", "mask"}
    if any(any(token in str(key).lower() for token in forbidden) for key in allocation):
        raise ValueError("CSI-only allocation configuration may not contain risk/interference/jammer fields")


def resolve_r4_fixed_profile(path: str | Path) -> UEPProfile:
    artifact_path = Path(path)
    if not artifact_path.is_file():
        raise FileNotFoundError(f"required fixed UEP artifact is absent: {artifact_path}")
    payload = json.loads(artifact_path.read_text())
    try:
        candidate = payload["selected"]["x_best"]["candidate"]["candidate"]
        repetition = tuple(int(value) for value in candidate["repetition"])
        power_share = tuple(float(value) for value in candidate["power_share"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("UEP artifact does not contain selected.x_best candidate r/p") from error
    return UEPProfile("digital_inherited_x_best", repetition, power_share=power_share)


def _allocation_slots(allocation, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    indices = allocation.selected_candidate_indices.to(device)
    power = allocation.power_per_source_copy.to(device) if hasattr(allocation, "power_per_source_copy") else allocation.power_per_resource.to(device)
    valid = indices.ge(0) if indices.ndim == 2 else torch.ones_like(indices, dtype=torch.bool)
    slots, powers = [], []
    for copy in range(indices.shape[0]):
        source_mask = valid[copy]
        slots.append(indices[copy, source_mask])
        powers.append(power[copy, source_mask])
    slot_tensor, power_tensor = torch.cat(slots), torch.cat(powers)
    if slot_tensor.numel() != 5760 or int(torch.unique(slot_tensor).numel()) != 5760:
        raise AssertionError("fixed R4 allocation must select each of 5760 data RE exactly once")
    return slot_tensor, power_tensor


def _qpsk_r4_llrs(*, qpsk: torch.Tensor, allocation, condition: R4ForwardCondition, jammer_type: str, jsr_db: float | None, jammer_seed: int, config: dict[str, Any], eps: float = 1e-12) -> tuple[torch.Tensor, dict[str, float]]:
    """Use the canonical R4 OFDM/pilot/fading/LS path for QPSK soft LLRs."""
    from channels.physical_ofdm import NR_LIKE_R4
    profile = NR_LIKE_R4
    device = qpsk.device
    masks = active_grid_masks(profile, device=device)
    slots, powers = _allocation_slots(allocation, device)
    candidates = torch.zeros((qpsk.shape[0], profile.candidate_data_re), dtype=qpsk.dtype, device=device)
    candidates[:, slots] = qpsk * powers.sqrt()[None]
    data_grid = torch.zeros((qpsk.shape[0], profile.active_subcarriers, profile.n_ofdm_symbols), dtype=qpsk.dtype, device=device)
    data_grid[..., masks.candidate_data] = candidates
    jammer = build_r4_jammer(data_grid, masks.candidate_data, jammer_type=jammer_type, jsr_db=jsr_db, seed=jammer_seed, subband_fraction=float(config["jammer"]["subband_fraction"]), burst_fraction=float(config["jammer"]["burst_fraction"]), tone_count=int(config["jammer"]["tone_count"]), epsilon=eps)
    tx_grid, pilots = insert_physical_pilots(data_grid, profile)
    tx_time = modulate_tti(tx_grid, profile)
    coefficients = condition.tap_coefficients.to(device)
    if coefficients.shape[0] == 1 and qpsk.shape[0] > 1:
        coefficients = coefficients.expand(qpsk.shape[0], -1)
    taps = torch.zeros((qpsk.shape[0], max(condition.tap_delay_samples)+1), dtype=coefficients.dtype, device=device)
    delays = torch.tensor(condition.tap_delay_samples, device=device)[None].expand(qpsk.shape[0], -1)
    taps.scatter_(1, delays, coefficients)
    faded = apply_tti_multipath(tx_time, taps, profile)
    jammer_time = torch.zeros_like(tx_time) if jammer_type == "no_jammer" else modulate_tti(jammer.grid, profile)
    faded_jammer = torch.zeros_like(faded) if jammer_type == "no_jammer" else apply_tti_multipath(jammer_time, taps, profile)
    source_power = (qpsk.abs().square() * powers[None]).mean()
    variance = source_power / (10 ** (float(condition.snr_db) / 10))
    gen = torch.Generator(device=device).manual_seed(int(condition.noise_seed))
    noise = torch.complex(torch.randn(faded.shape, generator=gen, device=device), torch.randn(faded.shape, generator=gen, device=device)) * torch.sqrt(variance / 2)
    received = demodulate_tti(faded + faded_jammer + noise, profile)
    estimated = estimate_comb_dft_ls(received, pilots, profile, num_taps=coefficients.shape[-1], tap_delay_samples=condition.tap_delay_samples, ridge_lambda=1e-6)
    y = received[..., masks.candidate_data][:, slots]
    h = estimated[..., masks.candidate_data][:, slots] * powers.sqrt()[None]
    # Positive LLR denotes bit one, matching Sionna's decoder convention.
    matched = y * h.conj()
    scale = (2.0 * math.sqrt(2.0) / float(variance + eps))
    llrs = torch.stack((matched.real, matched.imag), dim=-1).reshape(qpsk.shape[0], -1) * scale
    csi_nmse = ((estimated - torch.fft.fft(taps, n=profile.n_fft)[:, list(profile.active_fft_bins), None]).abs().square().mean() / (torch.fft.fft(taps, n=profile.n_fft).abs().square().mean() + eps))
    return llrs, {"effective_sinr_db": float((h.abs().square().mean() / (variance + eps)).log10() * 10), "csi_nmse": float(csi_nmse), "pilot_evm": float((received[..., masks.pilot] - pilots[..., masks.pilot]).abs().square().mean().sqrt())}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row}) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def _load_model(path: Path, device: torch.device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = copy.deepcopy(payload["config"])
    config["device"] = str(device)
    codec, model = build_components(config, device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval(); freeze_codec_for_input_gradient(codec)
    return payload, config, codec, model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/eval_digital_crc_erasure.yaml")
    parser.add_argument("--checkpoint", default="runs/waveform_aware_wireless/r4_si_sdr_finetune/si_sdr_medium/local_step_003000.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="runs/waveform_aware_wireless/r4_digital_crc_erasure/smoke")
    parser.add_argument("--max-utterances", type=int)
    parser.add_argument("--max-realizations", type=int)
    parser.add_argument("--max-conditions", type=int, help="bounded CPU smoke subset after paired-plan filtering")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-long-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    validate_csi_only_allocation(config["jammer_aware_allocation"])
    profile_artifact = Path(config["fixed_uep_artifact"])
    profile = resolve_r4_fixed_profile(profile_artifact)
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file(): raise FileNotFoundError(checkpoint)
    output = Path(args.output_dir)
    if output.exists() and not args.overwrite: raise FileExistsError(f"refusing existing output directory: {output}")
    if output.exists():
        import shutil; shutil.rmtree(output)
    output.mkdir(parents=True)
    device = resolve_device(args.device)
    plan_path = Path(config["fixed_condition_plan"])
    plan = json.loads(plan_path.read_text())
    if args.max_utterances:
        sample_ids = sorted({row["sample_id"] for row in plan})[:args.max_utterances]
        plan = [row for row in plan if row["sample_id"] in sample_ids]
    if args.max_realizations: plan = [row for row in plan if int(row["realization_index"]) < args.max_realizations]
    if args.max_conditions: plan = plan[:args.max_conditions]
    if not args.dry_run and not args.allow_long_run and len(plan) > int(config["cpu_smoke_max_conditions"]):
        raise RuntimeError("full evaluation requires --allow-long-run")
    (output / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (output / "command.txt").write_text(" ".join(sys.argv)+"\n")
    (output / "checkpoint_manifest.json").write_text(json.dumps({"proposed_jscc": {"path":str(checkpoint), "sha256":sha256(checkpoint)}, "fixed_uep_artifact":{"path":str(profile_artifact),"sha256":sha256(profile_artifact)}, "refiner_mode":"no_refiner"}, indent=2)+"\n")
    if args.dry_run:
        (output / "dry_run.json").write_text(json.dumps({"conditions":len(plan),"profile_repetition":profile.repetition,"profile_power_share":profile.power_share},indent=2)+"\n"); return
    payload, model_config, codec, model = _load_model(checkpoint, device)
    engine = R4WaveformForward(codec, model, uep_profile=profile, jammer_aware_allocation=config["jammer_aware_allocation"], refiner_mode="no_refiner")
    physical_config = yaml.safe_load((ROOT / "configs/ofdm_nr_like_r4.yaml").read_text())
    transport = DigitalCRCTransport(index_bit_width=int(math.ceil(math.log2(int(codec.get_codebook().shape[1])))), crc_block_frames=int(config["digital"]["crc_block_frames"]), ldpc_iterations=int(config["digital"]["ldpc_iterations"]))
    rows: list[dict[str,Any]]=[]
    with torch.no_grad():
      for row in plan:
        waveform = load_batch([Path(row["sample_id"])], model_config, device)
        target = codec.encode_waveform(waveform)
        taps = _tap_coefficients(engine, physical_config, seed=int(row["channel_seed"]), device=device)
        warm = R4ForwardCondition(snr_db=float(row["snr_db"]),tti=0,tap_coefficients=taps,noise_seed=int(row["noise_seed"]))
        warmout=engine.forward(target,channel_condition=warm,training=False,jammer_type=_PHYSICAL_JAMMER_TYPE[row["jammer_type"]],jammer_jsr_db=row["jsr_db"],jammer_seed=int(row["jammer_seed"]))
        condition=R4ForwardCondition(snr_db=float(row["snr_db"]),tti=1,tap_coefficients=taps,noise_seed=int(row["noise_seed"]))
        proposed=engine.forward(target,channel_condition=condition,delayed_csi=warmout.next_delayed_csi,training=False,jammer_type=_PHYSICAL_JAMMER_TYPE[row["jammer_type"]],jammer_jsr_db=row["jsr_db"],jammer_seed=int(row["jammer_seed"]))
        proposed_wave=codec.decode_representation(proposed.raw_reconstruction)
        indices=codec.encode_rvq_indices(waveform)
        dry=transport.transmit(indices,profile.repetition,profile.power_share,paired_seed=int(row["noise_seed"]))
        llrs, extra=_qpsk_r4_llrs(qpsk=dry.qpsk_symbols,allocation=proposed.allocation,condition=condition,jammer_type=_PHYSICAL_JAMMER_TYPE[row["jammer_type"]],jsr_db=row["jsr_db"],jammer_seed=int(row["jammer_seed"]),config=config)
        decoded=transport.transmit(indices,profile.repetition,profile.power_share,paired_seed=int(row["noise_seed"]),llrs=llrs)
        embedding=codec.lookup_rvq_indices(decoded.recovered_indices)
        failed={i for i, value in enumerate(decoded.packet_failures[0].tolist()) if value}
        from speech_jscc.evaluation.digital_crc_erasure import erase_failed_crc_blocks
        digital_wave=codec.decode_representation(erase_failed_crc_blocks(embedding,decoded.layout,failed))
        slots, layer_power = _allocation_slots(proposed.allocation, device)
        layer_re_count = [240 * int(value) for value in profile.repetition]
        layer_energy = [float(240 * int(rep) * power) for rep, power in zip(profile.repetition, profile.per_re_layer_power().tolist(), strict=True)]
        mapping_hash = hashlib.sha256(slots.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        common={**row,"paired_seed":int(row["noise_seed"]),"allocation_mode":"csi_only","refiner_mode":"no_refiner","importance_order":[1,0,2,5,3,4,6,7],"repetition_profile":list(profile.repetition),"uep_power_fractions":list(profile.power_share),"layer_re_count":layer_re_count,"layer_transmit_energy":layer_energy,"mapping_hash":mapping_hash,"total_re_count":5760,"pilot_re_count":360,"data_re_count":5760,"total_transmit_energy":5760.0,"ldpc_implementation":decoded.ldpc_metadata["implementation"],"ldpc_effective_rate":decoded.ldpc_metadata["effective_rate"],"crc_block_count":decoded.layout.packet_count,"crc_failure_count":decoded.crc_failure_count,"erasure_ratio":decoded.erasure_ratio,**extra}
        rows.append({**common,"method":"proposed_jscc","si_sdr_db":float(si_sdr(proposed_wave,waveform).mean())})
        rows.append({**common,"method":"digital_crc_erasure","si_sdr_db":float(si_sdr(digital_wave,waveform).mean())})
    keys={method:{paired_condition_key(r) for r in rows if r["method"]==method} for method in ("proposed_jscc","digital_crc_erasure")}
    if keys["proposed_jscc"] != keys["digital_crc_erasure"]: raise ValueError("paired condition mismatch")
    with (output / "per_utterance_paired_rows.jsonl").open("w") as h:
        for r in rows: h.write(json.dumps(r)+"\n")
    _write_csv(output / "per_utterance_paired_rows.csv",rows)
    grouped: dict[tuple[Any, ...], dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        grouped[(r["jammer_type"], r["jsr_db"], r["snr_db"])][r["method"]].append(r)
    summary=[]
    for (jammer, jsr, snr), methods in grouped.items():
        jscc, digital = methods["proposed_jscc"], methods["digital_crc_erasure"]
        by_key = {paired_condition_key(r): r for r in jscc}
        digital_by_key = {paired_condition_key(r): r for r in digital}
        if set(by_key) != set(digital_by_key):
            raise ValueError("paired condition mismatch while aggregating summary")
        delta = [float(digital_by_key[key]["si_sdr_db"]) - float(by_key[key]["si_sdr_db"]) for key in by_key]
        def stats(values: list[float], prefix: str) -> dict[str, float]:
            ordered = sorted(values)
            return {f"{prefix}_mean_si_sdr_db":sum(values)/len(values), f"{prefix}_median_si_sdr_db":ordered[len(ordered)//2], f"{prefix}_std_si_sdr_db":float(torch.tensor(values).std(unbiased=False)), f"{prefix}_p10_si_sdr_db":ordered[max(0,math.ceil(.1*len(ordered))-1)]}
        summary.append({"jammer":jammer,"jsr_db":jsr,"snr_db":snr,"rows":len(jscc), **stats([float(r["si_sdr_db"]) for r in jscc],"proposed_jscc"), **stats([float(r["si_sdr_db"]) for r in digital],"digital_crc_erasure"), "paired_digital_minus_jscc_mean_si_sdr_db":sum(delta)/len(delta), "digital_crc_failure_ratio":sum(float(r["erasure_ratio"]) for r in digital)/len(digital)})
    _write_csv(output / "condition_summary.csv",summary)
    (output / "evaluation_integrity.json").write_text(json.dumps({"paired_conditions":len(keys["proposed_jscc"]),"rows":len(rows),"refiner_mode":"no_refiner","allocation_mode":"csi_only"},indent=2)+"\n")

if __name__ == "__main__": main()
