from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import torch
import yaml

from speech_jscc.config import resolve_device
from speech_jscc.evaluation.expanded_validation import ManifestEntry, file_sha256
from speech_jscc.evaluation.r4_layer_ablation import (
    distribution, layer_replacement_metrics, normalized_weights,
    pearson_spearman, replace_layers,
)
from speech_jscc.experiment import build_components
from speech_jscc.training.channel_free_revalidation import per_layer_nmse
from speech_jscc.training.r4_waveform_finetune import (
    freeze_codec_for_input_gradient, R4WaveformForward,
)
from src.evaluation.waveform_metrics import waveform_metrics
from train_channel_free_conv_conformer import load_batch
from evaluate_r4_expanded_validation import (
    _cache_codec_inputs, _checkpoint_payload, _conditions, _make_engine,
)


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/eval_r4_layer_ablation.yaml")
    p.add_argument("--output-dir")
    p.add_argument("--dataset-role", choices=("legacy_final", "expanded_selection"), default="legacy_final")
    p.add_argument("--device")
    p.add_argument("--max-utterances", type=int)
    p.add_argument("--max-realizations", type=int)
    p.add_argument("--layers", nargs="*", type=int)
    p.add_argument("--run-pairwise", action="store_true")
    p.add_argument("--allow-long-run", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def _jsonl(path):
    return [ManifestEntry(**json.loads(line)) for line in Path(path).read_text().splitlines() if line.strip()]


def _hash_tensor(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _mean(values):
    return float(sum(values) / len(values)) if values else None


def _write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({k for row in rows for k in row}))
        writer.writeheader(); writer.writerows(rows)


def _summaries(rows, key, bootstrap_seed):
    result = []
    groups = sorted({(int(r["layer"]), str(r["metric"])) for r in rows})
    for layer, metric in groups:
        members = [r for r in rows if int(r["layer"]) == layer and r["metric"] == metric]
        by_utt = {}
        for row in members:
            by_utt.setdefault(row["utterance_id"], []).append(float(row["value"]))
        utterance_values = [_mean(v) for v in by_utt.values()]
        result.append({"layer": layer, "metric": metric, "utterance_level": distribution(utterance_values, bootstrap_seed=bootstrap_seed + layer), "realization_level": distribution([float(r["value"]) for r in members], bootstrap_seed=bootstrap_seed + 100 + layer)})
    return result


def _metric_rows(*, reference, estimate, target, sample_rate, entry, snr, realization, model_label, layer, metric_prefix, effective_sinr, csi_nmse, pilot_evm):
    metrics = waveform_metrics(reference, estimate, sample_rate)
    target_layer = per_layer_nmse(target, target)
    return {
        "utterance_id": entry.utterance_id, "speaker_id": entry.speaker_id, "source_path": entry.source_path,
        "snr_db": float(snr), "realization": int(realization), "model": model_label, "layer": int(layer),
        "metric_prefix": metric_prefix, "si_sdr_db": float(metrics["si_sdr_db"]),
        "waveform_snr_db": float(metrics["waveform_snr_db"]), "stft_l1": float(metrics["stft_l1"]),
        "effective_sinr_db": float(effective_sinr), "csi_nmse": float(csi_nmse), "pilot_evm": float(pilot_evm),
        "per_layer_nmse": float(target_layer[int(layer)]),
    }


def _append_metric_rows(rows, *, reference, estimate, entry, target, snr, realization, model, layer, prefix, sample_rate, base_sdr, base_snr, base_stft, effective_sinr, csi_nmse, pilot_evm, layer_nmse_value=0.0, nmse_reference_model="selected_r4"):
    metrics = waveform_metrics(reference, estimate, sample_rate)
    if prefix == "clean_remove":
        values = {"intrinsic_si_sdr_drop_db": base_sdr - metrics["si_sdr_db"], "intrinsic_waveform_snr_drop_db": base_snr - metrics["waveform_snr_db"], "intrinsic_stft_increase": metrics["stft_l1"] - base_stft}
    elif prefix == "received_remove":
        values = {"received_remove_si_sdr_drop_db": base_sdr - metrics["si_sdr_db"], "received_remove_waveform_snr_drop_db": base_snr - metrics["waveform_snr_db"], "received_remove_stft_increase": metrics["stft_l1"] - base_stft}
    elif prefix == "selected_restore":
        values = {"oracle_si_sdr_restore_gain_db": metrics["si_sdr_db"] - base_sdr, "oracle_waveform_snr_restore_gain_db": metrics["waveform_snr_db"] - base_snr, "oracle_stft_reduction": base_stft - metrics["stft_l1"]}
    else:
        values = {"initial_oracle_si_sdr_restore_gain_db": metrics["si_sdr_db"] - base_sdr, "initial_oracle_waveform_snr_restore_gain_db": metrics["waveform_snr_db"] - base_snr, "initial_oracle_stft_reduction": base_stft - metrics["stft_l1"]}
    for metric, value in values.items():
        rows.append({"utterance_id": entry.utterance_id, "speaker_id": entry.speaker_id, "source_path": entry.source_path, "snr_db": float(snr), "realization": int(realization), "layer": int(layer), "metric": metric, "value": float(value), "per_layer_nmse": float(layer_nmse_value), "nmse_reference_model": nmse_reference_model, "effective_sinr_db": float(effective_sinr), "csi_nmse": float(csi_nmse), "pilot_evm": float(pilot_evm)})


def main():
    cli = args(); spec = yaml.safe_load(Path(cli.config).read_text())
    output = Path(cli.output_dir or spec["output_dir"])
    if output.exists() and any(output.iterdir()) and not cli.overwrite:
        raise SystemExit(f"refusing existing output directory: {output}")
    entries_path = spec["data"]["legacy_manifest"] if cli.dataset_role == "legacy_final" else spec["data"]["selection_manifest"]
    entries = _jsonl(entries_path)
    if cli.max_utterances: entries = entries[:cli.max_utterances]
    seeds = list(spec["evaluation"]["realization_seeds"])
    if cli.max_realizations: seeds = seeds[:cli.max_realizations]
    layers = sorted(set(cli.layers if cli.layers else range(8)))
    dry = {"dataset_role": cli.dataset_role, "utterances": len(entries), "realizations": len(seeds), "snr_db": spec["evaluation"]["snr_db"], "layers": layers, "pairwise": cli.run_pairwise, "output_dir": str(output)}
    if cli.dry_run:
        print(json.dumps(dry, indent=2)); return
    output.mkdir(parents=True, exist_ok=True)
    (output / "resolved_config.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))
    (output / "command.txt").write_text(" ".join(__import__("sys").argv) + "\n")
    (output / "environment.json").write_text(json.dumps({"python": platform.python_version(), "torch": torch.__version__, "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()}, indent=2))
    selected_path = Path(spec["selected_checkpoint"]); initial_path = Path(spec["initial_checkpoint"])
    selected_payload = _checkpoint_payload(selected_path); initial_payload = _checkpoint_payload(initial_path)
    model_config = copy.deepcopy(selected_payload["config"]); model_config["device"] = cli.device or spec["device"]
    device = resolve_device(model_config["device"])
    codec, template = build_components(model_config, device); freeze_codec_for_input_gradient(codec)
    template.load_state_dict(initial_payload["model"], strict=True); template.eval().requires_grad_(False); template._expanded_validation_config = model_config
    selected = copy.deepcopy(template); selected.load_state_dict(selected_payload["model"], strict=True); selected.eval().requires_grad_(False); selected._expanded_validation_config = model_config
    cache = _cache_codec_inputs(codec, entries, model_config, device); sample_rate = int(model_config["codec"]["sample_rate"])
    long_run = len(entries) * len(seeds) > 4
    if long_run and not cli.allow_long_run: raise SystemExit("full layer ablation requires --allow-long-run")
    detailed=[]; baseline=[]; initial_rows=[]; selected_rows=[]; pairwise_rows=[]
    previous_grad_enabled = torch.is_grad_enabled(); torch.set_grad_enabled(False)
    for snr in map(float, spec["evaluation"]["snr_db"]):
        for realization, seed in enumerate(seeds):
            conditions = _conditions(spec, _make_engine(codec, template, spec), snr=snr, seed=seed, count=len(entries))
            initial_engine = _make_engine(codec, template, spec); selected_engine = _make_engine(codec, selected, spec)
            report=None; reports=[]
            for cached, condition in zip(cache, conditions):
                reports.append(report); target=cached["target"].to(device); waveform=cached["waveform"].to(device)
                ir=initial_engine.forward(target, waveform, condition, report, training=False); report=ir.next_delayed_csi
                initial_rows.append((cached, condition, ir))
            for tti, (cached, condition, ir, shared_report) in enumerate(zip(cache, conditions, [x[2] for x in initial_rows[-len(cache):]], reports)):
                target=cached["target"].to(device); waveform=cached["waveform"].to(device)
                sr=selected_engine.forward(target, waveform, condition, shared_report, training=False)
                clean_wave=codec.decode_representation(target)
                for label, result in (("initial_r4", ir), ("selected_r4", sr)):
                    metrics=waveform_metrics(waveform, result.decoded_waveform, sample_rate)
                    baseline.append({"utterance_id":cached["entry"].utterance_id,"model":label,"snr_db":snr,"realization":realization,"si_sdr_db":metrics["si_sdr_db"],"waveform_snr_db":metrics["waveform_snr_db"],"stft_l1":metrics["stft_l1"],"effective_sinr_db":float(10*torch.log10(result.effective_sinr)),"aggregate_layer_nmse":float(per_layer_nmse(result.reconstruction,target).mean()),"per_layer_nmse":[float(v) for v in per_layer_nmse(result.reconstruction,target)],"summed_latent_nmse":float(((result.reconstruction.sum(1)-target.sum(1))**2).mean())})
                selected_base=waveform_metrics(waveform,sr.decoded_waveform,sample_rate); initial_base=waveform_metrics(waveform,ir.decoded_waveform,sample_rate); clean_base=waveform_metrics(waveform,clean_wave,sample_rate)
                for layer in layers:
                    clean_removed=codec.decode_representation(replace_layers(target,[layer],"zero")); received_removed=codec.decode_representation(replace_layers(sr.reconstruction,[layer],"zero")); selected_restored=codec.decode_representation(sr.reconstruction.clone().index_copy(1,torch.tensor([layer],device=device),target[:,layer:layer+1])); initial_restored=codec.decode_representation(ir.reconstruction.clone().index_copy(1,torch.tensor([layer],device=device),target[:,layer:layer+1]))
                    selected_layer_nmse = float(per_layer_nmse(sr.reconstruction, target)[layer]); initial_layer_nmse = float(per_layer_nmse(ir.reconstruction, target)[layer])
                    for est,prefix,base in ((clean_removed,"clean_remove",clean_base),(received_removed,"received_remove",selected_base),(selected_restored,"selected_restore",selected_base),(initial_restored,"initial_restore",initial_base)):
                        is_initial = prefix.startswith("initial")
                        nmse_value = initial_layer_nmse if is_initial else selected_layer_nmse
                        _append_metric_rows(detailed,reference=waveform,estimate=est,entry=cached["entry"],target=target,snr=snr,realization=realization,model="selected_r4" if prefix.startswith("selected") else "initial_r4" if is_initial else "clean_codec",layer=layer,prefix=prefix,sample_rate=sample_rate,base_sdr=base["si_sdr_db"],base_snr=base["waveform_snr_db"],base_stft=base["stft_l1"],effective_sinr=float(10*torch.log10(sr.effective_sinr)),csi_nmse=float(sr.csi_nmse),pilot_evm=float(sr.pilot_evm),layer_nmse_value=nmse_value,nmse_reference_model="initial_r4" if is_initial else "selected_r4")
                if cli.run_pairwise:
                    for left in layers:
                        for right in layers:
                            if right <= left:
                                continue
                            pair = sr.reconstruction.clone()
                            pair = pair.index_copy(1, torch.tensor([left], device=device), target[:, left:left+1])
                            pair = pair.index_copy(1, torch.tensor([right], device=device), target[:, right:right+1])
                            pair_wave = codec.decode_representation(pair)
                            pair_metric = waveform_metrics(waveform, pair_wave, sample_rate)
                            pairwise_rows.append({"utterance_id":cached["entry"].utterance_id,"layer_left":left,"layer_right":right,"snr_db":snr,"realization":realization,"oracle_pair_si_sdr_gain_db":float(pair_metric["si_sdr_db"]-selected_base["si_sdr_db"])})
    torch.set_grad_enabled(previous_grad_enabled)
    (output/"per_sample_layer_ablation.jsonl").write_text("\n".join(json.dumps(r,sort_keys=True) for r in detailed)+"\n")
    means={m:{str(s):{k:_mean([r[k] for r in baseline if r["model"]==m and r["snr_db"]==s]) for k in ("si_sdr_db","waveform_snr_db","stft_l1")} for s in sorted({r["snr_db"] for r in baseline})} for m in ("initial_r4","selected_r4")}
    reference_check = None
    if len(entries) == 64 and len(seeds) == 2 and 5.0 in map(float, spec["evaluation"]["snr_db"]):
        reference_check = {"initial_si_sdr_expected_db": 0.920, "selected_si_sdr_expected_db": 0.922, "tolerance_db": 0.15, "passed": abs(means["initial_r4"]["5.0"]["si_sdr_db"]-0.920) <= 0.15 and abs(means["selected_r4"]["5.0"]["si_sdr_db"]-0.922) <= 0.15}
        if not reference_check["passed"]:
            raise SystemExit("baseline reproduction outside tolerance; ablation aborted")
    (output/"baseline_reproduction.json").write_text(json.dumps({"rows":len(baseline),"means":means,"reference_check":reference_check,"smoke":len(entries)<4},indent=2))
    summaries=_summaries(detailed,"value",int(spec["evaluation"]["bootstrap_seed"]))
    for name, prefix in (("clean_layer_removal_summary.json","intrinsic"),("received_layer_removal_summary.json","received"),("oracle_layer_replacement_summary.json","oracle"),("initial_vs_selected_summary.json","initial")):
        (output/name).write_text(json.dumps({"rows":summaries,"prefix":prefix},indent=2))
    scores=[]
    for layer in layers:
        def mean_metric(name): return _mean([float(r["value"]) for r in detailed if r["layer"]==layer and r["metric"]==name])
        scores.append({"layer":layer,"intrinsic_si_sdr_drop":mean_metric("intrinsic_si_sdr_drop_db"),"oracle_si_sdr_restore_gain":mean_metric("oracle_si_sdr_restore_gain_db"),"received_remove_si_sdr_drop":mean_metric("received_remove_si_sdr_drop_db"),"initial_oracle_gain":mean_metric("initial_oracle_si_sdr_restore_gain_db")})
    intrinsic=[s["intrinsic_si_sdr_drop"] or 0 for s in scores]; wireless=[s["oracle_si_sdr_restore_gain"] or 0 for s in scores]; alpha=float(spec["priority_score"]["alpha"]); combined=[alpha*(wireless[i] or 0)+(1-alpha)*(s["received_remove_si_sdr_drop"] or 0) for i,s in enumerate(scores)]
    ordering={"intrinsic_order":sorted(layers,key=lambda l: scores[layers.index(l)]["intrinsic_si_sdr_drop"] or -1,reverse=True),"wireless_bottleneck_order":sorted(layers,key=lambda l: scores[layers.index(l)]["oracle_si_sdr_restore_gain"] or -1,reverse=True),"combined_order":sorted(layers,key=lambda l: combined[layers.index(l)],reverse=True)}
    (output/"importance_scores.json").write_text(json.dumps({"intrinsic_si_sdr_drop":intrinsic,"oracle_si_sdr_restore_gain":wireless,"combined_priority_score":combined,"alpha":alpha},indent=2)); (output/"importance_ordering.json").write_text(json.dumps(ordering,indent=2)); _write_csv(output/"importance_table.csv",scores)
    if cli.run_pairwise:
        single_gain = {int(row["layer"]): float(row["oracle_si_sdr_restore_gain"]) for row in scores}
        grouped_pairs = {}
        for row in pairwise_rows:
            grouped_pairs.setdefault((int(row["layer_left"]), int(row["layer_right"])), []).append(float(row["oracle_pair_si_sdr_gain_db"]))
        interaction_rows = []
        for (left, right), values in sorted(grouped_pairs.items()):
            pair_gain = sum(values) / len(values)
            interaction_rows.append({"layer_left": left, "layer_right": right, "pair_oracle_gain_db": pair_gain, "interaction_db": pair_gain - single_gain[left] - single_gain[right], "sample_count": len(values)})
        (output/"pairwise_oracle_replacement.json").write_text(json.dumps({"rows":interaction_rows},indent=2))
        _write_csv(output/"pairwise_interaction_matrix.csv",interaction_rows)
    artifact={"schema_version":2,"codec":{"representation_shape":[8,50,1024]},"evaluation":{"checkpoint":"selected_r4","snr_db":5.0,"utterances":len(entries),"realizations":len(seeds),"metric_definition":"official_unaligned_full_crop_si_sdr"},"importance":{"intrinsic_si_sdr_drop":intrinsic,"oracle_si_sdr_restore_gain":wireless,"combined_order":ordering["combined_order"],"intrinsic_order":ordering["intrinsic_order"],"wireless_bottleneck_order":ordering["wireless_bottleneck_order"],"layer_weights_sum_one":normalized_weights(wireless),"layer_weights_mean_one":normalized_weights(wireless,"mean_one")},"metadata":{"dataset_role":cli.dataset_role,"checkpoint_sha256":file_sha256(selected_path),"smoke":len(entries)<64}}
    (output/"layer_importance_r4_5db.yaml").write_text(yaml.safe_dump(artifact,sort_keys=False))
    split_report={"status":"not_run","reason":"selection split requires separate --dataset-role run"}
    legacy_candidates = [
        output.parent / "legacy_final_corrected" / "importance_ordering.json",
        output.parent / "legacy_final" / "importance_ordering.json",
        output / "../legacy_final/importance_ordering.json",
    ]
    legacy_order_path = next((path for path in legacy_candidates if path.exists()), legacy_candidates[0])
    if cli.dataset_role == "expanded_selection" and legacy_order_path.exists():
        legacy=json.loads(legacy_order_path.read_text())
        current=ordering["wireless_bottleneck_order"]; old=legacy["wireless_bottleneck_order"]
        rank_current={v:i for i,v in enumerate(current)}; rank_old={v:i for i,v in enumerate(old)}
        xs=[rank_old[i] for i in range(8)]; ys=[rank_current[i] for i in range(8)]
        concord=sum((xs[i]-xs[j])*(ys[i]-ys[j]) > 0 for i in range(8) for j in range(i+1,8)); discord=sum((xs[i]-xs[j])*(ys[i]-ys[j]) < 0 for i in range(8) for j in range(i+1,8))
        split_report={"status":"computed","legacy_wireless_order":old,"selection_wireless_order":current,"spearman":float(pearson_spearman(xs,ys)["spearman"] or 0.0),"kendall_tau":float((concord-discord)/28.0),"top1_agreement":int(old[0]==current[0]),"top3_overlap":len(set(old[:3])&set(current[:3]))}
    (output/"split_comparison.json").write_text(json.dumps(split_report,indent=2))
    correlation_rows=[]
    for layer in layers:
        for metric in ("intrinsic_si_sdr_drop_db", "oracle_si_sdr_restore_gain_db", "received_remove_si_sdr_drop_db"):
            members=[r for r in detailed if r["layer"]==layer and r["metric"]==metric]
            correlation_rows.append({"layer":layer,"metric":metric,**pearson_spearman([float(r["per_layer_nmse"]) for r in members],[float(r["value"]) for r in members])})
    _write_csv(output/"layer_metric_correlations.csv",correlation_rows)
    speaker_rows=[]
    for layer in layers:
        for speaker in sorted({r["speaker_id"] for r in detailed}):
            members=[r for r in detailed if r["layer"]==layer and r["speaker_id"]==speaker and r["metric"]=="oracle_si_sdr_restore_gain_db"]
            if members: speaker_rows.append({"speaker_id":speaker,"layer":layer,"oracle_si_sdr_restore_gain_db":_mean([float(r["value"]) for r in members])})
    _write_csv(output/"speaker_level_summary.csv",speaker_rows)
    run_status = "full legacy 64x2" if len(entries) == 64 and len(seeds) == 2 else "smoke/subset"
    (output/"report.md").write_text("# R4 Layer Ablation\n\nOfficial unaligned full-crop SI-SDR was used. Alignment was not applied. Execution status: %s.\n\nPer-layer NMSE in removal/restore rows refers to the corresponding selected or initial reconstruction and is used only as an explanatory predictor; clean latent removal itself has no reconstruction error.\n\n## Ordering\n\n"%run_status+json.dumps(ordering,indent=2)+"\n\nBaseline and detailed artifacts are in the JSONL/JSON files.\n")
    print(json.dumps({"output_dir":str(output),"baseline_rows":len(baseline),"ablation_rows":len(detailed),"ordering":ordering},indent=2))


if __name__ == "__main__": main()
