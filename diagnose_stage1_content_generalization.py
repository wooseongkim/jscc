from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
import yaml

from speech_jscc.config import load_config, resolve_device
from speech_jscc.diagnostics.content_generalization import (
    CONTENT_ENGINE_VERSION, CONTENT_STAGE_VERSION, CONTENT_STAGES, SUBSET_SIZES,
    aggregate_dataset_statistics, build_content_subsets, build_content_validation_suite,
    content_group_gate, content_realization_seed, forward_content_path, parse_speaker_id,
    summarize_content_metrics,
)
from speech_jscc.diagnostics.o5_root_cause import linear_slope, stable_tensor_hash
from speech_jscc.diagnostics.g0_exposure import EpochSampler, compute_train_baselines, evaluate_baselines, exposure_metric_summary, steps_for_epochs, gradient_norms
from speech_jscc.diagnostics.conv_conformer_integration import INTEGRATION_VERSION, J1_STAGE, J2_STAGE, J3_STAGE, J4_STAGE, build_j1_validation_suite, build_j2_validation_suite, build_j3_validation_suite, build_j4_validation_suite, forward_integration_path, j1_realization_policy, j1_stage_gate, j2_realization_policy, jammer_power_diagnostics, realization_policy, stage_gate, validate_stage_metadata
from speech_jscc.diagnostics.j2_barrage import J2_THRESHOLDS, classify_j2, file_sha256, j2_gate, summarize_layer_groups, validate_j2_resume_metadata, verify_j1_artifact
from speech_jscc.diagnostics.j3_narrowband import J3_THRESHOLDS, aggregate_channel_diagnostics, classify_j3, j3_gate, j3_initialization_metadata, j3_policy, narrowband_diagnostics, sinr_fields, validate_j3_resume_metadata, verify_j2_artifact, write_training_curves
from speech_jscc.diagnostics.j4_burst import J4_THRESHOLDS, burst_diagnostics, classify_j4, j4_gate, j4_policy, tail_statistics, validate_j4_resume_metadata, verify_j3_accepted
from speech_jscc.models.architecture_checkpoint import architecture_metadata
from speech_jscc.diagnostics.random_distribution import SEED_DERIVATION_VERSION, SeedDeriver
from speech_jscc.experiment import build_components
from train_latent_jscc import RepresentationSource, layer_weighted_latent_mse
from train_stage1_fixed_tx import _make_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="External Stage-1 content-generalization diagnostic")
    parser.add_argument("--config", required=True); parser.add_argument("--stage", required=True, choices=(*CONTENT_STAGES,J1_STAGE,J2_STAGE,J3_STAGE,J4_STAGE))
    parser.add_argument("--subset-size", required=True, choices=SUBSET_SIZES); parser.add_argument("--steps", type=int)
    parser.add_argument("--max-epochs",type=int); parser.add_argument("--batch-size",type=int,default=4); parser.add_argument("--num-workers",type=int,default=0)
    parser.add_argument("--seed", type=int, default=23); parser.add_argument("--output-dir", required=True); parser.add_argument("--device")
    parser.add_argument("--checkpoint-every", type=int, default=250); parser.add_argument("--validation-every", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=25); parser.add_argument("--resume"); parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true"); parser.add_argument("--allow-long-run", action="store_true")
    parser.add_argument("--selected-range"); parser.add_argument("--initialization-mode",choices=("fresh","j1_transfer"),default="fresh")
    parser.add_argument("--parent-checkpoint"); parser.add_argument("--j1-summary")
    parser.add_argument("--selected-distribution"); parser.add_argument("--j2-summary")
    parser.add_argument("--j3-manifest")
    return parser.parse_args()


def _git() -> tuple[str, bool]:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], text=True, capture_output=True, check=False).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], text=True, capture_output=True, check=False).stdout.strip())
    return commit, dirty


def _find_index(source: RepresentationSource, utterance_id: str) -> int:
    if source.dataset is None: raise ValueError("content diagnostics require manifest-backed real latents")
    matches = [index for index, path in enumerate(source.dataset.paths)
               if path.as_posix().endswith(utterance_id) or path.name == Path(utterance_id).name]
    if len(matches) != 1: raise ValueError(f"utterance ID does not uniquely resolve: {utterance_id}")
    return matches[0]


def _example(source: RepresentationSource, utterance_id: str) -> tuple[torch.Tensor, torch.Tensor]:
    latent, waveform = source.dataset[_find_index(source, utterance_id)]
    return latent.unsqueeze(0), waveform.unsqueeze(0)

def _examples(source,ids):
    rows=[_example(source,x) for x in ids]; return torch.cat([x[0] for x in rows]),torch.cat([x[1] for x in rows])


def _source_for_id(sources: dict[str, RepresentationSource], utterance_id: str) -> RepresentationSource:
    for source in sources.values():
        try: _find_index(source, utterance_id); return source
        except ValueError: pass
    raise ValueError(f"utterance not found in train/validation sources: {utterance_id}")


def _paired(codec, model, config, target, waveform, stage, seed, device, snr_db=10.0,jsr_db=0.0,jammer_type="none",jammed_fraction=None):
    current=config
    if jammed_fraction is not None: current=json.loads(json.dumps(config));current["channel"]["jammed_fraction"]=float(jammed_fraction)
    return _make_batch(codec, model, current, target=target, waveform=waveform, snr_db=snr_db,
                       jsr_db=jsr_db, jammer_type=jammer_type, seed=seed, device=device)


def _with_target(batch, target, waveform):
    return replace(batch, representation=target, waveform=waveform)


def _evaluate(stage, codec, model, config, sources, suite, device, fixed_batch, baselines):
    model.eval(); grouped: dict[str, dict[str, list[torch.Tensor]]] = {}
    with torch.no_grad():
        for scenario in suite["scenarios"]:
            source = _source_for_id(sources, scenario["utterance_id"]); target, waveform = _example(source, scenario["utterance_id"])
            batch = None
            if stage == "g2_fixed_clean": batch = _with_target(fixed_batch, target, waveform)
            elif stage in {"g3_random_clean",J1_STAGE,J2_STAGE,J3_STAGE,J4_STAGE}: batch = _paired(codec, model, config, target, waveform, stage, scenario["channel_seed"], device,scenario.get("snr_db",10.),scenario.get("jsr_db",0.),scenario.get("jammer_type","none"),scenario.get("jammed_fraction"))
            result = forward_content_path("g3_random_clean" if stage in {J1_STAGE,J2_STAGE,J3_STAGE,J4_STAGE} else stage, codec, model, target, config, batch=batch)
            item = grouped.setdefault(scenario["group"], {"reconstruction": [], "target": [],"channel_metrics":[],"sample_metrics":[]})
            item["reconstruction"].append(result["reconstruction"].detach()); item["target"].append(target)
            if stage==J4_STAGE:item["sample_metrics"].append(exposure_metric_summary(result["reconstruction"].detach(),target,group=scenario["group"]))
            if batch is not None:
                error=(result["decoder_input"]-result["data_symbols"]).abs().square().mean(); channel={"csi_nmse":float(result["csi_nmse"].mean()),"pilot_evm":float(result["pilot_evm"].mean()),"data_evm":float(error.sqrt()),"equalized_symbol_mse":float(error),"post_equalization_sinr_db":float(result["post_equalization_sinr"].mean()),"maximum_equalizer_gain":float(result["estimated_channel"].abs().clamp_min(1e-8).reciprocal().max())}
                if stage in {J1_STAGE,J2_STAGE}: channel.update(jammer_power_diagnostics(batch,result["transmitted"]))
                if stage==J3_STAGE: channel.update(narrowband_diagnostics(result["transmitted"],batch.jammer,batch.jammer_mask,batch.pilot_mask,requested_fraction=scenario["jammed_fraction"],requested_global_jsr_db=scenario["jsr_db"],faded_signal=result["faded_signal"],faded_jammer=result["faded_jammer"]));channel.update(sinr_fields(result["post_equalization_sinr"]))
                if stage==J4_STAGE: channel.update(burst_diagnostics(result["transmitted"],batch.jammer,batch.jammer_mask,batch.pilot_mask,requested_fraction=scenario["jammed_fraction"],requested_global_jsr_db=scenario["jsr_db"],faded_signal=result["faded_signal"],faded_jammer=result["faded_jammer"]));channel.update(sinr_fields(result["post_equalization_sinr"]))
                item["channel_metrics"].append(channel)
    model.train(); output = {}
    for group, values in grouped.items():
        metrics = exposure_metric_summary(torch.cat(values["reconstruction"]), torch.cat(values["target"]), group=group)
        ids=[x["utterance_id"] for x in suite["scenarios"] if x["group"]==group]; baseline=evaluate_baselines(torch.cat(values["target"]),ids,baselines,group=group)
        metrics["baselines"]={name:(value["aggregate"]["normalized_mse"] if value.get("aggregate") else None) for name,value in baseline.items()}
        metrics["baseline_per_layer"]={name:value.get("per_layer") for name,value in baseline.items() if value.get("per_layer")}
        metrics["gate"] = content_group_gate(metrics); output[group] = metrics
        if values["channel_metrics"]:
            metrics["channel_metrics"]=aggregate_channel_diagnostics(values["channel_metrics"])
        if stage==J4_STAGE and values["sample_metrics"]:
            groups=[summarize_layer_groups(row) for row in values["sample_metrics"]]
            named={"aggregate_improvement":[row["aggregate"]["relative_improvement_over_zero"] for row in groups],"layers1_to_7_improvement":[row["layers1_to_7"]["relative_improvement_over_zero"] for row in groups],"layers6_to_7_improvement":[row["layers6_to_7"]["relative_improvement_over_zero"] for row in groups],"layer7_improvement":[row["layer7"]["relative_improvement_over_zero"] for row in groups],"layer7_correlation":[row["layer7"]["pearson_correlation"] for row in groups],"layer7_power_ratio":[row["layer7"]["power_ratio"] for row in groups]}
            metrics["tail_statistics"]={key:tail_statistics(value) for key,value in named.items()}
    if stage=="g3_random_clean":
        unseen=[x for x in suite["scenarios"] if x["group"]=="unseen_speaker_unseen_utterance_unseen_channel"]
        for snr in (5.0,10.0,15.0):
            recon=[]; targets=[]; ids=[]
            for scenario in unseen:
                source=_source_for_id(sources,scenario["utterance_id"]); target,waveform=_example(source,scenario["utterance_id"]); seed=scenario["channel_seed"]+int(snr)*1000
                batch=_paired(codec,model,config,target,waveform,stage,seed,device,snr); result=forward_content_path(stage,codec,model,target,config,batch=batch)
                recon.append(result["reconstruction"].detach());targets.append(target);ids.append(scenario["utterance_id"])
            group=f"unseen_speaker_snr_{int(snr)}db"; target=torch.cat(targets); metrics=exposure_metric_summary(torch.cat(recon),target,group=group); base=evaluate_baselines(target,ids,baselines,group=group)
            metrics["baselines"]={name:(value["aggregate"]["normalized_mse"] if value.get("aggregate") else None) for name,value in base.items()}; metrics["baseline_per_layer"]={name:value.get("per_layer") for name,value in base.items() if value.get("per_layer")}; metrics["gate"]=content_group_gate(metrics); output[group]=metrics
    return output


def _statistics(source, ids, preprocessing):
    examples = []
    for utterance_id in ids:
        latent, waveform = _example(source, utterance_id)
        examples.append({"latent": latent, "duration_seconds": waveform.shape[-1] / preprocessing["sample_rate"],
                         "speaker_id": parse_speaker_id(utterance_id)})
    return aggregate_dataset_statistics(examples, preprocessing) if examples else {"unavailable": True}


def _g3_clean_comparison(subset_size: str, validation: dict[str, Any]) -> dict[str, Any]:
    path = Path("runs/stage1_conv_conformer_integration/g3_random_clean") / f"subset_{subset_size}" / "summary.json"
    if not path.exists():
        return {"summary_path": str(path), "status": "unavailable"}
    clean = json.loads(path.read_text()).get("validation", {})
    matches = {
        "seen_utterance_unseen_channel": "seen_utterance_unseen_channel",
        "same_speaker_unseen_utterance_unseen_channel": "same_speaker_unseen_utterance_unseen_channel",
        "unseen_speaker_unseen_utterance_unseen_channel": "unseen_speaker_unseen_utterance_unseen_channel",
        "j1_unseen_snr_5db": "unseen_speaker_snr_5db",
        "j1_unseen_snr_10db": "unseen_speaker_snr_10db",
        "j1_unseen_snr_15db": "unseen_speaker_snr_15db",
    }
    rows = {}
    for jammed_group, clean_group in matches.items():
        if jammed_group not in validation or clean_group not in clean:
            continue
        jammed_loss = float(validation[jammed_group]["aggregate"]["normalized_mse"])
        clean_loss = float(clean[clean_group]["aggregate"]["normalized_mse"])
        rows[jammed_group] = {"matching_clean_group": clean_group, "clean_loss": clean_loss,
            "jammed_loss": jammed_loss, "absolute_loss_degradation": jammed_loss-clean_loss,
            "relative_loss_degradation": (jammed_loss-clean_loss)/max(clean_loss,1e-12)}
    return {"summary_path": str(path), "status": "matched_by_subset_and_validation_group", "groups": rows}


def main() -> None:
    args = parse_args()
    if args.steps is None and args.max_epochs is None: raise SystemExit("--steps or --max-epochs is required")
    if args.steps is not None and args.steps < 0: raise SystemExit("steps must be nonnegative")
    requested_steps=args.steps or 0
    if (requested_steps > 5 or (args.max_epochs or 0)>2) and not args.allow_long_run: raise SystemExit("long runs require --allow-long-run")
    if args.dry_run:
        print(json.dumps({"dry_run": True, "stage": args.stage, "subset_size": args.subset_size,
                          "steps": args.steps, "max_epochs":args.max_epochs,"batch_size":args.batch_size,"output_dir": args.output_dir,
                          "selected_range":args.selected_range,"initialization_mode":args.initialization_mode,"command": " ".join(sys.argv)}, indent=2)); return
    out = Path(args.output_dir)
    if out.exists() and not args.overwrite and not args.resume: raise SystemExit(f"refusing existing output directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config); config["seed"] = args.seed
    if args.device: config["device"] = args.device
    device = resolve_device(config.get("device", "auto")); torch.manual_seed(args.seed); random.seed(args.seed)
    codec, model = build_components(config, device); codec.eval()
    if getattr(model,"architecture",None)!="conv_conformer_v1": raise SystemExit("Conv-Conformer integration requires model.architecture=conv_conformer_v1")
    for parameter in codec.parameters(): parameter.requires_grad_(False)
    sources = {"train": RepresentationSource(config, codec, device, "train"), "val": RepresentationSource(config, codec, device, "val")}
    data = config["data"]
    split_manifest = build_content_subsets(Path(data["train_manifest"]), Path(data["valid_manifest"]),
                                           Path(data["latent_cache_dir"]), seed=args.seed)
    subset = split_manifest["subsets"][args.subset_size]; suite = build_content_validation_suite(subset, args.seed)
    selected_range=None;j1_accepted=None;selected_distribution=None;j2_accepted=None;j3_accepted=None
    if args.stage==J1_STAGE: suite=build_j1_validation_suite(suite,args.seed)
    if args.stage==J2_STAGE:
        if not args.selected_range or not args.j1_summary: raise SystemExit("J2 requires --selected-range and --j1-summary")
        selected_range=json.loads(Path(args.selected_range).read_text())
        if not selected_range.get("defined"): raise SystemExit("J2 selected range is not defined")
        snr_range=selected_range["selected_snr_range_db"];jsr_range=selected_range["selected_jsr_range_db"]
        suite=build_j2_validation_suite(suite,args.seed,snr_range,jsr_range)
        checkpoint_path=args.parent_checkpoint or str(Path(args.j1_summary).parent/"diagnostic_last.pt")
        j1_accepted=verify_j1_artifact(args.j1_summary,checkpoint_path)
        if args.initialization_mode=="j1_transfer" and not args.parent_checkpoint: raise SystemExit("j1_transfer requires --parent-checkpoint")
    if args.stage==J3_STAGE:
        if not args.selected_distribution or not args.j2_summary or not args.parent_checkpoint:raise SystemExit("J3 requires selected distribution and accepted J2 parent")
        selected_distribution=json.loads(Path(args.selected_distribution).read_text())
        if not selected_distribution.get("defined"):raise SystemExit("J3 distribution is not defined")
        j2_accepted=verify_j2_artifact(args.j2_summary,args.parent_checkpoint);suite=build_j3_validation_suite(suite,args.seed,selected_distribution)
    if args.stage==J4_STAGE:
        if not args.selected_distribution or not args.j3_manifest or not args.parent_checkpoint:raise SystemExit("J4 requires selected distribution and accepted J3 parent")
        selected_distribution=json.loads(Path(args.selected_distribution).read_text())
        if not selected_distribution.get("defined"):raise SystemExit("J4 distribution is not defined")
        j3_accepted=verify_j3_accepted(args.j3_manifest,args.parent_checkpoint);suite=build_j4_validation_suite(suite,args.seed,selected_distribution)
    if args.max_epochs is not None: args.steps=steps_for_epochs(args.max_epochs,len(subset["train_ids"]),args.batch_size)
    derive = SeedDeriver(args.seed); weights = torch.ones(model.encoder.num_layers, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["train"]["learning_rate"]))
    if args.stage==J2_STAGE and args.initialization_mode=="j1_transfer":
        parent=torch.load(args.parent_checkpoint,map_location="cpu",weights_only=False)
        if parent.get("stage_metadata",{}).get("diagnostic_stage")!=J1_STAGE: raise SystemExit("J2 transfer parent must be accepted J1")
        model.load_state_dict(parent["model"],strict=True)
    if args.stage==J3_STAGE:
        parent=torch.load(args.parent_checkpoint,map_location="cpu",weights_only=False)
        if parent.get("stage_metadata",{}).get("diagnostic_stage")!=J2_STAGE:raise SystemExit("J3 parent must be accepted J2")
        model.load_state_dict(parent["model"],strict=True)
    if args.stage==J4_STAGE:
        parent=torch.load(args.parent_checkpoint,map_location="cpu",weights_only=False)
        if parent.get("stage_metadata",{}).get("diagnostic_stage")!=J3_STAGE:raise SystemExit("J4 parent must be accepted J3")
        model.load_state_dict(parent["model"],strict=True)
    baselines=compute_train_baselines(((identifier,_example(sources["train"],identifier)[0][0]) for identifier in subset["train_ids"]))
    first_target, first_waveform = _example(sources["train"], subset["train_ids"][0])
    fixed_batch = None
    if args.stage == "g2_fixed_clean":
        fixed_batch = _paired(codec, model, config, first_target, first_waveform, args.stage,
                              content_realization_seed(args.stage, args.seed, 0), device)
    initial_parameters = torch.cat([p.detach().flatten().cpu() for p in model.parameters()])
    commit, dirty = _git(); preprocessing = {"sample_rate": int(config["codec"]["sample_rate"]),
        "waveform_samples": int(config["codec"]["waveform_samples"]), "codec_type": config["codec"]["type"],
        "representation_shape": list(codec.representation_shape)}
    provenance = {"diagnostic_type": "stage1_content_generalization", "diagnostic_engine_version": CONTENT_ENGINE_VERSION,
        "stage_definition_version": ({J2_STAGE:"j2_strong_barrage_boundary_v1",J3_STAGE:"j3_random_narrowband_v1",J4_STAGE:"j4_random_burst_v1"}.get(args.stage,CONTENT_STAGE_VERSION)), "stage_name": args.stage, "subset_size": args.subset_size,
        "seed_derivation_version": SEED_DERIVATION_VERSION, "train_utterance_ids": subset["train_ids"],
        "validation_suite_hash": suite["validation_suite_hash"], "train_manifest_hash": split_manifest["train_manifest_hash"],
        "validation_manifest_hash": split_manifest["validation_manifest_hash"], "latent_cache_hash": split_manifest["latent_cache_hash"],
        "preprocessing": preprocessing, "model_initialization_hash": stable_tensor_hash(initial_parameters),
        "fixed_channel_hash": stable_tensor_hash(fixed_batch.signal_fading) if fixed_batch else None,
        "fixed_noise_hash": stable_tensor_hash(fixed_batch.noise) if fixed_batch else None,
        "diagnostic_stage":args.stage,"diagnostic_stage_version":INTEGRATION_VERSION,"model_architecture":"conv_conformer_v1","architecture_version":model.architecture_version,
        "representation_shape":list(model.encoder.representation_shape),"total_data_channel_uses":1920,"per_layer_channel_uses":[240]*8,"resource_mapping_version":"pilot_reserved_v1","grid_shape":[64,32],"pilot_resource_count":128,"data_resource_count":1920,
        "estimator":"dft_tap_ls" if args.stage!="g1_pilot_reserved_identity" else "none","equalizer":"estimated_zf" if args.stage!="g1_pilot_reserved_identity" else "none","channel_mode":"none" if args.stage=="g1_pilot_reserved_identity" else "multipath_block","fixed_or_random_channel":{"g1_pilot_reserved_identity":"none","g2_fixed_clean":"fixed","g3_random_clean":"random_per_step",J1_STAGE:"random_per_step",J2_STAGE:"random_per_step",J3_STAGE:"random_per_step",J4_STAGE:"random_per_step"}[args.stage],"snr_policy":{"g1_pilot_reserved_identity":"none","g2_fixed_clean":"fixed_10_db","g3_random_clean":"uniform_5_15_db",J1_STAGE:"uniform_5_15_db",J2_STAGE:(selected_range or {}).get("selected_snr_range_db"),J3_STAGE:(selected_distribution or {}).get("selected_snr_range_db"),J4_STAGE:(selected_distribution or {}).get("selected_snr_range_db")}[args.stage],"jammer":({J3_STAGE:"random_narrowband",J4_STAGE:"random_full_band_burst"}.get(args.stage,"random_full_band_barrage" if args.stage in {J1_STAGE,J2_STAGE} else "none")),"jsr_policy":("uniform_-15_-10_db_transmit_reference" if args.stage==J1_STAGE else ((selected_range or {}).get("selected_jsr_range_db") if args.stage==J2_STAGE else ((selected_distribution or {}).get("selected_global_jsr_range_db") if args.stage in {J3_STAGE,J4_STAGE} else "none"))),
        "initialization_seed":args.seed,"initialization_mode":({J2_STAGE:args.initialization_mode,J3_STAGE:"j2_transfer",J4_STAGE:"j3_transfer"}.get(args.stage,"fresh_initialization_control")),
        "parent_checkpoint":str(Path(args.parent_checkpoint).resolve()) if args.parent_checkpoint else None,
        "parent_checkpoint_hash":file_sha256(args.parent_checkpoint) if args.parent_checkpoint else None,
        "accepted_j1":j1_accepted,"accepted_j2":j2_accepted,"accepted_j3":j3_accepted,"selected_training_range":selected_range,"selected_training_distribution":selected_distribution,
        "selected_range_hash":file_sha256(args.selected_range) if args.selected_range else None,
        "selected_distribution_hash":file_sha256(args.selected_distribution) if args.selected_distribution else None,
        "stage_local_steps": args.steps, "cumulative_optimizer_steps": args.steps, "git_commit": commit,
        "working_tree_dirty": dirty}
    if args.stage==J3_STAGE:
        provenance.update(j3_initialization_metadata(Path(args.parent_checkpoint).resolve(),file_sha256(args.parent_checkpoint),Path(args.j2_summary).resolve(),file_sha256(args.j2_summary)))
    start = 0; history = []; channel_hashes = set(); noise_hashes = set(); jammer_channel_hashes=set(); jammer_waveform_hashes=set();mask_hashes=set(); snr_samples=[]; jsr_samples=[];fraction_samples=[]
    if args.resume:
        payload = torch.load(args.resume, map_location="cpu", weights_only=False); saved = payload["provenance"]
        for key in ("stage_name", "subset_size", "validation_suite_hash", "train_manifest_hash", "validation_manifest_hash", "latent_cache_hash", "model_initialization_hash"):
            if saved.get(key) != provenance.get(key): raise SystemExit(f"resume provenance mismatch: {key}")
        validate_stage_metadata(payload.get("stage_metadata"),{k:provenance[k] for k in ("diagnostic_stage","model_architecture","architecture_version","resource_mapping_version","validation_suite_hash")}); model.load_state_dict(payload["model"], strict=True); optimizer.load_state_dict(payload["optimizer"])
        if args.stage==J2_STAGE:
            validate_j2_resume_metadata({"stage":saved["stage_name"],"initialization_mode":saved["initialization_mode"],"selected_range_hash":saved["selected_range_hash"]},
                {"stage":provenance["stage_name"],"initialization_mode":provenance["initialization_mode"],"selected_range_hash":provenance["selected_range_hash"]})
        if args.stage==J3_STAGE:
            validate_j3_resume_metadata(
                {"stage":saved["stage_name"],"selected_distribution_hash":saved["selected_distribution_hash"],"parent_checkpoint_hash":saved["parent_checkpoint_hash"]},
                {"stage":provenance["stage_name"],"selected_distribution_hash":provenance["selected_distribution_hash"],"parent_checkpoint_hash":provenance["parent_checkpoint_hash"]})
        if args.stage==J4_STAGE:
            validate_j4_resume_metadata({"stage":saved["stage_name"],"selected_distribution_hash":saved["selected_distribution_hash"],"parent_checkpoint_hash":saved["parent_checkpoint_hash"],"accepted_manifest_hash":saved["accepted_j3"]["accepted_manifest_sha256"]},{"stage":provenance["stage_name"],"selected_distribution_hash":provenance["selected_distribution_hash"],"parent_checkpoint_hash":provenance["parent_checkpoint_hash"],"accepted_manifest_hash":provenance["accepted_j3"]["accepted_manifest_sha256"]})
        start = int(payload["step"]); history = payload.get("history", [])
        channel_hashes.update(row["channel_hash"] for row in history if row.get("channel_hash")); noise_hashes.update(row["noise_hash"] for row in history if row.get("noise_hash"))
        jammer_channel_hashes.update(row["jammer_channel_hash"] for row in history if row.get("jammer_channel_hash"))
        jammer_waveform_hashes.update(row["jammer_waveform_hash"] for row in history if row.get("jammer_waveform_hash"))
        mask_hashes.update(row["jammer_mask_hash"] for row in history if row.get("jammer_mask_hash"))
        snr_samples.extend(float(row["channel_metrics"]["requested_snr_db"]) for row in history if row.get("channel_metrics"))
        jsr_samples.extend(float(row["jsr_db"]) for row in history if row.get("jsr_db") is not None)
        fraction_samples.extend(float(row["jammed_fraction"]) for row in history if row.get("jammed_fraction") is not None)
        provenance = {**saved, "stage_local_steps": args.steps, "cumulative_optimizer_steps": args.steps,
                      "git_commit": commit, "working_tree_dirty": dirty}
    train_stats = _statistics(sources["train"], subset["train_ids"], preprocessing)
    validation_stats = {
        "seen_utterance_unseen_channel": _statistics(sources["train"], subset["seen_utterance_ids"], preprocessing),
        "same_speaker_unseen_utterance_unseen_channel": _statistics(sources["train"], subset["same_speaker_unseen_ids"], preprocessing),
        "unseen_speaker_unseen_utterance_unseen_channel": _statistics(sources["val"], subset["unseen_speaker_ids"], preprocessing),
    }
    (out / "resolved_config.yaml").write_text(yaml.safe_dump({**config, "content_generalization": provenance}, sort_keys=True))
    (out / "subset_manifest.json").write_text(json.dumps({**split_manifest, "selected_subset": args.subset_size}, indent=2))
    (out / "validation_suite.json").write_text(json.dumps(suite, indent=2)); (out / "dataset_statistics.json").write_text(json.dumps({"train": train_stats, "validation": validation_stats}, indent=2))
    (out / "command.txt").write_text(" ".join(sys.argv) + "\n"); (out / "environment.json").write_text(json.dumps({"python": sys.version, "torch": torch.__version__, "platform": platform.platform(), "git_commit": commit, "working_tree_dirty": dirty}, indent=2))
    metrics_path = out / "metrics.jsonl"; model.train(); sampler=EpochSampler(subset["train_ids"],batch_size=args.batch_size,seed=args.seed,subset_key=f"{args.stage}_{args.subset_size}")
    stream=sampler.batch_stream(args.max_epochs) if args.max_epochs is not None else ((step,[subset["train_ids"][derive.seed("content_train_utterance",step)%len(subset["train_ids"])]] ) for step in range(1,args.steps+1))
    with metrics_path.open("a" if args.resume else "w") as handle:
        for step,utterance_ids in stream:
            if step<=start: continue
            utterance_id = utterance_ids[0]; target,waveform=_examples(sources["train"],utterance_ids); batch = None
            if args.stage == "g2_fixed_clean": batch = _paired(codec,model,config,target,waveform,args.stage,content_realization_seed(args.stage,args.seed,0),device)
            elif args.stage == "g3_random_clean":
                policy=realization_policy("g3_random_clean",args.seed,step); batch = _paired(codec, model, config, target, waveform, args.stage,
                                policy["seed"], device,policy["snr_db"])
            elif args.stage==J1_STAGE:
                policy=j1_realization_policy(args.seed,step); batch=_paired(codec,model,config,target,waveform,args.stage,policy["seed"],device,policy["snr_db"],policy["jsr_db"],"barrage"); snr_samples.append(policy["snr_db"]);jsr_samples.append(policy["jsr_db"]);jammer_channel_hashes.add(stable_tensor_hash(batch.jammer_fading));jammer_waveform_hashes.add(stable_tensor_hash(batch.jammer))
            elif args.stage==J2_STAGE:
                policy=j2_realization_policy(args.seed,step,selected_range["selected_snr_range_db"],selected_range["selected_jsr_range_db"]);batch=_paired(codec,model,config,target,waveform,args.stage,policy["seed"],device,policy["snr_db"],policy["jsr_db"],"barrage");snr_samples.append(policy["snr_db"]);jsr_samples.append(policy["jsr_db"]);jammer_channel_hashes.add(stable_tensor_hash(batch.jammer_fading));jammer_waveform_hashes.add(stable_tensor_hash(batch.jammer))
            elif args.stage==J3_STAGE:
                policy=j3_policy(args.seed,step,selected_distribution["selected_snr_range_db"],selected_distribution["selected_global_jsr_range_db"],selected_distribution["selected_jammed_subcarrier_fractions"]);batch=_paired(codec,model,config,target,waveform,args.stage,policy["seed"],device,policy["snr_db"],policy["jsr_db"],"narrowband",policy["jammed_fraction"]);snr_samples.append(policy["snr_db"]);jsr_samples.append(policy["jsr_db"]);fraction_samples.append(policy["jammed_fraction"]);jammer_channel_hashes.add(stable_tensor_hash(batch.jammer_fading));jammer_waveform_hashes.add(stable_tensor_hash(batch.jammer));mask_hashes.add(stable_tensor_hash(batch.jammer_mask))
            elif args.stage==J4_STAGE:
                policy=j4_policy(args.seed,step,selected_distribution["selected_snr_range_db"],selected_distribution["selected_global_jsr_range_db"],selected_distribution["selected_burst_fractions"]);batch=_paired(codec,model,config,target,waveform,args.stage,policy["seed"],device,policy["snr_db"],policy["jsr_db"],"burst",policy["burst_fraction"]);snr_samples.append(policy["snr_db"]);jsr_samples.append(policy["jsr_db"]);fraction_samples.append(policy["burst_fraction"]);jammer_channel_hashes.add(stable_tensor_hash(batch.jammer_fading));jammer_waveform_hashes.add(stable_tensor_hash(batch.jammer));mask_hashes.add(stable_tensor_hash(batch.jammer_mask))
            if batch is not None:
                channel_hashes.add(stable_tensor_hash(batch.signal_fading)); noise_hashes.add(stable_tensor_hash(batch.noise))
            optimizer.zero_grad(set_to_none=True); mapped_stage={"g1_pilot_reserved_identity":"g1_mapping_train","g2_fixed_clean":"g2_fixed_clean","g3_random_clean":"g3_random_clean",J1_STAGE:"g3_random_clean",J2_STAGE:"g3_random_clean",J3_STAGE:"g3_random_clean",J4_STAGE:"g3_random_clean"}[args.stage]; result = forward_integration_path(mapped_stage, codec, model, target, config, batch=batch)
            loss, per_layer = layer_weighted_latent_mse(result["reconstruction"], target, weights, config["train"]["latent_normalization"])
            loss.backward(); grads=gradient_norms(model); torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["train"].get("gradient_clip_norm", 5.0))); optimizer.step()
            metrics = exposure_metric_summary(result["reconstruction"].detach(), target, group="train")
            record = {"step": step, "utterance_id": utterance_id, "loss": float(loss.detach()),
                      "aggregate": metrics["aggregate"], "per_layer": metrics["per_layer"], "layer0_summary": metrics["layer0_summary"],
                      "channel_hash": stable_tensor_hash(batch.signal_fading) if batch is not None else None,
                      "noise_hash": stable_tensor_hash(batch.noise) if batch is not None else None,"gradient_norms":grads,"sample_presentations":sum(sampler.presentation_counts.values()) if args.max_epochs is not None else step}
            if batch is not None:
                data_error=(result["decoder_input"]-result["data_symbols"]).abs().square().mean()
                record["channel_metrics"]={"requested_snr_db":float(batch.snr_db.mean()),"measured_snr_db":float((10*torch.log10(result["transmitted"].detach().abs().square().mean()/batch.noise.abs().square().mean().clamp_min(1e-12))).detach()),
                    "csi_nmse":float(result["csi_nmse"].detach().mean()),"pilot_evm":float(result["pilot_evm"].detach().mean()),"data_evm":float(data_error.detach().sqrt()),"equalized_symbol_mse":float(data_error.detach()),
                    "post_equalization_sinr_db":float(result["post_equalization_sinr"].detach().mean()),"maximum_equalizer_gain":float(result["estimated_channel"].detach().abs().clamp_min(1e-8).reciprocal().max()),
                    "fifth_percentile_channel_power":float(torch.quantile(batch.signal_fading.abs().square().flatten(),.05))}
                if args.stage in {J1_STAGE,J2_STAGE}:
                    record["jammer_metrics"]=jammer_power_diagnostics(batch,result["transmitted"]); record["jammer_channel_hash"]=stable_tensor_hash(batch.jammer_fading); record["jammer_waveform_hash"]=stable_tensor_hash(batch.jammer); record["jsr_db"]=float(batch.jsr_db.mean())
                if args.stage==J3_STAGE:
                    record["jammer_metrics"]=narrowband_diagnostics(result["transmitted"],batch.jammer,batch.jammer_mask,batch.pilot_mask,requested_fraction=policy["jammed_fraction"],requested_global_jsr_db=policy["jsr_db"],faded_signal=result["faded_signal"],faded_jammer=result["faded_jammer"]);record["channel_metrics"].update(sinr_fields(result["post_equalization_sinr"]));record["jammer_channel_hash"]=stable_tensor_hash(batch.jammer_fading);record["jammer_waveform_hash"]=stable_tensor_hash(batch.jammer);record["jammer_mask_hash"]=stable_tensor_hash(batch.jammer_mask);record["jsr_db"]=float(batch.jsr_db.mean());record["jammed_fraction"]=float(policy["jammed_fraction"])
                if args.stage==J4_STAGE:
                    record["jammer_metrics"]=burst_diagnostics(result["transmitted"],batch.jammer,batch.jammer_mask,batch.pilot_mask,requested_fraction=policy["burst_fraction"],requested_global_jsr_db=policy["jsr_db"],faded_signal=result["faded_signal"],faded_jammer=result["faded_jammer"]);record["channel_metrics"].update(sinr_fields(result["post_equalization_sinr"]));record["jammer_channel_hash"]=stable_tensor_hash(batch.jammer_fading);record["jammer_waveform_hash"]=stable_tensor_hash(batch.jammer);record["jammer_mask_hash"]=stable_tensor_hash(batch.jammer_mask);record["jsr_db"]=float(batch.jsr_db.mean());record["jammed_fraction"]=float(policy["burst_fraction"])
            if step == 1 or step % args.validation_every == 0 or step == args.steps:
                record["validation"] = _evaluate(args.stage, codec, model, config, sources, suite, device, fixed_batch,baselines)
            history.append(record); handle.write(json.dumps(record) + "\n"); handle.flush()
            if step == 1 or step % args.log_every == 0 or step == args.steps:
                print(json.dumps({"step": step, "loss": record["loss"], "validation": {k:v["aggregate"]["normalized_mse"] for k,v in record.get("validation", {}).items()}}, sort_keys=True), flush=True)
            if step % args.checkpoint_every == 0 or step == args.steps:
                torch.save({"diagnostic_type": "conv_conformer_integration", "provenance": provenance,"stage_metadata":{k:provenance[k] for k in ("diagnostic_stage","model_architecture","architecture_version","resource_mapping_version","validation_suite_hash")},
                            "model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step,
                            "history": history}, out / "diagnostic_last.pt")
    final_validation = next(row["validation"] for row in reversed(history) if row.get("validation"))
    gates = {group: metrics["gate"] for group, metrics in final_validation.items()}
    if args.stage == "g3_random_clean" and (len(channel_hashes) < min(args.steps, 2) or len(noise_hashes) < min(args.steps, 2)):
        gates["realization_diversity"] = {"passed": False, "reasons": ["channel_or_noise_did_not_change"]}
    revised=stage_gate(final_validation); gate = {"passed":revised["stage_pass"],**revised,"group_gates":gates}
    if args.stage=="g2_fixed_clean" and (len(channel_hashes)!=1 or len(noise_hashes)!=1):
        gate["passed"]=gate["stage_pass"]=False; gate["fixed_realization_pass"]=False
    elif args.stage=="g2_fixed_clean": gate["fixed_realization_pass"]=True
    if args.stage=="g3_random_clean":
        diversity=len(channel_hashes)>=min(args.steps,2) and len(noise_hashes)>=min(args.steps,2); snr_rows=[v for k,v in final_validation.items() if k.startswith("unseen_speaker_snr_")]; snr_pass=all(v["gate"]["passed"] and v["layers1_to_7_summary"]["relative_improvement_over_zero"]>=.05 for v in snr_rows)
        gate["random_channel_generalization_pass"]=gate["stage_pass"] and diversity and snr_pass; gate["path_learnability_pass"]=gate["stage_pass"]; gate["content_generalization_pass"]=revised["same_speaker_generalization_pass"]; gate["speaker_generalization_pass"]=revised["unseen_speaker_generalization_pass"]; gate["snr_robustness_pass"]=snr_pass; gate["stage_pass"]=gate["stage_pass"] and diversity and snr_pass
    if args.stage==J1_STAGE:
        gate=j1_stage_gate(final_validation,channel_diversity=len(channel_hashes),
            jammer_channel_diversity=len(jammer_channel_hashes),jammer_waveform_diversity=len(jammer_waveform_hashes),
            noise_diversity=len(noise_hashes),required_diversity=min(args.steps,2),
            parameters_finite=all(bool(torch.isfinite(parameter).all()) for parameter in model.parameters()))
        gate["group_gates"]=gates
    j2_classification=None
    if args.stage==J2_STAGE:
        unseen=summarize_layer_groups(final_validation["unseen_speaker_unseen_utterance_unseen_channel"])
        strongest=summarize_layer_groups(final_validation["j2_strongest_selected_condition"])
        strongest["channel"]=final_validation["j2_strongest_selected_condition"].get("channel_metrics",{})
        required=min(args.steps,2);randomness_pass=all(value>=required for value in (len(channel_hashes),len(noise_hashes),len(jammer_channel_hashes),len(jammer_waveform_hashes)))
        coverage_pass=bool(snr_samples and jsr_samples and min(snr_samples)>=selected_range["selected_snr_range_db"][0] and max(snr_samples)<=selected_range["selected_snr_range_db"][1] and min(jsr_samples)>=selected_range["selected_jsr_range_db"][0] and max(jsr_samples)<=selected_range["selected_jsr_range_db"][1])
        parameters_finite=all(bool(torch.isfinite(parameter).all()) for parameter in model.parameters())
        gate=j2_gate(unseen,strongest,randomness_pass=randomness_pass,coverage_pass=coverage_pass,parameters_finite=parameters_finite,thresholds=J2_THRESHOLDS);gate["group_gates"]=gates
        validation_curve=[(row["step"],row["validation"]["unseen_speaker_unseen_utterance_unseen_channel"]["aggregate"]["normalized_mse"]) for row in history if row.get("validation")]
        validation_slope=linear_slope([value for _,value in validation_curve[max(0,int(len(validation_curve)*.8)):]])
        final_is_best=validation_curve[-1][1]<=min(value for _,value in validation_curve)+1e-12
        j2_classification=classify_j2(gate,final_is_best=final_is_best,loss_slope=validation_slope)
        gate["classification"]=j2_classification;gate["final_validation_is_best"]=final_is_best;gate["final_validation_slope"]=validation_slope
    j3_classification=None
    if args.stage==J3_STAGE:
        unseen=summarize_layer_groups(final_validation["unseen_speaker_unseen_utterance_unseen_channel"]);strongest=summarize_layer_groups(final_validation["j3_strongest_selected_condition"]);strongest["channel"]=final_validation["j3_strongest_selected_condition"].get("channel_metrics",{})
        jammer_rows=[row.get("jammer_metrics",{}) for row in history if row.get("jammer_metrics")];configured_fractions={float(x) for x in selected_distribution["selected_jammed_subcarrier_fractions"]};observed_fractions={float(x) for x in fraction_samples};snr_range=selected_distribution["selected_snr_range_db"];jsr_range=selected_distribution["selected_global_jsr_range_db"]
        coverage=bool(snr_samples and jsr_samples and min(snr_samples)>=snr_range[0] and max(snr_samples)<=snr_range[1] and min(jsr_samples)>=jsr_range[0] and max(jsr_samples)<=jsr_range[1] and configured_fractions.issubset(observed_fractions))
        infra={"finite":all(row["aggregate"]["finite"] for row in history),"diversity":min(len(channel_hashes),len(noise_hashes),len(jammer_channel_hashes),len(jammer_waveform_hashes),len(mask_hashes))>=min(args.steps,2),"coverage":coverage,"no_leakage":all(float(row.get("leakage_power_outside_band",1))<=1e-12 for row in jammer_rows),"contiguous":all(row.get("contiguous_band_verified",False) for row in jammer_rows),"metadata":bool(j2_accepted)}
        gate=j3_gate(unseen,strongest,infrastructure=infra,thresholds=J3_THRESHOLDS);curve=[(row["step"],row["validation"]["unseen_speaker_unseen_utterance_unseen_channel"]["aggregate"]["normalized_mse"]) for row in history if row.get("validation")];slope=linear_slope([x[1] for x in curve[max(0,int(len(curve)*.8)):]]);best=curve[-1][1]<=min(x[1] for x in curve)+1e-12;j3_classification=classify_j3(gate,best,slope);gate.update({"classification":j3_classification,"final_validation_is_best":best,"final_validation_slope":slope,"group_gates":gates})
    j4_classification=None
    if args.stage==J4_STAGE:
        unseen=summarize_layer_groups(final_validation["unseen_speaker_unseen_utterance_unseen_channel"]);strongest=summarize_layer_groups(final_validation["j4_strongest_selected_condition"]);strongest["channel"]=final_validation["j4_strongest_selected_condition"].get("channel_metrics",{})
        jammer_rows=[row.get("jammer_metrics",{}) for row in history if row.get("jammer_metrics")];configured={float(x) for x in selected_distribution["selected_burst_fractions"]};observed={float(x) for x in fraction_samples};snr_range=selected_distribution["selected_snr_range_db"];jsr_range=selected_distribution["selected_global_jsr_range_db"]
        coverage=bool(snr_samples and jsr_samples and min(snr_samples)>=snr_range[0] and max(snr_samples)<=snr_range[1] and min(jsr_samples)>=jsr_range[0] and max(jsr_samples)<=jsr_range[1] and configured.issubset(observed));infra={"finite":all(row["aggregate"]["finite"] for row in history),"diversity":min(len(channel_hashes),len(noise_hashes),len(jammer_channel_hashes),len(jammer_waveform_hashes),len(mask_hashes))>=min(args.steps,2),"coverage":coverage,"no_leakage":all(float(row.get("leakage_power_outside_burst",1))<=1e-12 for row in jammer_rows),"contiguous":all(row.get("contiguous_burst_verified",False) for row in jammer_rows),"full_band":all(row.get("full_band_inside_burst_verified",False) for row in jammer_rows),"metadata":bool(j3_accepted)}
        tails=final_validation["unseen_speaker_unseen_utterance_unseen_channel"]["tail_statistics"];tail={"layer7_improvement_p10":tails["layer7_improvement"]["p10"],"layer7_negative_rate":tails["layer7_improvement"]["negative_rate"]}
        gate=j4_gate(unseen,strongest,infrastructure=infra,tail=tail,thresholds=J4_THRESHOLDS);curve=[(row["step"],row["validation"]["unseen_speaker_unseen_utterance_unseen_channel"]["aggregate"]["normalized_mse"]) for row in history if row.get("validation")];slope=linear_slope([x[1] for x in curve[max(0,int(len(curve)*.8)):]]);best=curve[-1][1]<=min(x[1] for x in curve)+1e-12;j4_classification=classify_j4(gate,best,slope);gate.update({"classification":j4_classification,"final_validation_is_best":best,"final_validation_slope":slope,"group_gates":gates})
    losses = [row["loss"] for row in history]; window = losses[max(0, int(len(losses) * .8)):]
    summary = {"provenance": provenance, "steps": args.steps, "best_step": min(history, key=lambda row: row["loss"])["step"],
               "final_train": history[-1], "validation": final_validation, "gate": gate,
               "channel_hash_diversity": len(channel_hashes), "noise_hash_diversity": len(noise_hashes),
               "fraction_finite_batches": sum(row["aggregate"]["finite"] for row in history) / len(history),
               "final_window_loss_slope": linear_slope(window), "dataset_statistics": {"train": train_stats, "validation": validation_stats}}
    if args.stage==J1_STAGE:
        summary.update({"jammer_channel_hash_diversity":len(jammer_channel_hashes),
            "jammer_waveform_hash_diversity":len(jammer_waveform_hashes),
            "snr_sample_range":[min(snr_samples),max(snr_samples)] if snr_samples else None,
            "jsr_sample_range":[min(jsr_samples),max(jsr_samples)] if jsr_samples else None,
            "g3_clean_comparison":_g3_clean_comparison(args.subset_size,final_validation)})
    if args.stage==J2_STAGE:
        unseen_groups=summarize_layer_groups(final_validation["unseen_speaker_unseen_utterance_unseen_channel"])
        summary.update({"classification":j2_classification,"provisional_thresholds":J2_THRESHOLDS,
            "jammer_channel_hash_diversity":len(jammer_channel_hashes),"jammer_waveform_hash_diversity":len(jammer_waveform_hashes),
            "snr_sample_range":[min(snr_samples),max(snr_samples)],"jsr_sample_range":[min(jsr_samples),max(jsr_samples)],
            "initialization_comparison_metrics":{"unseen_loss":unseen_groups["aggregate"]["normalized_mse"],
                "layers1_to_7_improvement":unseen_groups["layers1_to_7"]["relative_improvement_over_zero"],
                "layers6_to_7_improvement":unseen_groups["layers6_to_7"]["relative_improvement_over_zero"],
                "layer7_improvement":unseen_groups["layer7"]["relative_improvement_over_zero"],
                "gradient_finite_ratio":summary["fraction_finite_batches"],"output_power_ratio":unseen_groups["aggregate"]["power_ratio"],
                "post_eq_sinr_conditioned_loss":final_validation["j2_strongest_selected_condition"]["aggregate"]["normalized_mse"],
                "convergence_slope":gate["final_validation_slope"]}})
    if args.stage==J3_STAGE:
        summary.update({"classification":j3_classification,"provisional_thresholds":J3_THRESHOLDS,"jammer_channel_hash_diversity":len(jammer_channel_hashes),"jammer_waveform_hash_diversity":len(jammer_waveform_hashes),"jammer_mask_hash_diversity":len(mask_hashes),"snr_sample_range":[min(snr_samples),max(snr_samples)],"global_jsr_sample_range":[min(jsr_samples),max(jsr_samples)],"jammed_fraction_values":sorted(set(fraction_samples)),"stochastic_diversity":{"legitimate_channel_hashes":len(channel_hashes),"jammer_channel_hashes":len(jammer_channel_hashes),"jammer_waveform_hashes":len(jammer_waveform_hashes),"awgn_hashes":len(noise_hashes),"mask_hashes":len(mask_hashes)},"accepted_j2":j2_accepted,"selected_training_distribution":selected_distribution})
        write_training_curves(history,out/"plots")
    if args.stage==J4_STAGE:
        summary.update({"classification":j4_classification,"provisional_thresholds":J4_THRESHOLDS,"jammer_channel_hash_diversity":len(jammer_channel_hashes),"jammer_waveform_hash_diversity":len(jammer_waveform_hashes),"jammer_mask_hash_diversity":len(mask_hashes),"snr_sample_range":[min(snr_samples),max(snr_samples)],"global_jsr_sample_range":[min(jsr_samples),max(jsr_samples)],"burst_fraction_values":sorted(set(fraction_samples)),"stochastic_diversity":{"legitimate_channel_hashes":len(channel_hashes),"jammer_channel_hashes":len(jammer_channel_hashes),"jammer_waveform_hashes":len(jammer_waveform_hashes),"awgn_hashes":len(noise_hashes),"mask_hashes":len(mask_hashes)},"accepted_j3":j3_accepted,"selected_training_distribution":selected_distribution});write_training_curves(history,out/"plots")
    (out / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
