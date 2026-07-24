#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json,shutil
from pathlib import Path
import torch
from speech_jscc.config import load_config
from speech_jscc.experiment import build_components
from speech_jscc.diagnostics.j2_barrage import file_sha256
from speech_jscc.diagnostics.j3_narrowband import j3_initialization_metadata
from speech_jscc.diagnostics.o5_root_cause import stable_tensor_hash

SCHEMA="j3_accepted_provenance_v2"

def _json(path):return json.loads(Path(path).read_text())

def verify_loaded_weights(config_path,j2_checkpoint,j3_summary):
    config=load_config(config_path);config["device"]="cpu";_,model=build_components(config,torch.device("cpu"))
    parent=torch.load(j2_checkpoint,map_location="cpu",weights_only=False);model.load_state_dict(parent["model"],strict=True)
    actual=stable_tensor_hash(torch.cat([p.detach().flatten().cpu() for p in model.parameters()]))
    expected=_json(j3_summary)["provenance"]["model_initialization_hash"]
    if actual!=expected:raise ValueError("J2 strict-loaded parameter hash does not match J3 recorded initialization hash")
    return {"initial_weights_loaded":True,"strict_load_succeeded":True,"loaded_parameter_hash":actual}

def correct_j3_provenance(j3_summary,j3_checkpoint,j2_summary,j2_checkpoint,*,config_hash,initial_weights_loaded):
    paths=map(Path,(j3_summary,j3_checkpoint,j2_summary,j2_checkpoint));j3_summary,j3_checkpoint,j2_summary,j2_checkpoint=paths
    original_bytes=j3_summary.read_bytes();original=json.loads(original_bytes);parent=_json(j2_summary)
    if original.get("classification")!="PASS" or parent.get("classification")!="PASS":raise ValueError("accepted J2 and passing J3 summaries required")
    original_copy=j3_summary.with_name("summary.original.json")
    if original_copy.exists() and original_copy.read_bytes()!=original_bytes:raise ValueError("existing summary.original.json differs from source evidence")
    if not original_copy.exists():original_copy.write_bytes(original_bytes)
    transfer=j3_initialization_metadata(str(j2_checkpoint.resolve()),file_sha256(j2_checkpoint),str(j2_summary.resolve()),file_sha256(j2_summary))
    transfer.update({"initial_weights_loaded":bool(initial_weights_loaded),"model_weights_loaded_successfully":bool(initial_weights_loaded),
      "loaded_architecture_version":original["provenance"].get("architecture_version"),"loaded_preprocessing":original["provenance"].get("preprocessing"),
      "loaded_codec_metadata":original["provenance"].get("preprocessing"),"git_commit":original["provenance"].get("git_commit"),"configuration_sha256":config_hash,
      "j3_checkpoint_sha256":file_sha256(j3_checkpoint),"provenance_corrected":True,"provenance_schema_version":SCHEMA})
    corrected={**original,"provenance":{**original["provenance"],**transfer}}
    corrected_path=j3_summary.with_name("summary.corrected.json");corrected_path.write_text(json.dumps(corrected,indent=2)+"\n")
    correction={"schema_version":SCHEMA,"original_summary_sha256":hashlib.sha256(original_bytes).hexdigest(),"corrected_summary_sha256":file_sha256(corrected_path),
      "scientific_metrics_unchanged":original.get("validation")==corrected.get("validation") and original.get("final_train")==corrected.get("final_train"),**transfer}
    j3_summary.with_name("provenance_correction.json").write_text(json.dumps(correction,indent=2)+"\n")
    accepted={"schema_version":SCHEMA,"classification":"ACCEPTED_PASS","summary_path":str(corrected_path.resolve()),"summary_sha256":file_sha256(corrected_path),
      "original_summary_path":str(original_copy.resolve()),"original_summary_sha256":file_sha256(original_copy),"checkpoint_path":str(j3_checkpoint.resolve()),
      "checkpoint_sha256":file_sha256(j3_checkpoint),"provenance_correction_path":str(j3_summary.with_name("provenance_correction.json").resolve()),
      "provenance_corrected":True,**transfer}
    j3_summary.with_name("accepted_manifest.json").write_text(json.dumps(accepted,indent=2)+"\n");return accepted

def main():
    p=argparse.ArgumentParser();p.add_argument("--j3-dir",required=True);p.add_argument("--j2-dir",required=True);p.add_argument("--config",required=True);a=p.parse_args()
    j3=Path(a.j3_dir);j2=Path(a.j2_dir);proof=verify_loaded_weights(a.config,j2/"diagnostic_last.pt",j3/"summary.json")
    accepted=correct_j3_provenance(j3/"summary.json",j3/"diagnostic_last.pt",j2/"summary.json",j2/"diagnostic_last.pt",config_hash=file_sha256(a.config),initial_weights_loaded=proof["initial_weights_loaded"])
    print(json.dumps({"proof":proof,"accepted":accepted},indent=2))
if __name__=="__main__":main()
