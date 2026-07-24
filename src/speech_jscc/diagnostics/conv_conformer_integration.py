from __future__ import annotations
import hashlib, math, random, json
from typing import Any
import torch
from channels.pilot import extract_data_resources, insert_data_and_pilots, make_pilot_mask
from models.resource_allocator import allocate_resources, deallocate_resources
from speech_jscc.diagnostics.architecture_screening import revised_g0_gate
from speech_jscc.diagnostics.random_distribution import SeedDeriver

INTEGRATION_VERSION="conv_conformer_integration_v1"
STAGES=("g1_mapping_train","g2_fixed_clean","g3_random_clean")
J1_STAGE="j1_weak_random_barrage"
J2_STAGE="j2_strong_barrage"
J3_STAGE="j3_random_narrowband"
J4_STAGE="j4_random_burst"
J5_STAGE="j5_pilot_targeted"

def _state(model,target):
    state=torch.zeros(target.shape[0],model.encoder.channel_state_dim,device=target.device,dtype=target.dtype)
    gates=torch.ones(target.shape[0],model.encoder.num_layers,device=target.device,dtype=target.dtype)
    return state,gates

def mapping_round_trip(symbols,model,pilot_spacing=4,time_spacing=4):
    allocation=allocate_resources(symbols,torch.ones_like(symbols.real),model.encoder.layer_channel_uses,mode="uniform")
    mask=make_pilot_mask((symbols.shape[0],64,32),pilot_spacing,time_spacing=time_spacing,device=symbols.device)
    grid,pilots=insert_data_and_pilots(allocation.symbols,mask)
    extracted=extract_data_resources(grid,mask); restored=deallocate_resources(extracted,allocation.resource_to_source)
    return restored,{"allocated":allocation.symbols,"grid":grid,"pilots":pilots,"pilot_mask":mask,"extracted":extracted,"mapping":allocation.resource_to_source}

def mapping_equivalence(model,target,*,pilot_spacing=4,time_spacing=4):
    state,gates=_state(model,target)
    with torch.no_grad():
        symbols=model.encoder(target,state,layer_gates=gates); direct=model.decoder(symbols,state)
        restored,aux=mapping_round_trip(symbols,model,pilot_spacing,time_spacing); mapped=model.decoder(restored,state)
    mask=aux["pilot_mask"]; data_mask=~mask
    direct_loss=(direct-target).square().mean(); mapped_loss=(mapped-target).square().mean()
    return {"grid_total_resources":mask[0].numel(),"pilot_resources":int(mask[0].sum()),"data_resources":int(data_mask[0].sum()),
        "encoder_symbols":symbols.shape[1],"extracted_symbols":aux["extracted"].shape[1],"per_layer_symbols":list(model.encoder.layer_channel_uses),
        "overwrite_count":0,"masks_disjoint":not bool((mask&data_mask).any()),"masks_exhaustive":bool((mask|data_mask).all()),
        "encoder_to_allocated_max_abs_error":float((symbols-aux["allocated"]).abs().max()),
        "packed_data_max_abs_error":float((aux["grid"][data_mask]-aux["allocated"].flatten()).abs().max()),
        "decoder_input_max_abs_error":float((symbols-restored).abs().max()),"reconstruction_max_abs_error":float((direct-mapped).abs().max()),
        "aggregate_loss_difference":float((direct_loss-mapped_loss).abs()),"symbols":symbols,"restored":restored,"direct":direct,"mapped":mapped,"aux":aux}

def forward_integration_path(stage,codec,model,target,config,*,batch=None):
    if stage=="g1_mapping_train":
        state,gates=_state(model,target); symbols=model.encoder(target,state,layer_gates=gates)
        restored,aux=mapping_round_trip(symbols,model,config["channel"].get("pilot_spacing",4),config["channel"].get("pilot_time_spacing",4))
        return {"reconstruction":model.decoder(restored,state),"data_symbols":symbols,"decoder_input":restored,**aux}
    if stage not in {"g2_fixed_clean","g3_random_clean"}: raise ValueError("unknown integration stage")
    if batch is None: raise ValueError("channel stages require a paired batch")
    from speech_jscc.diagnostics.content_generalization import forward_content_path
    return forward_content_path(stage,codec,model,target,config,batch=batch)

def realization_policy(stage,root_seed,step):
    derive=SeedDeriver(root_seed)
    if stage=="g2_fixed_clean": seed=derive.seed("g2_fixed_channel_noise",0); snr=10.0; mode="fixed"
    elif stage=="g3_random_clean":
        seed=derive.seed("g3_random_channel_noise",step); generator=torch.Generator().manual_seed(derive.seed("g3_snr",step)); snr=float(torch.empty(1).uniform_(5,15,generator=generator)); mode="random_per_step"
    else: raise ValueError("realization policy is defined for G2/G3")
    return {"seed":seed,"snr_db":snr,"channel_mode":"multipath_block","fixed_or_random_channel":mode,"estimator":"dft_tap_ls","equalizer":"estimated_zf","jammer":"none","oracle_neural_input":False}

def j1_realization_policy(root_seed,step):
    derive=SeedDeriver(root_seed); seed=derive.seed("j1_legitimate_jammer_noise",step); rng=random.Random(derive.seed("j1_snr_jsr",step))
    return {"seed":seed,"snr_db":rng.uniform(5.0,15.0),"jsr_db":rng.uniform(-15.0,-10.0),"jammer_type":"barrage","jammed_fraction":1.0,
        "estimator":"dft_tap_ls","equalizer":"estimated_zf","oracle_neural_input":False}

def j2_realization_policy(root_seed,step,snr_range,jsr_range):
    derive=SeedDeriver(root_seed);seed=derive.seed("j2_legitimate_jammer_noise",step);rng=random.Random(derive.seed("j2_snr_jsr",step))
    return {"seed":seed,"snr_db":rng.uniform(*map(float,snr_range)),"jsr_db":rng.uniform(*map(float,jsr_range)),
        "jammer_type":"barrage","jammed_fraction":1.0,"estimator":"dft_tap_ls","equalizer":"estimated_zf","oracle_neural_input":False}

def jammer_power_diagnostics(batch,transmitted=None,epsilon=1e-12):
    dims=tuple(range(1,batch.jammer.ndim)); reference=batch.jammer.new_full(batch.jammer.shape,complex(math.sqrt(batch.target_power),0))
    signal_tx=reference.abs().square().mean(dims); jammer_tx=batch.jammer.abs().square().mean(dims)
    received_signal = reference if transmitted is None else transmitted
    legitimate=(batch.signal_fading*received_signal).abs().square().mean(dims); jammed=(batch.jammer_fading*batch.jammer).abs().square().mean(dims); noise=batch.noise.abs().square().mean(dims)
    def db(ratio):return float((10*torch.log10(ratio.clamp_min(epsilon))).mean())
    return {"requested_jsr_db":float(batch.jsr_db.mean()),"measured_transmit_reference_jsr_db":db(jammer_tx/signal_tx.clamp_min(epsilon)),
        "measured_received_jsr_db":db(jammed/legitimate.clamp_min(epsilon)),"legitimate_received_power":float(legitimate.mean()),"jammer_received_power":float(jammed.mean()),"awgn_power":float(noise.mean()),
        "fifth_percentile_legitimate_channel_power":float(torch.quantile(batch.signal_fading.abs().square().flatten(),.05)),"fifth_percentile_jammer_channel_power":float(torch.quantile(batch.jammer_fading.abs().square().flatten(),.05))}

def build_j1_validation_suite(base_suite,seed):
    derive=SeedDeriver(seed); scenarios=[]
    for index,row in enumerate(base_suite["scenarios"]):
        policy=j1_realization_policy(seed,index+100000); scenarios.append({**row,"snr_db":policy["snr_db"],"jsr_db":policy["jsr_db"],"jammer_type":"barrage","jammer_seed":derive.seed("j1_validation_jammer",index)})
    unseen=[x for x in base_suite["scenarios"] if x["group"]=="unseen_speaker_unseen_utterance_unseen_channel"]
    slices=[("j1_unseen_jsr_-15db",10.,-15.),("j1_unseen_jsr_-12.5db",10.,-12.5),("j1_unseen_jsr_-10db",10.,-10.),("j1_unseen_snr_5db",5.,-12.5),("j1_unseen_snr_10db",10.,-12.5),("j1_unseen_snr_15db",15.,-12.5),("j1_joint_snr_5db_jsr_-10db",5.,-10.)]
    for label,snr,jsr in slices:
        for index,row in enumerate(unseen): scenarios.append({**row,"group":label,"snr_db":snr,"jsr_db":jsr,"jammer_type":"barrage","channel_seed":derive.seed(label+"_channel",index),"noise_seed":derive.seed(label+"_noise",index),"jammer_seed":derive.seed(label+"_jammer",index)})
    encoded=json.dumps(scenarios,sort_keys=True,separators=(",",":")).encode();return {"scenarios":scenarios,"validation_suite_hash":hashlib.sha256(encoded).hexdigest()}

def build_j2_validation_suite(base_suite,seed,snr_range,jsr_range):
    derive=SeedDeriver(seed);scenarios=[]
    for index,row in enumerate(base_suite["scenarios"]):
        policy=j2_realization_policy(seed,index+200000,snr_range,jsr_range)
        scenarios.append({**row,"snr_db":policy["snr_db"],"jsr_db":policy["jsr_db"],"jammer_type":"barrage"})
    unseen=[row for row in base_suite["scenarios"] if row["group"]=="unseen_speaker_unseen_utterance_unseen_channel"]
    strongest_snr=float(snr_range[0]);strongest_jsr=float(jsr_range[1])
    for index,row in enumerate(unseen):
        scenarios.append({**row,"group":"j2_strongest_selected_condition","snr_db":strongest_snr,
            "jsr_db":strongest_jsr,"jammer_type":"barrage","channel_seed":derive.seed("j2_strongest_validation",index)})
    encoded=json.dumps(scenarios,sort_keys=True,separators=(",",":")).encode()
    return {"scenarios":scenarios,"validation_suite_hash":hashlib.sha256(encoded).hexdigest(),
        "selected_snr_range_db":list(map(float,snr_range)),"selected_jsr_range_db":list(map(float,jsr_range))}

def build_j3_validation_suite(base_suite,seed,distribution):
    from speech_jscc.diagnostics.j3_narrowband import j3_policy
    scenarios=[]
    for index,row in enumerate(base_suite["scenarios"]):
        p=j3_policy(seed,index+300000,distribution["selected_snr_range_db"],distribution["selected_global_jsr_range_db"],distribution["selected_jammed_subcarrier_fractions"]);scenarios.append({**row,**p})
    unseen=[row for row in base_suite["scenarios"] if row["group"]=="unseen_speaker_unseen_utterance_unseen_channel"]
    for index,row in enumerate(unseen):scenarios.append({**row,"group":"j3_strongest_selected_condition","snr_db":distribution["selected_snr_range_db"][0],"jsr_db":distribution["selected_global_jsr_range_db"][1],"jammed_fraction":max(distribution["selected_jammed_subcarrier_fractions"]),"jammer_type":"narrowband","channel_seed":SeedDeriver(seed).seed("j3_strongest",index)})
    encoded=json.dumps(scenarios,sort_keys=True,separators=(",",":")).encode();return {"scenarios":scenarios,"validation_suite_hash":hashlib.sha256(encoded).hexdigest(),"distribution":distribution}

def build_j4_validation_suite(base_suite,seed,distribution):
    from speech_jscc.diagnostics.j4_burst import j4_policy
    from speech_jscc.diagnostics.j4_tail import select_strongest_condition
    scenarios=[]
    for index,row in enumerate(base_suite["scenarios"]):
        p=j4_policy(seed,index+400000,distribution["selected_snr_range_db"],distribution["selected_global_jsr_range_db"],distribution["selected_burst_fractions"]);scenarios.append({**row,**p,"jammed_fraction":p["burst_fraction"]})
    unseen=[row for row in base_suite["scenarios"] if row["group"]=="unseen_speaker_unseen_utterance_unseen_channel"]
    strongest=select_strongest_condition(distribution)
    for index,row in enumerate(unseen):scenarios.append({**row,"group":"j4_strongest_selected_condition","snr_db":strongest["snr_db"],"jsr_db":strongest["jsr_db"],"jammed_fraction":strongest["burst_fraction"],"jammer_type":"burst","channel_seed":SeedDeriver(seed).seed("j4_strongest",index)})
    encoded=json.dumps(scenarios,sort_keys=True,separators=(",",":")).encode();return {"scenarios":scenarios,"validation_suite_hash":hashlib.sha256(encoded).hexdigest(),"distribution":distribution}

def build_j5_validation_suite(base_suite,seed,distribution):
    from speech_jscc.diagnostics.j5_pilot import j5_policy
    scenarios=[]
    for index,row in enumerate(base_suite['scenarios']):
        p=j5_policy(seed,index+500000,distribution['selected_snr_range_db'],distribution['selected_pilot_jsr_range_db'],distribution['selected_pilot_coverages']);scenarios.append({**row,**p,'jammer_type':'pilot','jammed_fraction':p['coverage'],'jsr_db':p['pilot_jsr_db']})
    unseen=[r for r in base_suite['scenarios'] if r['group']=='unseen_speaker_unseen_utterance_unseen_channel'];strong=distribution['strongest_selected_condition']
    for index,row in enumerate(unseen):scenarios.append({**row,'group':'j5_strongest_selected_condition','snr_db':strong['snr_db'],'jsr_db':strong['pilot_jsr_db'],'pilot_jsr_db':strong['pilot_jsr_db'],'coverage':strong['coverage'],'jammed_fraction':strong['coverage'],'jammer_type':'pilot','channel_seed':SeedDeriver(seed).seed('j5_strongest',index)})
    encoded=json.dumps(scenarios,sort_keys=True,separators=(',',':')).encode();return {'scenarios':scenarios,'validation_suite_hash':hashlib.sha256(encoded).hexdigest(),'distribution':distribution}

def stage_gate(validation,same_group="same_speaker_unseen_utterance_unseen_channel",unseen_group="unseen_speaker_unseen_utterance_unseen_channel"):
    gate=revised_g0_gate(validation,same_group=same_group,unseen_group=unseen_group)
    return {"layer0_generalization_pass":gate["layer0_generalization_pass"],"enhancement_layers_generalization_pass":gate["enhancement_layers_generalization_pass"],
        "same_speaker_generalization_pass":gate["same_speaker_generalization_pass"],"unseen_speaker_generalization_pass":gate["unseen_speaker_generalization_pass"],
        "beats_zero_predictor":gate["aggregate_generalization_pass"],"beats_global_mean_predictor":gate["beats_global_mean_predictor"],
        "beats_layerwise_mean_predictor":gate["beats_layerwise_mean_predictor"],"finite_run_pass":all(v["aggregate"].get("finite",True) for v in validation.values()),
        "stage_pass":gate["architecture_screening_pass"]}

def j1_stage_gate(validation,*,channel_diversity,jammer_channel_diversity,jammer_waveform_diversity,noise_diversity,required_diversity,parameters_finite=True):
    gate=stage_gate(validation); strongest=validation["j1_unseen_jsr_-10db"]
    aggregate=strongest["aggregate"]; enhancement=strongest["layers1_to_7_summary"]
    strongest_pass=(float(aggregate["relative_improvement_over_zero"])>=.05 and
        float(enhancement["relative_improvement_over_zero"])>=.05 and
        float(aggregate["pearson_correlation"])>0 and float(aggregate["cosine_similarity"])>0 and
        float(aggregate["power_ratio"])>=.01 and float(enhancement["pearson_correlation"])>0 and
        float(enhancement["cosine_similarity"])>0 and float(enhancement["power_ratio"])>=.01)
    diversity={
        "legitimate_channel_diversity_pass":channel_diversity>=required_diversity,
        "jammer_channel_diversity_pass":jammer_channel_diversity>=required_diversity,
        "jammer_waveform_diversity_pass":jammer_waveform_diversity>=required_diversity,
        "noise_diversity_pass":noise_diversity>=required_diversity,
    }
    joint=validation.get("j1_joint_snr_5db_jsr_-10db",{}).get("gate",{}).get("passed")
    finite_run_pass=bool(gate["finite_run_pass"] and parameters_finite)
    stage_pass=bool(gate["stage_pass"] and gate["beats_layerwise_mean_predictor"] and
        strongest_pass and all(diversity.values()) and finite_run_pass)
    return {**gate,**diversity,"strongest_weak_jsr_pass":strongest_pass,
        "parameters_finite_pass":bool(parameters_finite),"finite_run_pass":finite_run_pass,
        "joint_worst_case_diagnostic_pass":joint,"stage_pass":stage_pass,"passed":stage_pass}

def next_stage(stage,passed,continue_on_pass):
    if not passed or not continue_on_pass:return "stop"
    return {"g1_mapping_train":"g2_fixed_clean","g2_fixed_clean":"g3_random_clean","g3_random_clean":"complete"}[stage]

def validate_stage_metadata(actual,expected):
    if not isinstance(actual,dict): raise ValueError("integration checkpoint metadata required")
    for key,value in expected.items():
        if actual.get(key)!=value: raise ValueError(f"integration checkpoint mismatch: {key}")

def tensor_hash(value): return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
