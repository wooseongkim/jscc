from __future__ import annotations
import argparse, copy, hashlib, json, random, shutil
from pathlib import Path
import torch, yaml
from channels.multipath import exponential_pdp
from channels.temporal_multipath import correlated_tap_trajectory, jakes_slot_correlation
from speech_jscc.config import resolve_device
from speech_jscc.experiment import build_components
from speech_jscc.training.channel_free_feasibility import decode_frozen_representation_with_gradient
from speech_jscc.training.r4_si_sdr_finetune import base_loss_weights, si_sdr_weight
from speech_jscc.training.r4_waveform_finetune import R4ForwardCondition,R4WaveformForward,freeze_codec_for_input_gradient,r4_training_objective,validate_initial_checkpoint_metadata
from train_channel_free_conv_conformer import fixed_paths,load_batch,sha256

def _component_grad_norm(component, parameters):
 parameters=list(parameters)
 if component is None:
  return 0.0, True
 gradients=torch.autograd.grad(component,parameters,retain_graph=True,allow_unused=True)
 values=[g.detach().float() for g in gradients if g is not None]
 if not values:return 0.0, True
 squared=sum((g.square().sum() for g in values),values[0].new_zeros(()))
 return float(squared.sqrt()), bool(all(torch.isfinite(g).all() for g in values))

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',default='configs/train_r4_si_sdr_finetune.yaml');p.add_argument('--experiment',choices=('control_no_si_sdr','si_sdr_low','si_sdr_medium'),required=True);p.add_argument('--max-steps',type=int);p.add_argument('--device');p.add_argument('--output-dir');p.add_argument('--dry-run',action='store_true');p.add_argument('--overwrite',action='store_true');p.add_argument('--allow-long-run',action='store_true');a=p.parse_args()
 cfgp=Path(a.config).resolve();c=yaml.safe_load(cfgp.read_text()); root=Path(__file__).resolve().parent
 src=root/c['source_checkpoint']; out=Path(a.output_dir) if a.output_dir else root/c['output_root']/a.experiment; steps=a.max_steps or c['training']['local_steps']; dev=a.device or c['device']
 if a.dry_run: print(json.dumps({'experiment':a.experiment,'source_checkpoint':str(src),'source_global_step':c['source_global_step'],'local_steps':steps,'device':dev},indent=2));return
 if steps>5 and not a.allow_long_run: raise SystemExit('requires --allow-long-run')
 if out.exists():
  if not a.overwrite: raise SystemExit(f'refusing existing output: {out}')
  shutil.rmtree(out)
 out.mkdir(parents=True); payload=torch.load(src,map_location='cpu',weights_only=False); mc=copy.deepcopy(payload['config']);
 for k in ('config_path','checkpoint_path'):
  if k in mc['codec'] and not Path(mc['codec'][k]).is_absolute(): mc['codec'][k]=str(root/mc['codec'][k])
 mc['device']=dev;validate_initial_checkpoint_metadata(mc); device=resolve_device(dev);codec,model=build_components(mc,device);model.load_state_dict(payload['model']);freeze_codec_for_input_gradient(codec);trainable=[*model.encoder.parameters(),*model.decoder.parameters()];opt=torch.optim.Adam(trainable,lr=float(c['training']['learning_rate']))
 base=base_loss_weights(int(c['source_global_step']),c['loss']); target=float(c['experiments'][a.experiment]['si_sdr_target_weight']);(out/'resolved_config.yaml').write_text(yaml.safe_dump(c));(out/'source_checkpoint.json').write_text(json.dumps({'path':str(src),'source_global_step':c['source_global_step'],'resume_mode':'weights_only','restored_fields':['model'],'reset_fields':['optimizer','rng','scheduler']}));(out/'source_checkpoint_sha256.txt').write_text(sha256(src));
 train,_=fixed_paths(mc,int(mc['seed'])); phys=yaml.safe_load((root/'configs/ofdm_nr_like_r4.yaml').read_text());pdp=exponential_pdp(phys['channel']['num_taps'],phys['channel']['pdp_decay']);rho=jakes_slot_correlation(phys['physical']['user_speed_mps'],phys['physical']['carrier_frequency_hz'],.001);traj=correlated_tap_trajectory(slots=steps,batch_size=1,pdp=pdp,rho=rho,seed=123);engine=R4WaveformForward(codec,model);report=None;hist={}
 with (out/'training_log.jsonl').open('w') as f:
  for local in range(steps):
   snr=random.Random(local).choices([5.,10.,15.],[.7,.2,.1])[0];wave=load_batch([train[local%len(train)]],mc,device)
   with torch.no_grad(): target_lat=codec.encode_waveform(wave)
   result=engine.forward(target_lat,wave,R4ForwardCondition(snr_db=snr,tti=local,tap_coefficients=traj[local].to(device),noise_seed=10000+local,fixed_mapping=local==0),report,training=True);report=result.next_delayed_csi;state=target_lat.new_zeros((1,model.encoder.channel_state_dim));cf=model.decoder(model.encoder(target_lat,state),state);w=dict(base);w['si_sdr']=si_sdr_weight(local,target=target,start=c['loss']['si_sdr_ramp']['start'],full=c['loss']['si_sdr_ramp']['full']);loss,parts=r4_training_objective(result.reconstruction,target_lat,wave,lambda x:decode_frozen_representation_with_gradient(codec,x),weights=w,channel_free_reconstruction=cf,fft_sizes=tuple(c['loss']['fft_sizes']),si_sdr_clip_db=c['loss']['si_sdr_clip_db']);opt.zero_grad();weighted_si=w['si_sdr']*parts.get('si_sdr',loss.new_zeros(()));si_enc,si_enc_finite=_component_grad_norm(weighted_si if w['si_sdr'] else None,model.encoder.parameters());si_dec,si_dec_finite=_component_grad_norm(weighted_si if w['si_sdr'] else None,model.decoder.parameters());loss.backward();enc=sum((x.grad.detach().float().square().sum() for x in model.encoder.parameters() if x.grad is not None),target_lat.new_zeros(())).sqrt();dec=sum((x.grad.detach().float().square().sum() for x in model.decoder.parameters() if x.grad is not None),target_lat.new_zeros(())).sqrt();total_finite=bool(torch.isfinite(enc) and torch.isfinite(dec));torch.nn.utils.clip_grad_norm_(trainable,c['training']['gradient_clip_norm']);opt.step();row={'local_step':local+1,'source_global_step':c['source_global_step'],'si_sdr_weight':w['si_sdr'],'total_loss':float(loss.detach()),'loss_components':{k:float(v.detach()) for k,v in parts.items()},'weighted_si_sdr_loss':float(weighted_si.detach()),'si_sdr_encoder_grad_norm':si_enc,'si_sdr_decoder_grad_norm':si_dec,'si_sdr_encoder_grad_finite':si_enc_finite,'si_sdr_decoder_grad_finite':si_dec_finite,'jscc_encoder_grad_norm':float(enc),'jscc_decoder_grad_norm':float(dec),'total_gradient_finite':total_finite,'optimizer_update_applied':True,'codec_parameters_with_grad':sum(x.grad is not None for x in codec.parameters())};f.write(json.dumps(row)+'\n');hist[str(snr)]=hist.get(str(snr),0)+1
   if local in {0,249,499,749,999,1499,1999,2499,2999,steps-1}: torch.save({'diagnostic_type':'r4_si_sdr_finetune','model':model.state_dict(),'optimizer':opt.state_dict(),'local_step':local+1,'source_global_step':c['source_global_step'],'config':mc},out/f'local_step_{local+1:06d}.pt')
 torch.save({'diagnostic_type':'r4_si_sdr_finetune','model':model.state_dict(),'local_step':steps,'config':mc},out/'last.pt');(out/'snr_sampling_histogram.json').write_text(json.dumps(hist));(out/'loss_summary.json').write_text(json.dumps({'steps':steps,'si_sdr_target_weight':target}));
if __name__=='__main__':main()
