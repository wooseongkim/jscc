from __future__ import annotations
import argparse,hashlib,json,random,sys
from pathlib import Path
import torch,yaml
from speech_jscc.config import load_config,resolve_device
from speech_jscc.experiment import build_components
from speech_jscc.data import resolve_waveform_splits,load_waveform_segment
from speech_jscc.diagnostics.architecture_screening import direct_bypass
from speech_jscc.diagnostics.content_generalization import parse_speaker_id
from src.evaluation.clean_end_to_end import CheckpointSelector
from src.evaluation.waveform_metrics import waveform_metrics
from speech_jscc.training.channel_free_feasibility import multi_resolution_stft_loss,negative_si_sdr_loss,summed_latent_loss,configure_bottleneck,select_unseen_speaker_paths,enable_frozen_rnn_backward,decode_frozen_representation_with_gradient
from train_latent_jscc import layer_weighted_latent_mse

def parse():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--experiment',choices=('a','b','c'),required=True);p.add_argument('--output-dir',required=True);p.add_argument('--device',default='cuda');p.add_argument('--steps',type=int,default=4096);p.add_argument('--batch-size',type=int,default=4);p.add_argument('--seed',type=int,default=23);p.add_argument('--validation-every',type=int,default=100);p.add_argument('--checkpoint-every',type=int,default=500);p.add_argument('--overwrite',action='store_true');p.add_argument('--resume');p.add_argument('--allow-long-run',action='store_true');p.add_argument('--dry-run',action='store_true');return p.parse_args()
def load_batch(paths,cfg,device):return torch.stack([load_waveform_segment(p,int(cfg['codec']['sample_rate']),int(cfg['codec']['waveform_samples'])) for p in paths]).to(device)
def encode(codec,wave):
 with torch.no_grad():return codec.encode_waveform(wave)
def validate(codec,model,paths,cfg,device):
 rows=[];sr=int(cfg['codec']['sample_rate']);model.eval()
 with torch.no_grad():
  for path in paths:
   wave=load_batch([path],cfg,device);target=codec.encode_waveform(wave);state=torch.zeros(1,model.encoder.channel_state_dim,device=device);recon=direct_bypass(model,target,state);clean=codec.decode_representation(target);decoded=codec.decode_representation(recon);clean_m=waveform_metrics(wave,clean,sr);current=waveform_metrics(wave,decoded,sr);latent,_=layer_weighted_latent_mse(recon,target,torch.ones(8,device=device),cfg['train']['latent_normalization']);rows.append({'path':str(path),'latent_loss':float(latent),'summed_latent_loss':float(summed_latent_loss(recon,target)),'clean_si_sdr':clean_m['si_sdr_db'],'si_sdr':current['si_sdr_db'],'delta_si_sdr':current['si_sdr_db']-clean_m['si_sdr_db'],'waveform_snr':current['waveform_snr_db'],'delta_waveform_snr':current['waveform_snr_db']-clean_m['waveform_snr_db'],'stft_ratio':current['stft_l1']/max(clean_m['stft_l1'],1e-12)})
 return {'rows':rows,'latent_loss':sum(x['latent_loss'] for x in rows)/len(rows),'delta_si_sdr':sum(x['delta_si_sdr'] for x in rows)/len(rows),'si_sdr':sum(x['si_sdr'] for x in rows)/len(rows),'delta_waveform_snr':sum(x['delta_waveform_snr'] for x in rows)/len(rows),'stft_ratio':sum(x['stft_ratio'] for x in rows)/len(rows)}
def main():
 a=parse();cfg=load_config(a.config)
 if 'latent_cache_dir' in cfg.get('data',{}):raise SystemExit('latent cache is forbidden')
 uses=7680 if a.experiment=='c' else 1920;cfg=configure_bottleneck(cfg,uses)
 if a.dry_run:print(json.dumps({'dry_run':True,'experiment':a.experiment,'steps':a.steps,'channel_uses':uses,'latent_cache':False,'output_dir':a.output_dir},indent=2));return
 if a.steps>5 and not a.allow_long_run:raise SystemExit('long feasibility training requires --allow-long-run')
 out=Path(a.output_dir)
 if out.exists() and not (a.overwrite or a.resume):raise SystemExit(f'refusing existing output directory: {out}')
 cfg['device']=a.device;device=resolve_device(a.device);codec,model=build_components(cfg,device);codec.eval();[p.requires_grad_(False) for p in codec.parameters()];rnn_backward_modules=0
 if a.experiment in ('b','c'):
  decoder=getattr(getattr(codec,'model',codec),'decoder',codec);rnn_backward_modules=enable_frozen_rnn_backward(decoder)
 train_paths,val_paths=resolve_waveform_splits(cfg['data'],a.seed);rng=random.Random(a.seed);rng.shuffle(train_paths);train_paths=train_paths[:256];unseen=select_unseen_speaker_paths(train_paths,val_paths,limit=64,seed=a.seed+1);validation=unseen[:8];optimizer=torch.optim.Adam(model.parameters(),lr=float(cfg['train']['learning_rate']));weights=torch.ones(8,device=device);selector=CheckpointSelector();history=[];start=0
 if a.resume:
  cp=torch.load(a.resume,map_location='cpu',weights_only=False);model.load_state_dict(cp['model'],strict=True);optimizer.load_state_dict(cp['optimizer']);start=cp['step'];history=cp['history']
 out.mkdir(parents=True,exist_ok=True);(out/'resolved_config.yaml').write_text(yaml.safe_dump(cfg));(out/'command.txt').write_text(' '.join(sys.argv)+'\n');(out/'dataset_manifest.json').write_text(json.dumps({'train':[str(p) for p in train_paths],'unseen_validation':[str(p) for p in unseen],'train_hash':hashlib.sha256('\n'.join(map(str,train_paths)).encode()).hexdigest(),'validation_hash':hashlib.sha256('\n'.join(map(str,unseen)).encode()).hexdigest(),'latent_cache_used':False},indent=2));lambdas=cfg['feasibility'];order=[]
 for epoch in range((a.steps*a.batch_size+len(train_paths)-1)//len(train_paths)+1):
  current=list(train_paths);random.Random(a.seed+epoch).shuffle(current);order.extend(current)
 with (out/'metrics.jsonl').open('a' if a.resume else 'w') as log:
  for step in range(start+1,a.steps+1):
   paths=order[(step-1)*a.batch_size:step*a.batch_size];wave=load_batch(paths,cfg,device);target=encode(codec,wave);state=torch.zeros(target.shape[0],model.encoder.channel_state_dim,device=device);model.train();optimizer.zero_grad(set_to_none=True);recon=direct_bypass(model,target,state);latent,_=layer_weighted_latent_mse(recon,target,weights,cfg['train']['latent_normalization']);summed=summed_latent_loss(recon,target);stft=latent.new_zeros(());neg_sisdr=latent.new_zeros(())
   if a.experiment in ('b','c'):
    clean=codec.decode_representation(target).detach();decoded=decode_frozen_representation_with_gradient(codec,recon);stft=multi_resolution_stft_loss(decoded,clean,fft_sizes=tuple(lambdas['fft_sizes']));neg_sisdr=negative_si_sdr_loss(decoded,clean);loss=latent+float(lambdas['lambda_sum'])*summed+float(lambdas['lambda_stft'])*stft+float(lambdas['lambda_sisdr'])*neg_sisdr
   else:loss=latent
   loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),float(cfg['train']['gradient_clip_norm']));optimizer.step();row={'step':step,'loss':float(loss.detach()),'latent_loss':float(latent.detach()),'summed_latent_loss':float(summed.detach()),'stft_loss':float(stft.detach()),'negative_si_sdr_loss':float(neg_sisdr.detach())}
   if step==1 or step%a.validation_every==0 or step==a.steps:
    value=validate(codec,model,validation,cfg,device);row['validation']=value;payload={'diagnostic_type':'channel_free_feasibility','experiment':a.experiment,'channel_uses':uses,'latent_cache_used':False,'model':model.state_dict(),'optimizer':optimizer.state_dict(),'step':step,'history':history+[row],'config':cfg};old_l=selector.best_latent['latent_loss'] if selector.best_latent else None;old_w=selector.best_waveform['delta_si_sdr'] if selector.best_waveform else None;selector.update(step=step,latent_loss=value['latent_loss'],delta_si_sdr=value['delta_si_sdr'],path='')
    if old_l is None or value['latent_loss']<old_l:torch.save(payload,out/'best_latent.pt');selector.best_latent['path']=str(out/'best_latent.pt')
    if old_w is None or value['delta_si_sdr']>old_w:torch.save(payload,out/'best_waveform.pt');selector.best_waveform['path']=str(out/'best_waveform.pt')
   history.append(row);log.write(json.dumps(row)+'\n');log.flush()
   if step%a.checkpoint_every==0 or step==a.steps:torch.save({'diagnostic_type':'channel_free_feasibility','experiment':a.experiment,'channel_uses':uses,'latent_cache_used':False,'model':model.state_dict(),'optimizer':optimizer.state_dict(),'step':step,'history':history,'config':cfg},out/'final.pt')
 summary={'experiment':a.experiment,'steps':a.steps,'channel_uses':uses,'latent_cache_used':False,'speech_tokenizer_trainable_parameters':sum(p.numel() for p in codec.parameters() if p.requires_grad),'frozen_decoder_rnn_backward_modules':rnn_backward_modules,'best_latent_checkpoint':selector.best_latent,'best_waveform_checkpoint':selector.best_waveform,'checkpoints_differ':selector.best_latent['step']!=selector.best_waveform['step'],'final':history[-1]};(out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
