"""Paired, scene-level analysis for the round-2 localization candidates."""
from __future__ import annotations
import csv, json
from pathlib import Path
import sys
import numpy as np
import torch
from scipy.stats import rankdata

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'artifacts'/'round2'
sys.path.insert(0,str(ROOT/'src'))
from voice_localization.config import load_config
from voice_localization.manifest import SceneRecord,read_jsonl
from voice_localization.dataset import load_map_sample
from voice_localization.metrics import pairwise_assignment_errors
from voice_localization.model import build_model
from run_round2 import peaks

def load_predict(path, dilations, device):
    state=torch.load(path,map_location=device,weights_only=False)
    model=build_model(dilations).to(device); model.load_state_dict(state.get('model',state)); model.eval(); return model

def errors_by_scene(model,records,device,radius=.6):
    scene_errors=[]; cnn_maps=[]; srp_maps=[]; xs0=ys0=None
    with torch.no_grad():
        for rec in records:
            feat,_,n,truth,xs,ys,_=load_map_sample(rec); xs0=xs;ys0=ys
            p=model(torch.from_numpy(feat[None]).to(device)).sigmoid()[0,0].cpu().numpy()
            srp=feat[0]
            pred=peaks(p,xs,ys,n,radius,True); base=peaks(srp,xs,ys,n,radius,True)
            scene_errors.append(pairwise_assignment_errors(truth,pred))
            cnn_maps.append(p); srp_maps.append(srp)
    return scene_errors,cnn_maps,srp_maps,xs0,ys0

def metrics(errors):
    e=np.concatenate(errors)
    return {'scene_count':len(errors),'source_count':int(e.size),'mean_error_m':float(e.mean()),'median_error_m':float(np.median(e)),
            'p90_error_m':float(np.percentile(e,90)),'within_0_5m':float(np.mean(e<=.5))}

def main():
    cfg=load_config(ROOT/'configs'/'full.yaml'); records=list(read_jsonl(ROOT/'data/full/manifests/scenes_validation.jsonl',SceneRecord))
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); seeds={}
    for seed in (42,43):
        old=load_predict(ROOT/f'artifacts/cnn_mixture_seed{seed}/best.pt',(1,1,1,1),device)
        new=load_predict(OUT/f'confirm/seed{seed}/best.pt',(1,2,4,8),device)
        olde,_,srps,xs,ys=errors_by_scene(old,records,device)
        newe,newmaps,_,_,_=errors_by_scene(new,records,device)
        srpe=[]
        for rec,srp in zip(records,srps):
            srpe.append(pairwise_assignment_errors(np.asarray(rec.source_positions_xy),peaks(srp,xs,ys,rec.n_sources,.6,True)))
        seeds[seed]={'old':olde,'new':newe,'srp':srpe,'newmaps':newmaps,'srpmaps':srps}
        print('ANALYSIS',seed,{k:metrics(seeds[seed][k]) for k in ('srp','old','new')},flush=True)

    # Scene-level paired bootstrap. Keep all sources from a sampled scene together.
    rng=np.random.default_rng(6102026); n=len(records); b=10000
    comparisons={}
    for candidate,reference in [('new','old'),('new','srp'),('old','srp')]:
        boot=[]
        for seed in (42,43):
            diffs=np.full((n,3),np.nan)
            for i,(a,z) in enumerate(zip(seeds[seed][candidate],seeds[seed][reference])):
                diffs[i,:len(a)]=a-z
            ids=rng.integers(0,n,size=(b,n)); sample=diffs[ids]
            sums=np.nansum(sample,axis=(1,2)); counts=np.sum(np.isfinite(sample),axis=(1,2))
            boot.append(sums/counts)
        avg=(boot[0]+boot[1])/2
        point=np.mean([metrics(seeds[s][candidate])['mean_error_m']-metrics(seeds[s][reference])['mean_error_m'] for s in (42,43)])
        comparisons[f'{candidate}_minus_{reference}']={'mean_delta_m':float(point),'ci95_m':[float(np.quantile(avg,.025)),float(np.quantile(avg,.975))],
             'bootstrap_resamples':b,'resampling_unit':'scene','negative_favors_candidate':True}

    strat=[]
    for label, fn in [('n_sources',lambda r:str(r.n_sources)),('n_microphones',lambda r:str(r.n_microphones)),('geometry',lambda r:r.geometry),
                      ('snr_band',lambda r:f'{int(r.snr_requested_db//5)*5}-{int(r.snr_requested_db//5)*5+5}'),
                      ('rt60_band',lambda r:f'{int((r.rt60_realized or r.rt60_requested)//.2)*.2:.1f}-{(int((r.rt60_realized or r.rt60_requested)//.2)+1)*.2:.1f}')]:
        levels=sorted(set(fn(r) for r in records))
        for level in levels:
            ids=[i for i,r in enumerate(records) if fn(r)==level]
            for method in ('srp','old','new'):
                e=[x for i in ids for x in seeds[42][method][i]]+[x for i in ids for x in seeds[43][method][i]]
                strat.append({'stratum':label,'level':level,'method':method,'scene_count':len(ids),'source_count':len(e),'mean_error_m':float(np.mean(e))})

    # Rank-normalized score fusion on the same validation scenes; exploratory.
    blends=[]; blend_errors={}
    for alpha in (.25,.5,.75):
        for seed in (42,43):
            fused=[]
            for rec,prob,srp in zip(records,seeds[seed]['newmaps'],seeds[seed]['srpmaps']):
                rnk=rankdata(srp.ravel(),method='average').reshape(srp.shape)/srp.size
                score=(1-alpha)*rnk+alpha*prob
                fused.append(pairwise_assignment_errors(np.asarray(rec.source_positions_xy),peaks(score,xs,ys,rec.n_sources,.6,True)))
            blends.append({'cnn_weight':alpha,'seed':seed,**metrics(fused)})
            blend_errors[(alpha,seed)]=fused

    blend_inference={}
    for alpha in (.25,.5,.75):
        bootstrap=[]
        for seed in (42,43):
            for ref in ('old','srp'):
                delta=np.full((n,3),np.nan)
                for i,(a,z) in enumerate(zip(blend_errors[(alpha,seed)],seeds[seed][ref])):
                    delta[i,:len(a)]=a-z
                ids=rng.integers(0,n,size=(b,n)); sample=delta[ids]
                bootstrap.append((seed,ref,np.nansum(sample,axis=(1,2))/np.sum(np.isfinite(sample),axis=(1,2))))
        for ref in ('old','srp'):
            sample_means=[x[2] for x in bootstrap if x[1]==ref]
            mean_delta=np.mean([metrics(blend_errors[(alpha,s)])['mean_error_m']-metrics(seeds[s][ref])['mean_error_m'] for s in (42,43)])
            avg=(sample_means[0]+sample_means[1])/2
            blend_inference[f"cnn_weight_{alpha}_minus_{ref}"]={'mean_delta_m':float(mean_delta),'ci95_m':[float(np.quantile(avg,.025)),float(np.quantile(avg,.975))]}

    blend_strat=[]
    for label, fn in [('n_sources',lambda r:str(r.n_sources)),('n_microphones',lambda r:str(r.n_microphones)),('geometry',lambda r:r.geometry),
                      ('snr_band',lambda r:f'{int(r.snr_requested_db//5)*5}-{int(r.snr_requested_db//5)*5+5}'),
                      ('rt60_band',lambda r:f'{int((r.rt60_realized if r.rt60_realized is not None else r.rt60_requested)//.2)*.2:.1f}-{(int((r.rt60_realized if r.rt60_realized is not None else r.rt60_requested)//.2)+1)*.2:.1f}')]:
        for level in sorted(set(fn(r) for r in records)):
            ids=[i for i,r in enumerate(records) if fn(r)==level]
            for alpha in (.25,.5,.75):
                e=[x for seed in (42,43) for i in ids for x in blend_errors[(alpha,seed)][i]]
                blend_strat.append({'stratum':label,'level':level,'cnn_weight':alpha,'scene_count':len(ids),
                                    'source_count':len(e),'mean_error_m':float(np.mean(e)),
                                    'p90_error_m':float(np.percentile(e,90)),'within_0_5m':float(np.mean(np.asarray(e)<=.5))})

    out={'protocol':{'split':'validation','scenes':n,'source_count':sum(r.n_sources for r in records),'decoder':'3x3 local maxima, radius 0.60 m',
                     'model_seeds':[42,43],'bootstrap_seed':6102026,'test_split_used':False},
         'metrics_by_seed':{str(s):{m:metrics(seeds[s][m]) for m in ('srp','old','new')} for s in (42,43)},
         'equal_seed_mean_delta_ci':comparisons,'blend_results':blends,'blend_paired_bootstrap':blend_inference,'stratified':strat,
         'blend_stratified':blend_strat}
    (OUT/'paired_analysis.json').write_text(json.dumps(out,indent=2))
    with (OUT/'stratified_metrics.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=strat[0].keys());w.writeheader();w.writerows(strat)
    with (OUT/'blend_stratified_metrics.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=blend_strat[0].keys());w.writeheader();w.writerows(blend_strat)
    print('PAIRED',json.dumps({'metrics':out['metrics_by_seed'],'comparisons':comparisons,'blends':blends,'blend_paired_bootstrap':blend_inference},indent=2),flush=True)

if __name__=='__main__': main()
