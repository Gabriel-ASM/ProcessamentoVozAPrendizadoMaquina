"""Resource-bounded localization experiments; writes only under artifacts/round2."""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import replace
import json
from pathlib import Path
import random
import time

import numpy as np

from voice_localization.audio import read_multichannel_audio
from voice_localization.config import ProjectConfig, load_config
from voice_localization.dataset import load_map_sample, make_torch_dataset
from voice_localization.manifest import SceneRecord, read_jsonl, write_jsonl
from voice_localization.metrics import localization_metrics
from voice_localization.spatial import make_grid, srp_phat_map

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "round2"


def subset(records, count: int, seed: int):
    groups = defaultdict(list)
    for row in records:
        groups[(row.n_sources, row.n_microphones, row.geometry)].append(row)
    rng = random.Random(seed)
    chosen = []
    keys = sorted(groups)
    while len(chosen) < min(count, len(records)):
        added = False
        for key in keys:
            if groups[key] and len(chosen) < count:
                i = rng.randrange(len(groups[key]))
                chosen.append(groups[key].pop(i))
                added = True
        if not added:
            break
    return chosen


def peaks(scores, xs, ys, n, radius, local=True):
    from scipy.ndimage import maximum_filter
    arr = np.asarray(scores, np.float32)
    if local:
        mask = arr >= maximum_filter(arr, size=3, mode="nearest")
        candidates = np.flatnonzero(mask.ravel())
        order = candidates[np.argsort(arr.ravel()[candidates])[::-1]]
    else:
        order = np.argsort(arr.ravel())[::-1]
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([xx.ravel(), yy.ravel()])
    selected = []
    for i in order:
        p = pts[i]
        if all(np.linalg.norm(p-q) >= radius for q in selected):
            selected.append(p)
            if len(selected) == n:
                return np.asarray(selected)
    for i in np.argsort(arr.ravel())[::-1]:
        p = pts[i]
        if all(not np.array_equal(p, q) for q in selected):
            selected.append(p)
        if len(selected) == n:
            break
    return np.asarray(selected)


def eval_decoder(records, config, radii=(.2,.3,.45,.6), local=True, map_override=None):
    from voice_localization.metrics import pairwise_assignment_errors
    sums = {r: [] for r in radii}
    for index, rec in enumerate(records):
        feat, _, n, truth, xs, ys, _ = load_map_sample(rec)
        scores = feat[0] if map_override is None else map_override[index]
        for radius in radii:
            est = peaks(scores, xs, ys, n, radius, local)
            sums[radius].extend(pairwise_assignment_errors(truth, est).tolist())
    return {str(r): {"mean_error_m": float(np.mean(e)), "median_error_m": float(np.median(e)),
                     "p90_error_m": float(np.percentile(e,90)), "within_0_5m": float(np.mean(np.array(e)<=.5))}
            for r,e in sums.items()}


def make_acoustic_maps(records, cfg):
    rows=[]; map_root=OUT/"acoustic_maps"; map_root.mkdir(parents=True,exist_ok=True)
    for wi, exponent in ((512,1.0),(512,.75),(1024,1.0),(1024,.75),(2048,1.0),(2048,.75)):
        maps=[]; started=time.perf_counter()
        for i,rec in enumerate(records):
            audio=read_multichannel_audio(rec.audio_path,cfg.simulation.audio.sample_rate)
            mic=np.asarray(rec.microphone_positions_xy,np.float32)
            mic=np.column_stack((mic,np.full(len(mic),cfg.simulation.room.microphone_height,np.float32)))
            result,_=srp_phat_map(audio,mic,cfg.simulation.audio.sample_rate,tuple(cfg.simulation.room.dimensions),
                resolution_m=cfg.grid.resolution_m,speed_of_sound=cfg.simulation.room.speed_of_sound,
                window=wi,hop=wi//2,fft_size=wi*2,phat_exponent=exponent)
            maps.append(result)
        metrics=eval_decoder(records,cfg,radii=(.3,),local=True,map_override=maps)["0.3"]
        name=f"w{wi}_p{exponent:g}"
        np.savez_compressed(map_root/f"{name}.npz",maps=np.stack(maps))
        rows.append({"variant":name,"window":wi,"phat_exponent":exponent,"seconds":time.perf_counter()-started,**metrics})
        print("ACOUSTIC",rows[-1],flush=True)
    (OUT/"acoustic_results.json").write_text(json.dumps(rows,indent=2))
    return rows


def make_best_acoustic_full(records,cfg):
    maps=[]; errs=[]; started=time.perf_counter(); root=OUT/'acoustic_maps'; root.mkdir(parents=True,exist_ok=True)
    from voice_localization.metrics import pairwise_assignment_errors
    for i,rec in enumerate(records):
        audio=read_multichannel_audio(rec.audio_path,cfg.simulation.audio.sample_rate)
        mic=np.asarray(rec.microphone_positions_xy,np.float32)
        mic=np.column_stack((mic,np.full(len(mic),cfg.simulation.room.microphone_height,np.float32)))
        m,_=srp_phat_map(audio,mic,cfg.simulation.audio.sample_rate,tuple(cfg.simulation.room.dimensions),
            resolution_m=cfg.grid.resolution_m,speed_of_sound=cfg.simulation.room.speed_of_sound,
            window=1024,hop=512,fft_size=2048,phat_exponent=.75)
        maps.append(m)
        _,_,n,truth,xs,ys,_=load_map_sample(rec)
        errs.append(pairwise_assignment_errors(truth,peaks(m,xs,ys,n,.6,True)))
        if (i+1)%300==0: print('ACOUSTIC_FULL',i+1,'/',len(records),'seconds',round(time.perf_counter()-started),flush=True)
    np.savez_compressed(root/'full_validation_w1024_p0.75.npz',maps=np.stack(maps))
    allerr=np.concatenate(errs)
    result={'variant':'window1024_phat0.75','scene_count':len(records),'seconds':time.perf_counter()-started,
        'mean_error_m':float(allerr.mean()),'median_error_m':float(np.median(allerr)),
        'p90_error_m':float(np.percentile(allerr,90)),'within_0_5m':float(np.mean(allerr<=.5))}
    for label,fn in [('n_sources',lambda r:str(r.n_sources)),('n_microphones',lambda r:str(r.n_microphones)),('geometry',lambda r:r.geometry)]:
        result[label]={}
        for level in sorted(set(fn(r) for r in records)):
            e=np.concatenate([errs[i] for i,r in enumerate(records) if fn(r)==level])
            result[label][level]={'scene_count':sum(fn(r)==level for r in records),'mean_error_m':float(e.mean()),'p90_error_m':float(np.percentile(e,90)),'within_0_5m':float(np.mean(e<=.5))}
    (OUT/'acoustic_full_validation.json').write_text(json.dumps(result,indent=2)); print('ACOUSTIC_FULL_RESULT',json.dumps(result),flush=True)
    return result


def write_subsets(train, val):
    root=OUT/"manifests"; root.mkdir(parents=True,exist_ok=True)
    train_s=subset(train,6000,2026); val_s=subset(val,600,2027)
    write_jsonl(train_s,root/"train.jsonl"); write_jsonl(val_s,root/"validation.jsonl")
    return train_s,val_s


def train_trials(train_records,val_records):
    import torch
    from torch.utils.data import DataLoader
    from voice_localization.model import build_model
    from voice_localization.training import _collate
    torch.set_num_threads(2)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tds=make_torch_dataset(OUT/"manifests"/"train.jsonl")
    vds=make_torch_dataset(OUT/"manifests"/"validation.jsonl")
    tl=DataLoader(tds,batch_size=16,shuffle=True,collate_fn=_collate,num_workers=0)
    vl=DataLoader(vds,batch_size=32,shuffle=False,collate_fn=_collate,num_workers=0)
    configs=[("base_lr1e-3_pw250",(1,1,1,1),.001,250.0),
             ("base_lr3e-4_pw250",(1,1,1,1),.0003,250.0),
             ("base_lr1e-3_pw125",(1,1,1,1),.001,125.0),
             ("dilated_lr1e-3_pw250",(1,2,4,8),.001,250.0),
             ("dilated_lr3e-4_pw250",(1,2,4,8),.0003,250.0),
             ("dilated_lr1e-3_pw125",(1,2,4,8),.001,125.0)]
    allrows=[]
    for ci,(name,dil,lr,pw) in enumerate(configs):
        torch.manual_seed(3100+ci); np.random.seed(3100+ci)
        model=build_model(dil).to(device); opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
        lossfn=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw],device=device))
        best=1e9; hist=[]; started=time.perf_counter()
        for ep in range(1,6):
            model.train(); tloss=[]
            for b in tl:
                x=b['input'].to(device); y=b['target'].to(device); opt.zero_grad(set_to_none=True)
                loss=lossfn(model(x),y); loss.backward(); opt.step(); tloss.append(float(loss.detach()))
            model.eval(); errs=[]
            from voice_localization.metrics import pairwise_assignment_errors
            with torch.no_grad():
                for b in vl:
                    prob=model(b['input'].to(device)).sigmoid().cpu().numpy()[:,0]
                    for p,t,n in zip(prob,b['positions'],b['n_sources']):
                        errs.extend(pairwise_assignment_errors(t,peaks(p,b['xs'],b['ys'],int(n),.3,True)).tolist())
            mean=float(np.mean(errs)); hist.append({"epoch":ep,"train_bce":float(np.mean(tloss)),"val_mean_error_m":mean})
            print("TRIAL",name,"epoch",ep,"val_m",round(mean,4),"elapsed",round(time.perf_counter()-started),flush=True)
            if mean<best:
                best=mean; folder=OUT/"models"/name; folder.mkdir(parents=True,exist_ok=True)
                torch.save({"model":model.state_dict(),"dilations":dil,"lr":lr,"pos_weight":pw,"epoch":ep},folder/"best.pt")
        row={"name":name,"dilations":dil,"learning_rate":lr,"positive_weight":pw,"best_val_mean_error_m":best,
             "seconds":time.perf_counter()-started,"history":hist}
        allrows.append(row); (OUT/"trial_results.json").write_text(json.dumps(allrows,indent=2))
    return allrows


def confirm_full():
    import hashlib
    import torch
    from torch.utils.data import DataLoader
    from voice_localization.model import build_model
    from voice_localization.training import _collate
    from voice_localization.metrics import pairwise_assignment_errors
    torch.set_num_threads(2)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_manifest=Path("data/full/manifests/scenes_train.jsonl")
    val_manifest=Path("data/full/manifests/scenes_validation.jsonl")
    train_records=list(read_jsonl(train_manifest,SceneRecord)); val_records=list(read_jsonl(val_manifest,SceneRecord))
    trainset=make_torch_dataset(train_manifest); valset=make_torch_dataset(val_manifest)
    tl=DataLoader(trainset,batch_size=16,shuffle=True,collate_fn=_collate,num_workers=0)
    vl=DataLoader(valset,batch_size=32,shuffle=False,collate_fn=_collate,num_workers=0)

    def evaluate(model):
        model.eval(); errors=[]
        with torch.no_grad():
            for b in vl:
                probs=model(b["input"].to(device)).sigmoid().cpu().numpy()[:,0]
                for p,t,n in zip(probs,b["positions"],b["n_sources"]):
                    errors.extend(pairwise_assignment_errors(t,peaks(p,b["xs"],b["ys"],int(n),.6,True)).tolist())
        e=np.asarray(errors)
        return {"mean_error_m":float(e.mean()),"median_error_m":float(np.median(e)),"p90_error_m":float(np.percentile(e,90)),"within_0_5m":float(np.mean(e<=.5)),"source_count":int(e.size)}

    baselines={}
    for seed in (42,43):
        path=ROOT/f"artifacts/cnn_mixture_seed{seed}/best.pt"
        if not path.exists(): path=ROOT/f"artifacts/cnn_mixture_seed{seed}/model.pt"
        state=torch.load(path,map_location=device,weights_only=False)
        model=build_model().to(device); model.load_state_dict(state.get("model",state))
        baselines[str(seed)]=evaluate(model)
        print("BASELINE",seed,baselines[str(seed)],flush=True)

    results=[]
    for seed in (42,43):
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        model=build_model((1,2,4,8)).to(device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="min",factor=.5,patience=3)
        criterion=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([125.5],device=device))
        folder=OUT/"confirm"/f"seed{seed}"; folder.mkdir(parents=True,exist_ok=True)
        best=float("inf"); stale=0; rows=[]; started=time.perf_counter()
        for epoch in range(1,51):
            ep_started=time.perf_counter(); model.train(); losses=[]
            for b in tl:
                x=b["input"].to(device); y=b["target"].to(device)
                optimizer.zero_grad(set_to_none=True); loss=criterion(model(x),y); loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
            val=evaluate(model); scheduler.step(val["mean_error_m"])
            row={"epoch":epoch,"train_bce":float(np.mean(losses)),"validation":val,"seconds":time.perf_counter()-ep_started,"learning_rate":optimizer.param_groups[0]["lr"]}
            rows.append(row)
            state={"model":model.state_dict(),"optimizer":optimizer.state_dict(),"epoch":epoch,"history":rows,"seed":seed,"dilations":[1,2,4,8],"positive_weight":125.5}
            torch.save(state,folder/"last.pt")
            if val["mean_error_m"]<best:
                best=val["mean_error_m"]; stale=0; torch.save(state,folder/"best.pt")
            else: stale+=1
            (folder/"history.json").write_text(json.dumps(rows,indent=2))
            print("CONFIRM",seed,"epoch",epoch,"mean",round(val["mean_error_m"],4),"P90",round(val["p90_error_m"],4),"elapsed",round(time.perf_counter()-started),flush=True)
            if stale>=8 or time.perf_counter()-started>=5400: break
        state=torch.load(folder/"best.pt",map_location=device,weights_only=False)
        model.load_state_dict(state["model"]); metric=evaluate(model)
        digest=hashlib.sha256((folder/"best.pt").read_bytes()).hexdigest()
        results.append({"seed":seed,"best_epoch":state["epoch"],"validation":metric,"baseline":baselines[str(seed)],
                        "delta_mean_error_m":metric["mean_error_m"]-baselines[str(seed)]["mean_error_m"],"checkpoint_sha256":digest,
                        "seconds":time.perf_counter()-started,"history":rows})
        (OUT/"confirm_results.json").write_text(json.dumps(results,indent=2))
    return results


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--phase",choices=["decoder","acoustic","acoustic-full","train","confirm","all"],default="all")
    parser.add_argument("--config",type=Path,default=ROOT/"configs"/"full.yaml"); args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True); cfg=load_config(args.config)
    train=list(read_jsonl(Path(cfg.dataset.manifest_root)/"scenes_train.jsonl",SceneRecord))
    val=list(read_jsonl(Path(cfg.dataset.manifest_root)/"scenes_validation.jsonl",SceneRecord))
    train_s,val_s=write_subsets(train,val)
    if args.phase in ("decoder","all"):
        result={"current_global":eval_decoder(val,cfg,radii=(.2,.3,.45,.6),local=False),
                "local_peaks":eval_decoder(val,cfg,radii=(.2,.3,.45,.6),local=True)}
        (OUT/"decoder_results.json").write_text(json.dumps(result,indent=2)); print("DECODER",json.dumps(result),flush=True)
    if args.phase in ("acoustic","all"): make_acoustic_maps(val_s[:600],cfg)
    if args.phase in ("train","all"): train_trials(train_s,val_s)
    if args.phase=="confirm": confirm_full()
    if args.phase=="acoustic-full": make_best_acoustic_full(val,cfg)

if __name__=="__main__": main()
