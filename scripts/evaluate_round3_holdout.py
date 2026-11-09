"""One-shot evaluation of the frozen Round-3 finalist on test-other."""
from __future__ import annotations
import csv
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from voice_localization.audio import read_multichannel_audio
from voice_localization.config import load_config
from voice_localization.manifest import SceneRecord, read_jsonl
from voice_localization.spatial import gcc_phat_pairs, srp_phat_map
from run_round3 import (OUT, DEVICE, errors_for_estimates, load_model, metric_summary,
                        paired_scene_bootstrap, peaks, physical_refine_from_corr, rank_map,
                        save_source_errors, sha256, stratified_metrics)


def verify_freezes(manifest_path):
    finalist = json.loads((OUT / "round3_finalist_frozen.json").read_text(encoding="utf-8"))
    holdout = json.loads((OUT / "holdout_manifest_frozen.json").read_text(encoding="utf-8"))
    if not finalist["frozen_before_independent_holdout_inference"] or not holdout["frozen_before_inference"]:
        raise RuntimeError("finalista/manifesto não estavam congelados antes da inferência")
    if sha256(manifest_path) != holdout["manifest_sha256"]:
        raise RuntimeError("hash do manifesto mudou após o congelamento")
    if int(holdout["scene_count"]) != 600:
        raise RuntimeError("holdout congelado não possui 600 cenas")
    if holdout["speaker_overlap_count"] or holdout["audio_overlap_count"]:
        raise RuntimeError("holdout contém sobreposição com áudio/falantes anteriores")
    for seed in (42,43):
        p = ROOT / "artifacts" / "round2" / "confirm" / f"seed{seed}" / "best.pt"
        if sha256(p) != finalist["checkpoint_sha256"][str(seed)]["sha256"]:
            raise RuntimeError(f"checkpoint seed{seed} mudou após congelamento")
    return holdout


def load_normalizer():
    path = ROOT / "data" / "full" / "maps" / "mixture" / "normalizer.npz"
    with np.load(path) as values:
        return float(values["mean"]), float(values["std"])


def scene_inference(rec, models, cfg, normalizer):
    audio = read_multichannel_audio(rec.audio_path, cfg.simulation.audio.sample_rate)
    mic_xy = np.asarray(rec.microphone_positions_xy, dtype=np.float32)
    mics = np.column_stack((mic_xy, np.full(len(mic_xy), cfg.simulation.room.microphone_height, dtype=np.float32)))
    raw512, grid = srp_phat_map(audio, mics, cfg.simulation.audio.sample_rate,
        tuple(cfg.simulation.room.dimensions), resolution_m=cfg.grid.resolution_m,
        speed_of_sound=cfg.simulation.room.speed_of_sound, window=512, hop=256, fft_size=1024)
    mean, std = normalizer
    model_input = ((raw512 - mean) / max(std,1e-8)).astype(np.float32)
    features = np.stack((model_input, np.full_like(model_input, rec.n_sources/3.0)), axis=0)
    probabilities = {}
    with torch.no_grad():
        tensor = torch.from_numpy(features[None]).to(DEVICE)
        for seed in (42,43):
            probabilities[seed] = models[seed](tensor).sigmoid()[0,0].cpu().numpy().astype(np.float32)
    raw2048, _ = srp_phat_map(audio, mics, cfg.simulation.audio.sample_rate,
        tuple(cfg.simulation.room.dimensions), resolution_m=cfg.grid.resolution_m,
        speed_of_sound=cfg.simulation.room.speed_of_sound, window=2048, hop=1024, fft_size=4096,
        phat_exponent=1.0, window_type="rectangular")
    corr512, pairs512 = gcc_phat_pairs(audio,512,256,1024,phat_exponent=1.0,window_type="hann")
    corr2048, pairs2048 = gcc_phat_pairs(audio,2048,1024,4096,phat_exponent=1.0,window_type="rectangular")
    outputs = {}
    for seed in (42,43):
        base_score = .75 * rank_map(raw512) + .25 * probabilities[seed]
        base = peaks(base_score, grid.xs, grid.ys, rec.n_sources, .60, True)
        base_refined = physical_refine_from_corr(rec,base,cfg,corr512,pairs512,512)
        candidate_score = .75 * rank_map(raw2048) + .25 * rank_map(probabilities[seed])
        candidate = peaks(candidate_score,grid.xs,grid.ys,rec.n_sources,.60,True)
        candidate_refined = physical_refine_from_corr(rec,candidate,cfg,corr2048,pairs2048,2048)
        outputs[f"round2_current_seed{seed}"] = base
        outputs[f"round2_plus_local_refinement_seed{seed}"] = base_refined
        outputs[f"round3_candidate_seed{seed}"] = candidate_refined
    return outputs, grid.xs, grid.ys


def grouped_scene_bootstrap(records, errors_by_method, baseline, candidate, repeats=10000, seed=20261109):
    rng = np.random.default_rng(seed)
    n = len(records)
    delta_sum = np.asarray([float(np.sum(a-b)) for a,b in zip(errors_by_method[candidate],errors_by_method[baseline])])
    count = np.asarray([len(x) for x in errors_by_method[baseline]],dtype=np.float64)
    point = float(delta_sum.sum()/count.sum())
    samples=[]
    for start in range(0,repeats,1000):
        size=min(1000,repeats-start)
        ids=rng.integers(0,n,size=(size,n))
        samples.extend((delta_sum[ids].sum(axis=1)/count[ids].sum(axis=1)).tolist())
    return {"baseline":baseline,"candidate":candidate,"mean_delta_m":point,
            "ci95_m":[float(np.quantile(samples,.025)),float(np.quantile(samples,.975))],
            "repeats":repeats,"unit":"scene","negative_favors_candidate":True}


def main():
    torch.set_num_threads(2)
    manifest = OUT / "manifests" / "holdout" / "scenes_test.jsonl"
    holdout = verify_freezes(manifest)
    records = list(read_jsonl(manifest, SceneRecord))
    if len(records) != 600:
        raise RuntimeError(f"esperadas 600 cenas congeladas, recebidas {len(records)}")
    cfg = load_config(ROOT / "configs" / "full.yaml")
    normalizer = load_normalizer()
    models = {s:load_model(s)[0] for s in (42,43)}
    estimates = {name:[] for name in (
        "round2_current_seed42","round2_plus_local_refinement_seed42","round3_candidate_seed42",
        "round2_current_seed43","round2_plus_local_refinement_seed43","round3_candidate_seed43")}
    started=time.perf_counter()
    for i,rec in enumerate(records):
        outputs,_,_=scene_inference(rec,models,cfg,normalizer)
        for key,value in outputs.items(): estimates[key].append(value)
        if (i+1)%50==0 or i+1==len(records):
            print("HOLDOUT_INFERENCE_PROGRESS",i+1,"/",len(records),flush=True)
    all_errors={name:errors_for_estimates(records,value) for name,value in estimates.items()}
    metrics=[{"method":name,**metric_summary(errors,records)} for name,errors in all_errors.items()]
    (OUT/"independent_holdout_metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
    save_source_errors(OUT/"independent_holdout_source_errors.csv",records,estimates)
    stratified_metrics(records,all_errors,OUT/"independent_holdout_stratified.csv")
    paired={}
    for seed in (42,43):
        paired[str(seed)]=grouped_scene_bootstrap(records,all_errors,f"round2_current_seed{seed}",
            f"round3_candidate_seed{seed}",seed=20261109+seed)
        paired[str(seed)+"_vs_refined_baseline"]=grouped_scene_bootstrap(records,all_errors,
            f"round2_plus_local_refinement_seed{seed}",f"round3_candidate_seed{seed}",seed=20261200+seed)
    (OUT/"independent_holdout_paired_bootstrap.json").write_text(json.dumps(paired,indent=2),encoding="utf-8")
    with (OUT/"independent_lt_0_1m_cases.csv").open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=["method","scene_id","source_index","reference_x_m","reference_y_m","estimate_x_m","estimate_y_m","error_m"])
        writer.writeheader()
        with (OUT/"independent_holdout_source_errors.csv").open(newline="",encoding="utf-8") as source:
            for row in csv.DictReader(source):
                if row["method"]=="round3_candidate_seed42" and float(row["error_m"]) < .10:
                    writer.writerow(row)
    with (OUT/"independent_success_failure_examples.csv").open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=["selection","method","scene_id","source_index","reference_x_m","reference_y_m","estimate_x_m","estimate_y_m","error_m"])
        writer.writeheader()
        with (OUT/"independent_holdout_source_errors.csv").open(newline="",encoding="utf-8") as source:
            rows=[r for r in csv.DictReader(source) if r["method"]=="round3_candidate_seed42"]
        rows.sort(key=lambda r:(float(r["error_m"]),r["scene_id"],int(r["source_index"])))
        for label,items in (("success",rows[:4]),("failure",rows[-4:])):
            for row in items:
                writer.writerow({"selection":label,**row})
    summary={"holdout_manifest_sha256":holdout["manifest_sha256"],"scene_count":len(records),
             "generation_seconds":holdout["generation_seconds"],"inference_seconds":time.perf_counter()-started,
             "metrics":metrics,"bootstrap":paired}
    (OUT/"independent_holdout_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    primary=next(x for x in metrics if x["method"]=="round3_candidate_seed42")
    print("INDEPENDENT_HOLDOUT_RESULT",json.dumps({"seed42_candidate":primary,
        "seed43_candidate":next(x for x in metrics if x["method"]=="round3_candidate_seed43"),
        "unique_lt_0_1m_cases_seed42":primary["within_lt_0_1m_count"],
        "inference_seconds":time.perf_counter()-started},ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
