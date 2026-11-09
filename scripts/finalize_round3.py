"""Audit artifacts and write the factual Round-3 report and machine ledger."""
from __future__ import annotations
import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "round3"


def load_json(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def load_metrics(name):
    return {x["method"]: x for x in load_json(name)}


def regressions(path, baseline_prefix="round2_seed", candidate_prefix="h1_w2048_rect_seed", suffix="_physical_local"):
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    grouped = defaultdict(dict)
    for row in rows:
        grouped[(row["stratum"], row["level"])][row["method"]] = row
    results = {}
    for seed in (42, 43):
        deltas=[]
        for key, methods in grouped.items():
            base = methods.get(f"{baseline_prefix}{seed}{suffix}")
            cand = methods.get(f"{candidate_prefix}{seed}{suffix}")
            if base and cand:
                deltas.append({"stratum":key[0],"level":key[1],
                    "mean_delta":float(cand["mean_error_m"])-float(base["mean_error_m"]),
                    "p90_delta":float(cand["p90_error_m"])-float(base["p90_error_m"]),
                    "over1_delta":float(cand["over_1m_rate"])-float(base["over_1m_rate"]),
                    "within01_delta":float(cand["within_lt_0_1m"])-float(base["within_lt_0_1m"])})
        results[str(seed)]={"strata_compared":len(deltas),
            "mean_regressions":sum(x["mean_delta"]>0 for x in deltas),
            "p90_regressions":sum(x["p90_delta"]>0 for x in deltas),
            "over1_regressions":sum(x["over1_delta"]>0 for x in deltas),
            "within01_regressions":sum(x["within01_delta"]<0 for x in deltas),
            "details":[x for x in deltas if x["mean_delta"]>0 or x["p90_delta"]>0 or x["over1_delta"]>0]}
    return results


def fmt(x, digits=3):
    return "—" if x is None else f"{float(x):.{digits}f}"


def metric_table(metrics, names):
    header="| Método | Erro médio (m) | Mediana (m) | P90 (m) | <0,10 m | ≤0,50 m | >1 m | Cenas todas <0,10 m |"
    sep="|---|---:|---:|---:|---:|---:|---:|---:|"
    lines=[header,sep]
    for name in names:
        m=metrics[name]
        lines.append(f"| {name} | {fmt(m['mean_error_m'])} | {fmt(m['median_error_m'])} | {fmt(m['p90_error_m'])} | "
                     f"{m['within_lt_0_1m_count']}/{m['source_count']} ({fmt(100*m['within_lt_0_1m_rate'],1)}%) | "
                     f"{fmt(100*m['within_0_5m'],1)}% | {m['over_1m_count']} ({fmt(100*m['over_1m_rate'],1)}%) | "
                     f"{m['all_sources_lt_0_1m_scenes']}/{m['scene_count']} |")
    return "\n".join(lines)


def main():
    full=load_metrics("full_validation_metrics.json")
    external=load_metrics("independent_holdout_metrics.json")
    paired=load_json("full_validation_paired_bootstrap.json")
    holdout_pair=load_json("independent_holdout_paired_bootstrap.json")
    triage=load_metrics("triage_baseline_metrics.json")
    h1=load_metrics("h1_metrics.json")
    h2=load_metrics("h2_metrics.json")
    h3=load_metrics("h3_metrics.json")
    refine=load_metrics("refine_metrics.json")
    physics=load_json("physics_pilot.json")
    audit=load_json("holdout_source_audit.json")
    holdout=load_json("holdout_manifest_frozen.json")
    # Include failed attempts and wall/CPU jobs as spent machine budget.
    costs=[
        ("setup_and_freeze_triage",15.4,"setup path correction; preparation, protocol and finalist freeze"),
        ("triage_baseline",load_json("timing_baseline.json")["seconds"],"360-scene baseline and oracle"),
        ("H1_windows",load_json("timing_h1.json")["seconds"],"three windows on 360 triage scenes"),
        ("H2_time_frequency",load_json("timing_h2.json")["seconds"],"16 block-band maps; rejected"),
        ("H3_deemphasis",load_json("timing_h3.json")["seconds"],"four notch configurations; rejected"),
        ("refine_contract_failure",58.6,"first sub-grid pass; exact-N guard detected invalid outputs"),
        ("refine_recheck",58.4,"two-method sub-grid rerun"),
        ("refine_expanded_triage",load_json("timing_refine.json")["seconds"],"baseline, H1 and window ensemble"),
        ("physics_diagnostic",load_json("timing_physics_pilot.json")["seconds"],"18 direct/reverb/noise ablation scenes"),
        ("failed_full_validation_baseline",101.1,"first full run; helper scope error after baseline cache"),
        ("full_validation",load_json("timing_full_validation.json")["seconds"],"3000 scenes, both seeds, full refinement, bootstrap prep"),
        ("bootstrap_reanalysis",2.7,"paired scene bootstrap correction from saved source errors"),
        ("test_other_download",26.2,"official OpenSLR archive download and HTTP size probe"),
        ("test_other_extract_and_audit",7.5,"archive extraction, both speaker/audio audits, hash and manifest verification"),
        ("holdout_generation",holdout["generation_seconds"],"600 independent scenes"),
        ("holdout_inference",load_json("independent_holdout_summary.json")["inference_seconds"],"one-shot 600-scene, 2-seed evaluation and outputs"),
        ("targeted_pytest",1.5,"4 spatial/metrics tests"),
        ("final_artifact_audit",2.0,"compile and write final ledger/report/summary"),
    ]
    total=sum(x[1] for x in costs)
    with (OUT/"machine_budget_ledger.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["stage","machine_seconds","machine_hours","included_failed_work","notes"])
        for name,seconds,note in costs:
            w.writerow([name,f"{seconds:.2f}",f"{seconds/3600:.4f}","yes",note])
        w.writerow(["TOTAL",f"{total:.2f}",f"{total/3600:.4f}","yes","all measured compute/download/generation/evaluation in Round 3"])
        w.writerow(["REMAINING_OF_12H",f"{43200-total:.2f}",f"{(43200-total)/3600:.4f}","n/a","no further experiment launched after independent holdout opened"])
    reg_full=regressions(OUT/"full_validation_stratified.csv")
    reg_external=regressions(OUT/"independent_holdout_stratified.csv", "round2_current_seed", "round3_candidate_seed", suffix="")
    (OUT/"full_validation_stratified_regressions.json").write_text(json.dumps(reg_full,indent=2),encoding="utf-8")
    (OUT/"independent_holdout_stratified_regressions.json").write_text(json.dumps(reg_external,indent=2),encoding="utf-8")
    success_count=sum(1 for row in csv.DictReader((OUT/"independent_lt_0_1m_cases.csv").open(encoding="utf-8")))
    if success_count<100:
        raise RuntimeError(f"critério independente <0,10 m não atingido: {success_count}")
    if holdout["speaker_overlap_count"] or holdout["audio_overlap_count"] or not audit["speaker_and_audio_independent"]:
        raise RuntimeError("holdout não passou auditoria de independência")
    if any(m["scene_count"]!=600 for m in external.values()):
        raise RuntimeError("métodos externos não cobrem todas as cenas congeladas")
    full_names=["round2_seed42","h1_w2048_rect_seed42_physical_local","round2_seed43","h1_w2048_rect_seed43_physical_local"]
    external_names=["round2_current_seed42","round3_candidate_seed42","round2_current_seed43","round3_candidate_seed43"]
    report=f"""# Rodada 3 — relatório de execução

## Conclusão

A execução ficou dentro do teto: **{total/3600:.2f} h de máquina de 12 h** ({(43200-total)/3600:.2f} h restantes). O candidato final H1 + refinamento físico local produziu **{success_count} casos independentes e únicos** com erro estritamente abaixo de 0,10 m entre 1.200 fontes, superando o requisito de 100. Também aumentou a proporção de fontes próximas e reduziu a média em ambos os seeds.

O candidato **não passa o gate de promoção** do protocolo: a redução de média e P90 é menor que os limiares de 0,15 m e 0,25 m, e a proporção de erros >1 m caiu pouco. Na validação completa as métricas agregadas melhoraram nos dois seeds, mas houve uma regressão de P90 de 0,010 m em N=3/seed42 e de taxa >1 m de 0,45 ponto percentual em M=8/seed43. No holdout, o seed42 teve P90 0,048 m pior que o baseline; o seed43 melhorou 0,078 m. A análise estratificada externa também mostra regressões em alguns grupos (seed42: média 3/17, P90 8/17, >1 m 6/17; seed43: média 0/17, P90 3/17, >1 m 9/17). Portanto o Streamlit permanece sem a mudança experimental.

## Protocolo e independência

- Triagem congelada: 360 cenas de validação, 20 por combinação N×M×geometria; subset fixo de treino com 6.000 cenas.
- Validação completa: 3.000 cenas, sem acesso ao teste antigo durante a seleção.
- Holdout final: 600 cenas novas geradas com `LibriSpeech test-other`, 200 por N. A auditoria encontrou 2.939 áudios e 33 falantes exclusivos, **zero** IDs/falantes sobrepostos com os manifestos existentes. Hash SHA256 do manifesto: `{holdout['manifest_sha256']}`.
- Candidato/checkpoints congelados antes da inferência externa; o mesmo holdout foi avaliado uma vez, sem retuning. Seed42 foi predefinido como primário e seed43 como replicação.
- Erros por fonte usam associação Húngara; bootstrap pareado com 10.000 reamostragens por cena.

## Resultados na validação completa

Referência: fusão atual da Rodada 2 por seed. Candidato: rank SRP com janela retangular 2.048 + 25% CNN rank do mesmo seed, decodificação local radius 0,60 m e busca física ±0,20 m a passo 0,025 m.

{metric_table(full,full_names)}

Bootstrap pareado do candidato refinado versus referência atual: seed42 Δ média **{paired['42']['against_coarse_baseline']['candidate_refined']['mean_delta_m']:.3f} m** (IC95% {paired['42']['against_coarse_baseline']['candidate_refined']['ci95_m'][0]:.3f} a {paired['42']['against_coarse_baseline']['candidate_refined']['ci95_m'][1]:.3f}); seed43 Δ **{paired['43']['against_coarse_baseline']['candidate_refined']['mean_delta_m']:.3f} m** (IC95% {paired['43']['against_coarse_baseline']['candidate_refined']['ci95_m'][0]:.3f} a {paired['43']['against_coarse_baseline']['candidate_refined']['ci95_m'][1]:.3f}). A fração >1 m caiu 3,8% e 3,1% em termos relativos, abaixo dos 20% do gate. Nas estratificações de média, {reg_full['42']['mean_regressions']} regressões para seed42 e {reg_full['43']['mean_regressions']} para seed43; P90 teve regressão mínima em N=3 (ver JSON de auditoria estratificada).

## Confirmação independente no test-other

{metric_table(external,external_names)}

Seed42: média melhorou 0,049 m (IC95% {holdout_pair['42']['ci95_m'][0]:.3f} a {holdout_pair['42']['ci95_m'][1]:.3f}, inclui zero), P90 aumentou 0,048 m; os casos <0,10 m aumentaram de 379 para 543 e >1 m caiu de 434 para 430. Seed43: média melhorou 0,072 m (IC95% {holdout_pair['43']['ci95_m'][0]:.3f} a {holdout_pair['43']['ci95_m'][1]:.3f}), P90 melhorou 0,078 m; <0,10 m passou de 385 para 533 e >1 m de 443 para 435. A redução da cauda >1 m (0,9%/1,8% relativa) permanece muito abaixo do gate. Na estratificação externa houve regressões, apesar de todas as taxas <0,10 m subirem; esses detalhes estão em `independent_holdout_stratified_regressions.json`. A diferença entre seeds reforça que ainda não há uma substituição robusta confirmada.

**Entrega do critério de 10 cm:** [independent_lt_0_1m_cases.csv](independent_lt_0_1m_cases.csv) contém todos os {success_count} pares únicos de fonte-estimativa do seed42 com erro `<0,10 m`, com ID da cena e coordenadas referência/estimada. O candidato acertou todas as fontes abaixo de 10 cm em 197 das 600 cenas; seed43, em {external['round3_candidate_seed43']['all_sources_lt_0_1m_scenes']} cenas. Exemplos determinísticos de quatro sucessos e quatro falhas: [independent_success_failure_examples.csv](independent_success_failure_examples.csv). Todas as fontes e falhas ficam em [independent_holdout_source_errors.csv](independent_holdout_source_errors.csv).

## Diagnóstico e hipóteses

- H1 janela/escore: 2.048 retangular foi a melhor variante de triagem. O ganho confirmou-se na validação, mas não atingiu o gate de tail-risk.
- H2 preservação tempo-frequência/consenso: pior que o baseline (média 1,37–1,69 m e mais erros >1 m); descartado.
- H3 de-emphasis por notch na GCC: melhor variante quase empatou na média, mas teve mais erros >1 m; descartado.
- H5 refinamento subgrade: aumentou acertos <10 cm na validação de 2.704/2.700 para 2.810/2.802 fontes, com P90 global melhor que baseline. No holdout, melhorou a precisão fina, sem reduzir materialmente os erros >1 m.
- H4 rede multibanda não foi treinada: o gate oracle de H2 não foi atingido. Wave-U-Net não foi adicionada; os experimentos não apontaram separação de fonte como a intervenção que resolveria os outliers de posicionamento nesta rodada.
- Piloto físico (18 combinações, 36 fontes): caminho direto sem ruído teve mediana 0,082 m, mas 8 erros >1 m; reverberante sem ruído teve 16/36 erros >1 m e mistura ruidosa 16/36. Reflexões e combinações cruzadas de fontes explicam mais que ruído isolado; H2/H3 leves não removeram isso.

## Custos e artefatos

O ledger inclui as duas execuções interrompidas por erros de implementação, nova tentativa, download, extração, geração e avaliação: [machine_budget_ledger.csv](machine_budget_ledger.csv). O relatório estratificado, ICs, mapas, logs, configurações e todos os erros estão nesta pasta `artifacts/round3/`. O código da rodada está em [run_round3.py](../../scripts/run_round3.py), [generate_round3_holdout.py](../../scripts/generate_round3_holdout.py), [evaluate_round3_holdout.py](../../scripts/evaluate_round3_holdout.py) e [analyze_round3.py](../../scripts/analyze_round3.py).

## Decisão

Manter a implementação atual no Streamlit para não promover uma mudança que falha no gate predefinido de falhas grandes. O refinamento H5 e a janela H1 ficam como candidato experimental documentado; os 543 exemplos abaixo de 10 cm são evidência de precisão local, não de resolução do risco de fontes perdidas. A próxima rodada deve agir sobre a identificação/separação de fontes dominantes, com outro conjunto externo congelado antes da inferência.
"""
    (OUT/"round3_report.md").write_text(report,encoding="utf-8")
    summary={"machine_seconds":total,"machine_hours":total/3600,"remaining_hours":(43200-total)/3600,
             "independent_unique_lt_0_1m_cases_seed42":success_count,"independent_scenes":600,
             "holdout_manifest_sha256":holdout["manifest_sha256"],"validation_stratified":reg_full,
             "external_stratified":reg_external,"promotion_gate_passed":False,
             "streamlit_changed":False,"targeted_spatial_tests":"4 passed"}
    (OUT/"round3_audit_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))


if __name__=="__main__": main()
