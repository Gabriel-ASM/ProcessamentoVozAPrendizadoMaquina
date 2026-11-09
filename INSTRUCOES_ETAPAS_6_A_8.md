# Instruções para concluir as etapas 6–8

Este roteiro parte do piloto em `configs/wave_pilot.yaml`. O piloto usa fala sintética para verificar código e propagação acústica. Os resultados finais usam LibriSpeech e salas simuladas; não misture os números sintéticos aos resultados do experimento.

**Status em 2026-11-07:** etapa 6, treinos/validações das seeds 42, 43 e 44, gate, avaliação única do teste reservado e relatório final concluídos. O gate reprovou a Wave-U-Net como ramo auxiliar de localização (Wave+SRP: +0,1914 m na validação); o teste confirmou a piora (+0,1420 m; IC95% +0,1209 a +0,1637 m). O relatório está em `artifacts/final/report/final_report.md`. A demonstração responde somente no endereço local `http://127.0.0.1:8501`; Streamlit AppTest acionou os dois extremos do domínio, exibiu quatro imagens por caso e não registrou exceções. Configurações e resultados de demonstração estão em `artifacts/final/demo_extremes*.json` e as imagens em `artifacts/final/demo_extremes/`. Consulte [EXECUCAO_FINAL.md](EXECUCAO_FINAL.md) para métricas e detalhes. Não rode novamente a avaliação do teste.

O resultado atualizado do piloto está em [RESULTADOS_PILOTO_WAVE_U_NET.md](RESULTADOS_PILOTO_WAVE_U_NET.md).

## 6. Gerar e auditar a base completa

### Pré-requisitos

Instale as partições `train-clean-100`, `dev-clean` e `test-clean` do LibriSpeech em `data/full/LibriSpeech`. O corpus não é distribuído com o projeto. Não reutilize o mesmo locutor em partições distintas. A configuração `configs/full.yaml` separa os arquivos finais de `smoke`, `pilot` e `wave_pilot` e grava, para cada cena, a mistura com ruído e o alvo limpo reverberante necessário à Wave-U-Net.

Os WAVs float32 das duas versões somam cerca de **46 GB decimais** para 30 mil cenas, assumindo seis canais e 32 mil amostras por canal. Reserve espaço adicional para o corpus, mapas, manifestos e margem para arquivos temporários. Se não houver esse espaço, reduza a configuração explicitamente e registre o novo tamanho da base; não remova os alvos limpos sem abandonar também o experimento Wave-U-Net.

No Anaconda Prompt, na raiz do projeto, configure o ambiente e prepare os manifestos:

```powershell
conda activate voz-gpu
$env:PYTHONPATH = (Join-Path (Get-Location) "src")
python -m voice_localization.cli validate-config --config configs/full.yaml
python -m voice_localization.cli prepare-corpus --root data/full/LibriSpeech --output data/full/manifests
```

Confira os totais e a separação de locutores nos manifestos de áudio antes de simular. Gere cada partição a partir de seu manifesto correspondente. Não use `--synthetic` nesta etapa.

O RT60 realizado depende da versão do Pyroomacoustics. A configuração deste projeto foi calibrada com a versão 0.10.1. Confirme a sua versão antes de gerar a base completa:

```powershell
python -c "import pyroomacoustics as pra; print(pra.__version__)"
```

Se a versão for diferente, faça primeiro uma calibração somente com cenas de treino sintéticas, ajuste `rt60_calibration_slope` e `rt60_calibration_intercept` em `configs/wave_pilot.yaml` e `configs/full.yaml`, regenere o piloto de treino e confirme que os RT60 realizados cobrem a faixa desejada. O script ajusta `realizado = slope × rt60_sabine_input + intercept`; não use validação ou teste para estimar esses coeficientes:

```powershell
python -m voice_localization.cli generate-scenes --config configs/wave_pilot.yaml --split train --synthetic
python scripts/calibrate_rt60.py --manifest data/wave_pilot/manifests/scenes_train.jsonl --output artifacts/wave_pilot/rt60_fit_train_only.json
```

Após gerar as três partições completas, audite os WAVs, os pares limpo/ruidoso, o SNR medido, o RT60 e a separação entre splits. A auditoria lê os áudios sequencialmente:

```powershell
python scripts/audit_scenes.py --config configs/full.yaml --output artifacts/full/scene_audio_audit.json
```

```powershell
python -m voice_localization.cli generate-scenes --config configs/full.yaml --split train --workers 2
python -m voice_localization.cli generate-scenes --config configs/full.yaml --split validation --workers 2
python -m voice_localization.cli generate-scenes --config configs/full.yaml --split test --workers 2
```

Os totais esperados são 24.000 cenas de treino, 3.000 de validação e 3.000 de teste. A geração pode ser retomada pelo mesmo comando. Verifique contagem, WAVs, valores finitos, SNR e RT60 realizados, arquivos limpos e mistura, falantes, receitas duplicadas e distribuição de N, M, geometria, SNR e RT60. Investigue falhas antes de extrair os mapas.

Extraia a representação da mistura ruidosa, ajuste a normalização somente com treino e audite o cache:

```powershell
python -m voice_localization.cli extract-maps --config configs/full.yaml --variant mixture --audio-variant mixture
python -m voice_localization.cli validate-data --config configs/full.yaml
```

Conserve uma cópia imutável dos manifestos, das configurações resolvidas e do relatório de auditoria. Registre o espaço e o tempo medidos por etapa. O conjunto de teste fica reservado para a avaliação final; sua integridade pode ser auditada sem usar seus erros para escolher modelos ou parâmetros.

**Saída da etapa:** 30.000 cenas completas e auditadas, manifestos sem vazamento entre partições, mapas SRP-PHAT válidos e configuração, custos e identificador da base preservados.

Auditoria medida nesta execução: `artifacts/full/resource_audit.json` registra ocupação por pasta/split, histogramas de N/M/geometria/SNR/RT60 por partição e tempos de extração dos mapas. Os histogramas cobrem todas as 30.000 cenas, sem valores ausentes nas faixas avaliadas. A geração completa não deixou duração de parede no manifesto; a janela de escrita dos arquivos foi preservada como metadado de proveniência, sem ser tratada como tempo exato da execução.

## 7. Treinar e comparar os modelos

Treine primeiro o CNN com mapas da mistura original. Treine a Wave-U-Net com os pares multicanais e o alvo limpo reverberante. Em ambos os casos, selecione checkpoints pela validação.

```powershell
python -m voice_localization.cli train --config configs/full.yaml --variant mixture --seed 42 --run-name cnn_mixture_seed42
python -m voice_localization.cli train-enhancer --config configs/full.yaml --seed 42 --run-name wave_seed42
```

Use o hash do checkpoint no nome da variante para manter cada mapa associado ao modelo que o produziu:

```powershell
$waveCheckpoint = "artifacts/wave_seed42/wave_unet/best.pt"
$waveHash = (Get-FileHash $waveCheckpoint -Algorithm SHA256).Hash.ToLower()
$waveVariant = "wave_$($waveHash.Substring(0, 12))"
```

A Wave-U-Net recebe até oito canais de áudio, as máscaras de canais existentes e as coordenadas dos microfones normalizadas pela sala. Ela prevê uma correção residual e mantém a quantidade de canais. Para criar as misturas aprimoradas de treino e validação:

```powershell
python -m voice_localization.cli enhance-audio --config configs/full.yaml --checkpoint $waveCheckpoint --split train
python -m voice_localization.cli enhance-audio --config configs/full.yaml --checkpoint $waveCheckpoint --split validation
```

Extraia os mapas Wave dos dois splits. Ao extrair o treino, o comando ajusta a normalização na partição de treino; a extração da validação reutiliza esses parâmetros.

```powershell
python -m voice_localization.cli extract-maps --config configs/full.yaml --split train --variant $waveVariant --audio-variant enhanced
python -m voice_localization.cli extract-maps --config configs/full.yaml --split validation --variant $waveVariant --audio-variant enhanced
```

Treine um segundo CNN sobre esses mapas. Esse modelo precisa ser treinado com dados transformados pela própria Wave-U-Net para que a avaliação combinada reflita o fluxo proposto.

```powershell
python -m voice_localization.cli train --config configs/full.yaml --variant $waveVariant --seed 42 --run-name cnn_wave_seed42
```

Durante o desenvolvimento, compare também as quatro combinações na validação: SRP-PHAT, CNN sobre a mistura, Wave-U-Net seguida de SRP-PHAT, e Wave-U-Net seguida do CNN treinado sobre mapas Wave. Use as mesmas cenas, posições, N conhecido e decodificador de picos. Meça erro por fonte, média, mediana, P90, proporção até 0,50 m e tempo por cena. Estratifique por N, M, geometria, SNR e RT60. Para a Wave-U-Net, registre também SI-SDR ou erro de reconstrução e o pico de memória GPU.

Ajuste hiperparâmetros e escolha modelos com validação. Repita o CNN e a Wave-U-Net com seeds 42, 43 e 44, usando diretórios separados. Para cada checkpoint Wave, calcule um novo `$waveVariant` pelo hash e repita aprimoramento, extração e treino do CNN associado. Não avalie o teste a cada alteração: congele arquiteturas, hiperparâmetros, normalizadores, checkpoints e regras de decodificação antes da rodada final. Os comandos abaixo representam a rodada final da seed 42, após congelar todas as escolhas:

```powershell
python -m voice_localization.cli enhance-audio --config configs/full.yaml --checkpoint $waveCheckpoint --split test
python -m voice_localization.cli extract-maps --config configs/full.yaml --split test --variant $waveVariant --audio-variant enhanced
python -m voice_localization.cli evaluate-baseline --config configs/full.yaml --split test --variant mixture --output artifacts/final/srp_test.json
python -m voice_localization.cli evaluate-baseline --config configs/full.yaml --split test --variant $waveVariant --output artifacts/final/wave_srp_test.json
python -m voice_localization.cli evaluate --config configs/full.yaml --checkpoint artifacts/cnn_mixture_seed42/best.pt --split test --variant mixture --output artifacts/final/cnn_mixture_test.json
python -m voice_localization.cli evaluate --config configs/full.yaml --checkpoint artifacts/cnn_wave_seed42/best.pt --split test --variant $waveVariant --output artifacts/final/cnn_wave_test.json
```

Execute a rodada final uma vez por seed congelada e preserve os resultados brutos por cena. Para intervalos de confiança, reamostre cenas inteiras (e todos os erros de fonte associados a cada cena), evitando tratar fontes da mesma cena como observações independentes. Relate diferenças pareadas entre métodos e variação entre seeds. Um resultado negativo ou condicionado a SNR, reverberação ou geometria também é uma conclusão válida.

Calcule os intervalos pareados sobre os JSONs brutos gerados acima. Repita para cada comparação abaixo e cada seed congelada:

```powershell
python scripts/compare_localization.py --reference artifacts/final/srp_test.json --candidate artifacts/final/cnn_mixture_test.json --output artifacts/final/paired_srp_cnn_test.json --samples 10000 --seed 42
python scripts/compare_localization.py --reference artifacts/final/srp_test.json --candidate artifacts/final/wave_srp_test.json --output artifacts/final/paired_srp_wave_test.json --samples 10000 --seed 42
python scripts/compare_localization.py --reference artifacts/final/srp_test.json --candidate artifacts/final/cnn_wave_test.json --output artifacts/final/paired_srp_wavecnn_test.json --samples 10000 --seed 42
python scripts/compare_localization.py --reference artifacts/final/cnn_mixture_test.json --candidate artifacts/final/cnn_wave_test.json --output artifacts/final/paired_cnn_wavecnn_test.json --samples 10000 --seed 42
```

**Saída da etapa:** checkpoints selecionados por validação, comparação pareada e estratificada no teste reservado, incerteza estatística, tempo e memória, e configurações/checkpoints que reproduzem cada linha da comparação.

## 8. Produzir análise, relatório e demonstração

Congele os arquivos da etapa 7 e gere uma tabela principal com contagem de cenas/fontes, média, mediana, P90, proporção até 0,50 m, intervalos de confiança, tempo por cena e seed. Inclua tabelas por N, M, geometria, faixa de SNR e faixa de RT60. Separe o custo da geração dos mapas, da inferência Wave-U-Net, da inferência CNN e do pipeline combinado.

Gere ainda curvas de treino/validação, gráficos de desempenho por condição acústica, distribuição de erros e exemplos espaciais escolhidos por regra explícita. Em cada figura, mostre a geometria dos microfones, posições verdadeiras, picos do SRP e posições estimadas, com eixos em metros. Inclua sucessos e falhas sem selecionar exemplos com base apenas em aparência.

Na interpretação, limite as conclusões ao domínio simulado: sala de 8 × 6 × 3 m, fontes e microfones a 1,5 m, fontes estáticas, sincronização ideal, N entre 1 e 3, M entre 4 e 8, SNR de 10–30 dB e RT60 solicitado de 0,2–0,8 s. A simulação com LibriSpeech não demonstra generalização para gravações reais ou outras salas.

Finalize a demonstração local em Streamlit descrita em `PLANO.md`. Ela deve usar os checkpoints congelados, permitir selecionar somente N, M, geometria, SNR e RT60 cobertos pelo experimento e apresentar as quatro combinações comparáveis. Identifique a amostra como simulação e exiba posição real, estimativas, métricas e tempo. Confira exemplos nos extremos do domínio e documente a configuração usada para gerar cada um.

**Pacote final:** código e configurações, IDs/manifestos e relatório de auditoria da base, histórico e checkpoints, resultados brutos e sumarizados, gráficos, demonstração e relatório acadêmico com limitações e instruções de reprodução.

### Verificação da demonstração nos extremos

Os dois casos abaixo foram gerados pelo botão da interface com os checkpoints congelados da seed 42. Cada execução mostrou os quatro métodos, quatro gráficos, posições, métricas e parâmetros realizados; o Streamlit AppTest não registrou exceções. São exemplos funcionais da demonstração, não uma avaliação adicional do teste nem evidência inferencial de desempenho.

| Caso | Parâmetros solicitados | Realizado | Erro médio por fonte (SRP / CNN mistura / Wave+SRP / Wave+CNN) |
|---|---|---|---|
| Limite inferior | N=1, M=4, perímetro, SNR=10 dB, RT60=0,20 s | SNR=9,97 dB; RT60=0,200 s | 0,111 / 0,190 / 0,111 / 0,111 m |
| Limite superior | N=3, M=8, círculo, SNR=30 dB, RT60=0,80 s | SNR=30,00 dB; RT60=0,792 s | 0,754 / 1,296 / 1,912 / 1,306 m |

As posições, seeds de cena, hashes de checkpoint, métricas completas e imagens por método estão em `artifacts/final/demo_extremes.json`, `artifacts/final/demo_extremes_ui.json` e `artifacts/final/demo_extremes/`.

## Referências acadêmicas para a linha Wave-U-Net

- Stoller, Ewert e Dixon, [*Wave-U-Net: A Multi-Scale Neural Network for End-to-End Audio Source Separation*](https://ismir2018.ircam.fr/doc/pdfs/205_Paper.pdf), ISMIR 2018. A proposta original opera diretamente no waveform com caminhos multiescala e conexões entre escalas.
- Ho et al., [*A Cross-Channel Attention-Based Wave-U-Net for Multi-Channel Speech Enhancement*](https://www.isca-archive.org/interspeech_2020/ho20_interspeech.html), Interspeech 2020. Esta extensão trata explicitamente as interações entre canais de microfone.

O código deste projeto é uma adaptação pequena e condicionada à geometria, inspirada nessas arquiteturas; não é uma reprodução fiel dos experimentos publicados. A métrica decisiva para este trabalho continua sendo erro de localização por fonte, além de SI-SDR, pois melhora de áudio não implica preservação dos atrasos espaciais.
