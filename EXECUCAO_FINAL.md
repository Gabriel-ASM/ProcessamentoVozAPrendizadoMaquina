# Execução completa das etapas 6–8

Este roteiro executa a base LibriSpeech, os treinos com seeds 42/43/44, a avaliação final e a demonstração Streamlit. Não use o conjunto de teste para ajustar modelos ou parâmetros.

## Status desta execução — 2026-11-07

- **Etapa 6 concluída:** base congelada com 24.000 cenas de treino, 3.000 de validação e 3.000 de teste. O teste foi acessado uma única vez após o freeze dos modelos. IDs dos splits e hash de configuração estão em `artifacts/full/frozen_dataset/freeze.json`.
- **Limitação RT60:** 274/30.000 cenas (0,91%) ficaram fora de [0,2; 0,8] s no RT60 realizado. Faixas medidas: treino 0,191–0,835 s; validação 0,196–0,812 s; teste 0,191–0,824 s. `build_final_report.py` registra a contagem e as faixas por split automaticamente.
- **Etapa 7:** treino e validação das seeds 42, 43 e 44 concluídos; resultados pareados e hashes congelados em `artifacts/final/validation/seedN/`.
- **Gate de validação:** reprovado para promover Wave-U-Net como ramo auxiliar. Wave+SRP teve delta médio de **+0,1914 m** contra SRP-PHAT, com peso igual entre as três seeds. CNN-Wave também piorou frente à CNN sobre a mistura em todas as seeds.
- **Teste reservado:** a avaliação de uso único terminou para as três seeds e quatro métodos; os JSONs brutos e registros de execução estão em `artifacts/final/seedN/`.
- **Relatório:** gerado e auditado em `artifacts/final/report/final_report.md`, com CSVs, gráficos e 8 exemplos espaciais.
- **Segunda rodada de localização:** triagem de seis CNNs, novo decodificador e seis variantes SRP concluídos dentro do orçamento de máquina. A CNN com contexto ampliado e fusão 75% SRP/25% CNN reduziram o erro na validação completa para 1,116 m em média entre duas seeds, contra 1,263 m do SRP e 1,286 m da CNN antiga, com IC95% pareado exploratório abaixo de zero. O teste reservado antigo permaneceu intocado; resultados, pesos, tempos e análise por condição estão em `artifacts/round2/round2_report.md`.
- **Recursos e distribuição:** `artifacts/full/resource_audit.json` registra o espaço usado, os tempos medidos e histogramas de N/M/geometria/SNR/RT60 por partição. `data/full` ocupa 124,23 GiB; os WAVs mistura/alvo somam 42,92 GiB. N está balanceado em 8.000/8.000/8.000 cenas no treino e 1.000/1.000/1.000 em validação e teste; as 18 combinações de N/M/geometria variam por no máximo uma cena dentro de cada partição. O manifesto de geração não registrou duração de parede do processo; a auditoria preserva intervalos de modificação dos arquivos como referência, claramente separados do tempo medido de extração de mapas e treinamento.
- **Demonstração:** servidor Streamlit ativo somente no loopback, em `http://127.0.0.1:8501`; health e página raiz respondem HTTP 200. Streamlit AppTest acionou a interface nos limites inferior e superior, gerou quatro imagens e métricas por caso sem exceções. Os detalhes e figuras estão em `artifacts/final/demo_extremes*.json` e `artifacts/final/demo_extremes/`.

### Validação já concluída

Erro médio de localização em metros (3.000 cenas por seed):

| Seed | SRP-PHAT | CNN-mixture | Wave + SRP | CNN-Wave |
|---:|---:|---:|---:|---:|
| 42 | 1,3219 | 1,4128 | 1,6359 | 1,4683 |
| 43 | 1,3219 | 1,3569 | 1,4858 | 1,4061 |
| 44 | 1,3219 | 1,3872 | 1,4184 | 1,4296 |

Diferença pareada no erro médio, candidato menos SRP-PHAT; IC95% bootstrap por cena (2.000 reamostragens):

| Seed | CNN-mixture | Wave + SRP | CNN-Wave |
|---:|---:|---:|---:|
| 42 | +0,0909 [+0,0577; +0,1237] | +0,3140 [+0,2785; +0,3508] | +0,1464 [+0,1100; +0,1815] |
| 43 | +0,0350 [+0,0013; +0,0682] | +0,1639 [+0,1358; +0,1915] | +0,0842 [+0,0493; +0,1198] |
| 44 | +0,0652 [+0,0296; +0,0985] | +0,0965 [+0,0716; +0,1223] | +0,1077 [+0,0724; +0,1425] |

Wave+SRP e CNN-Wave pioraram em localização frente ao SRP-PHAT nas três seeds. O delta médio igualmente ponderado de Wave+SRP foi **+0,1914 m**, portanto o gate pré-especificado reprovou a promoção da Wave como ramo auxiliar de localização. CNN-Wave também piorou contra CNN-mixture em todas as seeds. Na validação da seed 43, a Wave-U-Net aumentou SI-SDR em média +0,916 dB, embora Wave+SRP tenha elevado o erro em +0,1639 m e CNN-Wave em +0,0842 m. A qualidade acústica melhorou segundo SI-SDR, mas isso não preservou os atrasos espaciais relevantes à localização.

O gate foi pré-especificado em `scripts/check_validation_gate.py`: a Wave passaria se o erro médio por fonte de **Wave + SRP-PHAT** fosse estritamente menor que SRP-PHAT, com peso igual para cada seed. O gate foi reprovado e não foi redefinido após observar os resultados.

| Seed | Δ SI-SDR no treino | Δ SI-SDR na validação | Inferência Wave | Aprimoramento total (I/O + Wave) |
|---:|---:|---:|---:|---:|
| 42 | +1,259 dB | +1,440 dB | 0,069 s/cena | 0,216 s/cena (treino) |
| 43 | +0,665 dB | +0,916 dB | 0,038 s/cena | 0,133 s/cena (treino) |

### Teste reservado — avaliação única após os freezes

Métricas agregadas por seed e resumidas com média ± desvio padrão entre as três seeds (3.000 cenas por seed):

| Método | Erro médio (m) | Mediana (m) | P90 (m) | Dentro de 0,50 m |
|---|---:|---:|---:|---:|
| SRP-PHAT | 1,3540 ± 0,0000 | 0,7196 ± 0,0000 | 3,7134 ± 0,0000 | 45,80% ± 0,00 pp |
| CNN sobre mistura | 1,3844 ± 0,0293 | 0,7126 ± 0,0542 | 3,8331 ± 0,0414 | 46,29% ± 1,03 pp |
| Wave-U-Net + SRP-PHAT | 1,4959 ± 0,0999 | 0,9874 ± 0,1907 | 3,8934 ± 0,1122 | 41,23% ± 3,34 pp |
| Wave-U-Net + CNN | 1,4099 ± 0,0518 | 0,8118 ± 0,0878 | 3,8125 ± 0,0525 | 44,57% ± 1,52 pp |

Diferença pareada de erro médio entre seeds, candidato menos referência; IC95% por reamostragem de cenas:

| Comparação | Delta (m) | IC95% (m) |
|---|---:|---:|
| CNN sobre mistura − SRP-PHAT | +0,0305 | [−0,0023; +0,0620] |
| Wave-U-Net + SRP-PHAT − SRP-PHAT | **+0,1420** | **[+0,1209; +0,1637]** |
| Wave-U-Net + CNN − SRP-PHAT | **+0,0559** | **[+0,0239; +0,0867]** |
| Wave-U-Net + CNN − CNN sobre mistura | **+0,0255** | **[+0,0080; +0,0438]** |

O teste confirma, no sentido estritamente confirmatório, a decisão da validação: a Wave aumenta o erro de localização. O SI-SDR melhorou entre +0,56 e +1,36 dB nas três seeds, mas isso não trouxe benefício espacial. A estimativa de latência end-to-end por cena foi 157 ms para SRP-PHAT, 144 ms para CNN sobre mistura, 287 ms para Wave+SRP e 264 ms para Wave+CNN; os fluxos Wave também exigem cerca de 36 ms/cena de inferência Wave e extração de mapas mais lenta. O relatório detalha custos e estratos.

## 1. Corpus e base completa (concluída; comandos abaixo para reprodução)

Nesta execução, LibriSpeech foi preparado, a base foi gerada e auditada, e os hashes foram congelados. Para reconstruir os dados do zero, baixe `train-clean-100.tar.gz`, `dev-clean.tar.gz` e `test-clean.tar.gz` da [página oficial do LibriSpeech no OpenSLR](https://www.openslr.org/12) para `data/full/downloads`. Confira os nomes internos antes de extrair:

```powershell
tar -tzf data/full/downloads/train-clean-100.tar.gz | Select-Object -First 3
tar -tzf data/full/downloads/dev-clean.tar.gz | Select-Object -First 3
tar -tzf data/full/downloads/test-clean.tar.gz | Select-Object -First 3
```

Extraia cada arquivo em `data/full` para preservar o diretório `LibriSpeech` do próprio pacote. Em seguida prepare os manifestos, confirme Pyroomacoustics 0.10.1 e gere as três partições:

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) "src")
python -m voice_localization.cli validate-config --config configs/full.yaml
python -m voice_localization.cli prepare-corpus --root data/full/LibriSpeech --output data/full/manifests
python -c "import pyroomacoustics as pra; print(pra.__version__)"
python -m voice_localization.cli generate-scenes --config configs/full.yaml --split train --workers 2
python -m voice_localization.cli generate-scenes --config configs/full.yaml --split validation --workers 2
python -m voice_localization.cli generate-scenes --config configs/full.yaml --split test --workers 2
python scripts/audit_scenes.py --config configs/full.yaml --output artifacts/full/scene_audio_audit.json
python -m voice_localization.cli extract-maps --config configs/full.yaml --variant mixture --audio-variant mixture
python -m voice_localization.cli validate-data --config configs/full.yaml
python scripts/freeze_full_dataset.py --config configs/full.yaml
```

O resultado esperado é de 24.000 cenas de treino, 3.000 de validação e 3.000 de teste, sem falantes compartilhados, arquivos inválidos ou receitas duplicadas entre partições. A geração pode ser retomada repetindo o mesmo comando. Não avance para o treino se a auditoria apontar falhas bloqueantes.

### Espaço e tempos medidos

A auditoria local encontrou 124,23 GiB em `data/full`: 6,84 GiB de corpus extraído, 6,59 GiB em arquivos baixados, 42,92 GiB nos WAVs de mistura/alvo, 3,26 GiB nos quatro caches de mapas e 64,38 GiB nos áudios aprimorados das três seeds. O volume C: tinha 153,65 GiB livres no instante da auditoria. Os detalhes por pasta e split estão em `artifacts/full/resource_audit.json`.

A extração de mapas da mistura levou 3.371,97 s no treino (0,1405 s/cena), 410,68 s na validação e 413,27 s no teste. Os caches Wave variaram de 0,2105 a 0,2355 s/cena; treinamento, inferência, enhancement e memória estão discriminados em `artifacts/final/report/processing_costs.csv`. A versão do gerador não registrou o tempo total do processo de simulação. A auditoria inclui os intervalos de modificação dos arquivos de cena como referência de proveniência, sem tratá-los como duração exata do comando.

## 2. Treino e seleção na validação (concluídos)

O runner abaixo concluiu nesta máquina e os resultados/checkpoints estão congelados. Não o inicie novamente durante a avaliação do teste. Para uma reprodução futura com o mesmo manifesto congelado, os comandos são:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_training_seeds.ps1
```

Para cada seed, o script treina CNN sobre mistura, Wave-U-Net e CNN sobre mapas da Wave-U-Net; aprimora treino e validação; extrai os mapas com normalizador ajustado apenas no treino; avalia os quatro fluxos na validação; registra comparações pareadas e salva hashes dos checkpoints e normalizadores. O arquivo `artifacts/final/wave_validation_gate.json` aplicará a regra declarada para decidir se o ramo Wave atende ao critério de inclusão. Revise também os históricos de treino, perdas, SI-SDR e uso de GPU antes da avaliação automática do teste.

## 3. Teste congelado e relatório (concluídos)

A orquestração desta execução confirmou os freezes, executou a avaliação única e gerou o relatório. Para uma reprodução manual futura, só prossiga depois que `artifacts/final/wave_validation_gate.json` existir e todos os arquivos `artifacts/final/validation/seedN/frozen_models.json` estiverem fechados. Rode uma única vez:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_final_evaluation.ps1
```

O script recusa prosseguir se os arquivos congelados tiverem mudado. Os resultados brutos ficam em `artifacts/final/seed42`, `seed43` e `seed44`; tabelas gerais e estratificadas, bootstrap pareado por cena, curvas, distribuição dos erros, exemplos espaciais com regra explícita e relatório ficam em `artifacts/final/report/`.

## 4. Demonstração local

No ambiente Python usado pelo projeto, instale o extra da demonstração e rode:

```powershell
python -m pip install -e ".[demo]"
streamlit run app.py
```

A demonstração cria uma cena simulada com fala de treino do LibriSpeech e permite somente N, M, geometria, SNR e RT60 dentro do domínio estudado. Ela mostra posições reais e estimadas, erro, latência, SI-SDR, parâmetros realizados e hash da Wave-U-Net. Os áudios e as posições são resultados simulados e não devem ser apresentados como gravação real.

Nesta entrega, o servidor está ativo em `http://127.0.0.1:8501` no ambiente `voz-gpu` e escuta apenas no computador local. A rota `/_stcore/health` e a página raiz responderam HTTP 200; o processo permaneceu ativo. A interface foi exercitada por Streamlit AppTest em dois extremos do domínio: N=1/M=4/perímetro/SNR=10 dB/RT60=0,20 s e N=3/M=8/círculo/SNR=30 dB/RT60=0,80 s. Ambos mostraram os quatro métodos, imagens, estimativas e métricas sem exceções. As configurações realizadas e os resultados estão em `artifacts/final/demo_extremes_ui.json` e `artifacts/final/demo_extremes.json`; as oito figuras renderizadas estão em `artifacts/final/demo_extremes/`. Os exemplos usam cenas novas simuladas com origens de treino, não o teste reservado, e servem apenas à validação funcional da interface. Para evitar conflito entre runtimes OpenMP de PyTorch, Pyroomacoustics e Matplotlib no Windows, o app executa simulação e renderização em processos auxiliares separados. A interface identifica Wave como experimental e informa que não passou o gate de localização.

## 5. Artefatos de conclusão

- `data/full/manifests/`: manifestos dos áudios e das cenas por split.
- `artifacts/full/scene_audio_audit.json`: auditoria da base e da simulação.
- `artifacts/{cnn_mixture_seedN, wave_seedN, cnn_wave_seedN}/`: checkpoints, configuração, ambiente e históricos.
- `artifacts/final/validation/`: resultados usados para selecionar os modelos antes do teste.
- `artifacts/final/seedN/`: resultados brutos por cena no teste reservado.
- `artifacts/final/report/`: tabelas CSV, ICs, gráficos, exemplos espaciais e relatório acadêmico.
- `artifacts/final/demo_extremes*.json` e `artifacts/final/demo_extremes/`: configurações, métricas e PNGs dos dois extremos usados para verificar a demonstração.
- `app.py`: demonstração Streamlit baseada nos checkpoints e normalizadores congelados.
- `scripts/generate_demo_scene.py` e `scripts/render_demo_plots.py`: processos auxiliares da demonstração para simular e renderizar sem carregar PyTorch junto dessas bibliotecas.

As conclusões finais se limitam à sala simulada de 8 × 6 × 3 m, fontes e microfones a 1,5 m, N=1–3, M=4/6/8, SNR solicitado de 10–30 dB, RT60 solicitado de 0,2–0,8 s, fontes estáticas e sincronização ideal. A comparação não demonstra generalização para gravações reais ou outras salas.

## Nota acadêmica sobre Wave-U-Net

O trabalho original de Stoller, Ewert e Dixon apresenta uma U-Net 1D multiescala no waveform para separação de fontes; isso fundamenta a escolha de uma arquitetura temporal, mas não valida por si só enhancement multicanal para localização. Ho et al. ampliam Wave-U-Net para fala multicanal com codificação por canal e atenção entre canais, explicitando como explorar informação espacial. A implementação deste projeto é uma adaptação menor, condicionada pelas máscaras e coordenadas dos microfones, e deve ser descrita como tal — não como reprodução fiel desses artigos. Veja o [artigo original da Wave-U-Net](https://ismir2018.ircam.fr/doc/pdfs/205_Paper.pdf) e a [extensão multicanal da Interspeech 2020](https://www.isca-archive.org/interspeech_2020/ho20_interspeech.html).

O alvo supervisionado é a soma das fontes de fala após as respostas impulsivas da sala em cada canal, sem o ruído aditivo. Isso mantém a imagem espacial simulada; não se deve reduzi-la a mono, separar cada fonte nem remover a reverberação para este experimento. A conclusão sobre utilidade para localização depende de medir posições e erros pareados na validação/teste reservado; SI-SDR isolada não demonstra preservação dos atrasos intercanal nem ganho de localização.
