# Plano em camadas — localização de múltiplas fontes de voz

**Próxima rodada proposta:** a seção final, “Nova rodada de pesquisa — reduzir falhas grandes em até 12 horas de máquina”, contém a revisão de artigos, as hipóteses priorizadas e o orçamento fechado da rodada 3. As seções anteriores preservam o histórico do projeto.

## 1. Avaliação e decisões do projeto

O projeto tem uma sequência experimental coerente: gerar cenas acústicas com posições conhecidas, construir mapas SRP-PHAT, treinar uma CNN para refinar a localização e comparar os dois métodos nas mesmas cenas. A simulação permite controlar as condições e obter rótulos automaticamente. O resultado científico será identificar em quais condições o aprendizado melhora a localização, sem pressupor que a CNN necessariamente vencerá o método clássico.

O estado inicial da pasta continha apenas as duas propostas em PDF. A implementação descrita na seção 1.1 foi adicionada para tratar os principais riscos: qualidade da simulação, vazamento entre partições, erros na transformação de atrasos em coordenadas e custo de gerar 30 mil cenas. A geração completa continua precedida por um piloto verificável.

### Decisões confirmadas

- **N é sempre conhecido**, tanto para a CNN quanto para o baseline. Não haverá estimativa do número de fontes. Esta decisão substitui o trecho sobre contagem de fontes no primeiro PDF.
- Estimar apenas **(x, y)**, com fontes e microfones a **1,5 m**. A simulação acústica continua tridimensional, incluindo piso e teto.
- Executar geração e treinamento neste computador: aproximadamente **16 GB de RAM, NVIDIA MX550 com 2 GB de VRAM e processador i7-1355U**.
- A primeira entrega inclui **base completa de aproximadamente 30 mil cenas, parâmetros e framework de treinamento**, com visualização intermediária integrada.
- Testes mínimos de aprendizado estão incluídos na primeira entrega. O treinamento experimental completo e a apresentação final ficam para uma etapa futura.

### Premissas iniciais de implementação

Manter os parâmetros da proposta: sala de 8 × 6 × 3 m, cenas de 2 segundos, 16 kHz, N ∈ {1, 2, 3}, M ∈ {4, 6, 8}, RT60 desejado entre 0,2 e 0,8 s e SNR entre 10 e 30 dB. Fontes estáticas, microfones sincronizados, posições dos microfones conhecidas e ruído aditivo branco independente por canal. Generalização para outras salas, movimento, alturas desconhecidas e gravações reais não faz parte da primeira versão.

## 1.1. Estado da implementação

O plano já está representado por uma implementação inicial em `src/voice_localization`, com configurações em `configs/`, testes em `tests/` e os comandos descritos no README. Os componentes entregues são:

- configuração validada, seeds, manifestos JSONL e verificação de vazamento entre partições;
- descoberta do LibriSpeech e geração de cenas com backend Pyroomacoustics, além de backend anecoico controlado para smoke tests;
- geometrias de microfones, amostragem das fontes, normalização de áudio, SNR realizado e medição de RT60 por resposta impulsiva;
- STFT/GCC-PHAT, SRP-PHAT posicional em grade, alvos gaussianos, normalização ajustada somente no treino e decodificação de exatamente N picos;
- baseline, métricas com associação húngara, CNN convolucional, loop PyTorch com checkpoints, early stopping, TensorBoard e figuras intermediárias;
- ramo Wave-U-Net 1D multicanal residual, condicionado pela presença e coordenadas dos microfones, supervisionado com áudio limpo reverberante; a saída alimenta SRP-PHAT e a CNN auxiliar de mapas;
- comandos de preparação, geração, extração, validação, baseline, smoke, treino e avaliação.
- script `scripts/setup_and_run_gpu.ps1` para reutilizar o Anaconda, criar um ambiente GPU separado, instalar o wheel CUDA e executar a validação automaticamente.

A Wave-U-Net é uma adaptação do projeto, não uma reprodução fiel da arquitetura multicanal de Ho et al. nem um localizador de fontes por si só. Sua utilidade é julgada pelo erro de posição após realimentar o SRP-PHAT ou a CNN espacial, além da SI-SDR.

Em 2026-11-04, o ambiente `voz-gpu` (Python 3.11.5, PyTorch 2.14.0+cu126, CUDA na NVIDIA GeForce MX550 e Pyroomacoustics 0.10.1) passou na suíte automatizada então existente. O piloto Wave-U-Net com 72 cenas sintéticas também passou pela auditoria de áudio, SNR, RT60, separação entre splits e mapas. A etapa 6 foi concluída com LibriSpeech e 30.000 cenas congeladas; os modelos das seeds 42, 43 e 44 foram selecionados na validação e congelados. O gate agregado reprovou a Wave como ramo de localização, e o teste reservado foi avaliado uma única vez após os freezes. O relatório final está gerado. A demonstração Streamlit está servida somente no endereço local `http://127.0.0.1:8501`.

### Situação experimental em 2026-11-07

- Base congelada: 24.000 cenas de treino, 3.000 de validação e 3.000 de teste; IDs dos splits e hashes em `artifacts/full/frozen_dataset/freeze.json`.
- Limitação de acústica: 274/30.000 cenas (0,91%) tiveram RT60 realizado fora do intervalo solicitado [0,2; 0,8] s. As faixas medidas foram 0,191–0,835 s no treino, 0,196–0,812 s na validação e 0,191–0,824 s no teste. `build_final_report.py` registra a contagem e as faixas por split automaticamente.
- Validação congelada: seeds 42, 43 e 44 em `artifacts/final/validation/seedN/`, com hashes em `frozen_models.json`. Na seed 44, o Wave-U-Net terminou 5 épocas (L1 de validação 0,07165) e a CNN-Wave parou após 10 épocas; seu melhor erro de localização foi 1,302 m na época 2.
- Gate pré-especificado: reprovado. Wave+SRP piorou o erro médio em todas as seeds; delta médio igualmente ponderado de **+0,1914 m** contra SRP. CNN-Wave também teve erro maior que CNN-mixture nas três seeds. Portanto, os resultados atuais não justificam promover Wave-U-Net como ramo auxiliar de localização.
- A SI-SDR melhorou na validação (por exemplo, +0,916 dB na seed 43), mas não se traduziu em melhor localização. Isso confirma que SI-SDR é uma métrica auxiliar, não substituta da tarefa final.
- Teste reservado: avaliado uma única vez após congelamento dos modelos. Wave+SRP teve erro 1,496 m (delta +0,142 m; IC95% +0,121 a +0,164 m) contra SRP-PHAT; Wave+CNN teve 1,410 m (delta +0,056 m; IC95% +0,024 a +0,087 m). O teste confirma a decisão do gate, sem mudar seleção ou regra.
- Relatório: `artifacts/final/report/final_report.md`, junto às tabelas CSV, gráficos e exemplos espaciais.
- Recursos e distribuição: `artifacts/full/resource_audit.json` registra ocupação atual de 124,23 GiB para `data/full`, tamanhos por split, histogramas de N/M/geometria/SNR/RT60 e tempos medidos de extração de mapas. O gerador não gravou o tempo de parede; janelas de modificação dos arquivos foram anotadas apenas como proveniência, não como duração exata.
- Demonstração: Streamlit responde em `http://127.0.0.1:8501` (health HTTP 200), escutando apenas no computador local. A interface foi acionada por Streamlit AppTest nos dois extremos do domínio: ambos geraram quatro imagens dos métodos e métricas sem exceções. Configurações, resultados de cada cena e figuras estão em `artifacts/final/demo_extremes.json`, `artifacts/final/demo_extremes_ui.json` e `artifacts/final/demo_extremes/`.
- Segunda rodada: CNN com dilatações [1, 2, 4, 8] e fusão de mapas 75% SRP/25% CNN melhoraram a localização nas duas seeds e condições avaliadas na validação; erro médio agregado 1,116 m. A avaliação é exploratória porque as hipóteses foram selecionadas nessa validação; o teste reservado não foi reutilizado. Relatório e artefatos estão em `artifacts/round2/round2_report.md`.

## 2. Primeira etapa — dados completos e framework confiável

### Camada 0 — ambiente e contrato experimental

**Objetivo:** tornar cada execução reproduzível e explicitar o que será comparado.

- Criar ambiente isolado com Python 3.11, PyTorch, Pyroomacoustics, NumPy, SciPy, SoundFile, Matplotlib, TensorBoard e pytest. Fixar as versões após verificar instalação e execução CPU/CUDA no Windows; não alterar o Python global.
- Usar PyTorch diretamente, com um loop de treinamento pequeno e explícito, sem adicionar outro framework de treinamento inicialmente.
- Separar módulos de dados, simulação, representação espacial, baseline, modelo, treinamento, métricas e visualização.
- Centralizar parâmetros em configurações YAML, com perfis `smoke`, `pilot` e `full`. Registrar configuração resolvida, versões, seed e identificador do conjunto de dados em cada execução.
- Disponibilizar comandos separados para preparar corpus, gerar cenas, validar dados, extrair mapas, avaliar baseline, testar o framework, treinar e avaliar resultados.

**Critério de saída:** imports, leitura de áudio, simulação simples e forward/backward de uma rede pequena funcionando; configuração inválida produz erro claro antes de uma tarefa longa.

### Camada 1 — corpus, partições e simulador

**Objetivo:** produzir dados rastreáveis, com rótulos corretos e partições independentes.

- Usar LibriSpeech `train-clean-100`, `dev-clean` e `test-clean` como origens de treino, validação e teste, respectivamente. Verificar explicitamente que não há falantes compartilhados entre as partições utilizadas.
- Fixar os manifestos antes de gerar cenas: **24.000 de treino, 3.000 de validação e 3.000 de teste**. Balancear aproximadamente as combinações de N, M e família de geometria em cada partição; distribuir RT60 e SNR uniformemente nas faixas previstas.
- Usar duas famílias de arranjo: microfones distribuídos no perímetro de um retângulo interno à sala e microfones em círculo. Defaults: retângulo de (0,5; 0,5) a (7,5; 5,5) m, amostrado uniformemente no perímetro; círculo centrado em (4; 3) m, raio de 2 m e espaçamento angular uniforme. Registrar as coordenadas exatas.
- Amostrar fontes dentro da sala, com pelo menos 0,5 m das paredes, 0,5 m dos microfones e 0,75 m entre fontes. Usar falantes distintos na mesma cena e recortes com fala ativa; registrar a regra de detecção de atividade e rejeitar recortes predominantemente silenciosos.
- Normalizar os sinais de fala por RMS ativo antes da mistura. Definir SNR pela potência média da mistura limpa recebida nos microfones, antes de adicionar ruído; registrar SNR solicitado e realizado.
- Gerar resposta impulsiva e convolução com contexto anterior ao recorte de 2 s, evitando que todas as cenas comecem artificialmente sem reverberação. Aplicar qualquer ajuste de ganho igualmente a todos os canais, preservando as relações espaciais.
- Registrar por cena: ID, partição, seed, IDs dos áudios e falantes, recortes, N, M, posições, parâmetros acústicos, versão do gerador e caminhos dos artefatos. Identificar receitas duplicadas e impedir sua presença em partições diferentes.
- Medir RT60 realizado em uma amostra estratificada das respostas impulsivas. A aproximação usada para configurar a sala não garante que o RT60 realizado seja igual ao solicitado.

**Armazenamento:** áudio multicanal WAV float32 e manifestos JSONL, mais mapas pré-calculados em arquivos NumPy agrupados por lotes. Como a Wave-U-Net supervisionada requer a mistura e um alvo limpo reverberante, os dois conjuntos de WAV somam aproximadamente **46 GB decimais** para 30 mil cenas com M médio de 6. Corpus, mapas e arquivos temporários são adicionais. Não manter todas as cenas em RAM nem armazenar todas as respostas impulsivas por padrão.

**Critério de saída:** manifestos auditáveis, ausência de vazamento de falantes e receitas, áudio finito com 32.000 amostras por canal e metadados coerentes com N e M.

### Camada 2 — piloto, representação espacial e baseline

**Objetivo:** validar a física e o processamento antes de escalar.

- Gerar primeiro um piloto de **180 cenas**, cobrindo as 18 combinações de N, M e geometria, com diferentes SNR e RT60. Utilizar apenas origens de treino/validação para depuração; não inspecionar o teste para ajustar escolhas.
- Usar STFT com janela Hann de 512 amostras e salto de 256 amostras, equivalentes a 32 e 16 ms. Tratar explicitamente os limites de atraso: calcular correlações com padding adequado, inicialmente FFT de 1024 pontos, e testar atrasos próximos ao máximo físico do arranjo para evitar aliasing circular.
- Implementar SRP-PHAT **posicional**: para cada ponto candidato, calcular diferenças de distância até os microfones e os atrasos correspondentes, usando c = 343 m/s. Não substituir coordenadas por uma estimativa apenas de direção de chegada.
- Usar grade de 0,10 m no plano de altura fixa, com eixos e convenção de índices documentados. Agregar correlações ao longo do tempo e normalizar a soma pelo número de pares de microfones, permitindo comparar M diferentes.
- Produzir um mapa de entrada de tamanho fixo e um mapa-alvo com gaussianas nas posições verdadeiras, sigma inicial de 0,15 m e combinação por máximo. Fixar a normalização usando somente o treino e armazenar seus parâmetros.
- Extrair exatamente N posições para ambos os métodos com máximos locais e supressão de vizinhança, inicialmente raio de 0,30 m. Se faltarem máximos locais, completar pelos maiores pontos ainda permitidos; registrar a ocorrência. Ajustes posteriores usam somente validação.
- Associar estimativas e referências pelo algoritmo húngaro, minimizando a distância total. Medir erro médio em metros, mediana, percentil 90 e proporção de fontes localizadas a até 0,50 m.
- Medir tempo por cena, memória e armazenamento, discriminando simulação e extração. Projetar o custo das 30 mil cenas com as medições, sem prometer prazo antes do piloto.

**Critério de saída:** casos sintéticos de fonte única sem ruído/reverberação localizados a até duas células da grade; convenções de atraso verificadas; mapas e métricas corretos para fontes sem ordenação; custo projetado documentado.

### Camada 3 — geração completa e auditoria

**Objetivo:** concluir a base de treino, validação e teste antes do treinamento experimental.

- Gerar as 30 mil cenas apenas após a aprovação técnica do piloto pelos critérios anteriores. O piloto de depuração não entra automaticamente no conjunto final.
- Executar em lotes, inicialmente com dois processos de geração, aumentando somente se as medições permitirem. Usar escrita atômica, log de falhas e retomada por ID; não sobrescrever silenciosamente dados válidos.
- Pré-calcular os mapas SRP-PHAT e os alvos, evitando repetir a simulação durante o treinamento. Versionar o cache pela configuração que o produziu.
- Emitir relatório de contagens, distribuições, integridade, espaço utilizado e separação entre partições. No teste, limitar essa etapa a auditoria de integridade e protocolo, sem usar desempenho para tomar decisões.

**Critério de saída:** 30 mil cenas válidas, respectivas representações e manifestos, relatório de auditoria sem falhas bloqueantes e retomada de geração comprovada.

### Camada 4 — modelo, parâmetros e visualização intermediária

**Objetivo:** deixar o treinamento pronto para execução e suficientemente testado.

**Modelo inicial:** CNN totalmente convolucional, preservando a resolução da grade. Entrada com dois canais: mapa SRP-PHAT e plano constante N/3. Quatro convoluções 3 × 3 com padding, canais 16, 32, 32 e 16 e ativações ReLU; convolução final 1 × 1 para um mapa de logits. Aplicar sigmoid para visualização e extração das posições. A saída não usa softmax, pois há múltiplas fontes.

**Linha auxiliar de enhancement:** variante Wave-U-Net multicanal que recebe a mistura no waveform, máscara de microfones presentes e coordenadas dos sensores, e prevê uma correção residual por canal. O alvo correto aqui é a soma de todas as fontes de fala após a resposta impulsiva da sala em cada microfone, sem o ruído aditivo; ele fica alinhado à mistura e não deve ser convertido em mono, isolado por fonte ou tornado anecoico, pois isso alteraria as pistas espaciais avaliadas pelo localizador. O ramo entra na comparação somente se mapas SRP extraídos do áudio aprimorado melhorarem a localização na validação; melhora de SI-SDR isolada não basta.

| Parâmetro | Default inicial |
|---|---|
| Loss | BCEWithLogits com alvos gaussianos; peso positivo calculado apenas no treino e fixado |
| Otimizador | AdamW |
| Learning rate | 0,001 |
| Weight decay | 0,0001 |
| Batch | 16; reduzir para 8 e depois 4 se necessário por memória |
| Limite futuro | 50 épocas |
| Early stopping | 8 épocas sem melhora no erro médio de validação |
| Scheduler | ReduceLROnPlateau, fator 0,5 e paciência 3, pelo erro de validação |
| Checkpoint principal | Menor erro médio de localização na validação |
| Seed inicial | 42 |
| Precisão | float32 |

- Usar DataLoader sem simulação online e com `num_workers=0` inicialmente no Windows. Manter CPU como alternativa funcional, verificando CUDA antes de selecioná-la.
- Separar treino e validação no loop; salvar checkpoint melhor e último, incluindo modelo, otimizador, scheduler, época, estados aleatórios e configuração.
- Integrar TensorBoard: loss de treino/validação, erro de localização, proporção dentro da tolerância, learning rate, duração de época e memória GPU.
- Em um conjunto fixo de exemplos de validação, mostrar planta da sala, microfones, posições verdadeiras, mapa SRP-PHAT, mapa previsto e N posições extraídas. Salvar também PNGs e métricas CSV para consulta sem painel.
- Reportar métricas por N, M, geometria e faixas de SNR/RT60, evitando que a média global esconda falhas em cenas difíceis.

**Contratos entre componentes:** o gerador entrega áudio e metadados; o extrator entrega mapa e grade; o Dataset entrega entrada, alvo, N, coordenadas verdadeiras e ID; o modelo entrega logits espaciais; o decodificador recebe mapa e N e entrega N coordenadas; o avaliador recebe dois conjuntos de coordenadas e entrega métricas independentes da ordem.

**Critério de saída:** pipeline completo funcionando em teste curto, logs e imagens legíveis, checkpoints restauráveis e execução sem exceder a memória local.

## 3. Testes que liberam a primeira entrega

- **Dados:** formatos, coordenadas dentro dos limites, ausência de valores inválidos, número de canais/fontes correto, partições sem falantes compartilhados e seeds reproduzíveis.
- **Acústica e representação:** sinais com atraso conhecido, troca consistente da ordem dos microfones, fonte única em posições distintas e casos nos limites de atraso físico.
- **Métricas:** permutar a ordem das fontes não altera o erro; coincidência perfeita resulta em erro zero; deslocamento conhecido produz a distância esperada.
- **Aprendizado mínimo:** sobreajustar 8–16 cenas de treino, verificando redução de loss e erro em relação à inicialização. Limitar o teste a 500 passos; falha exige diagnóstico antes do treino completo. Esse teste não constitui evidência de generalização.
- **Infraestrutura:** executar treino e validação curtos, salvar/restaurar estado, retomar uma geração interrompida e verificar invalidação do cache quando muda o processamento.
- **Visualização:** verificar orientação dos eixos, unidades em metros, alinhamento entre mapas e coordenadas e distinção clara entre referência, baseline e previsão.

**Estado atual:** base congelada; treino, validação em três seeds, gate, teste reservado e relatório final concluídos. A evidência não sustenta promover Wave-U-Net para localização nesta configuração. O servidor Streamlit segue ativo em `http://127.0.0.1:8501`. A interface foi exercitada com os parâmetros extremos e apresentou os quatro métodos; as figuras também foram conferidas diretamente. Para evitar o conflito OpenMP entre PyTorch e as bibliotecas da simulação/renderização, estas duas tarefas rodam em processos auxiliares isolados. Resultados completos estão em [EXECUCAO_FINAL.md](EXECUCAO_FINAL.md), `artifacts/final/report/` e nos artefatos de demonstração extrema.

## 4. Etapa final — teste reservado e entrega

1. Concluídos: validação em três seeds, freezes, gate pré-especificado e avaliação única do teste.
2. Concluídos: relatório com métricas globais e por condição, bootstrap pareado, custos, RT60 realizado, curvas e exemplos espaciais.
3. Concluído: servidor Streamlit iniciado no loopback e health endpoint local respondendo. Streamlit AppTest acionou o botão “Gerar cena e comparar” nos limites inferior e superior do domínio; cada caso mostrou as quatro linhas, posição real, estimativas e métricas, sem exceções. Configurações, resultados e imagens estão registrados em `artifacts/final/demo_extremes*`; Wave permanece explicitamente experimental.
4. Na interpretação, manter Wave-U-Net como estudo negativo/experimental de enhancement; não a promover para localização sem uma nova hipótese ou arquitetura e validação independente.

O sucesso científico não depende de demonstrar superioridade universal da CNN. Depende de uma comparação justa e reproduzível, que mostre ganhos, limitações e custo dentro do domínio simulado. Não extrapolar os resultados da sala fixa para ambientes reais ou novas salas sem novos experimentos.

## 5. Ordem de execução e referências

**Ambiente → simulador e manifestos → piloto e baseline → base congelada → validação em três seeds → gate → teste reservado uma vez → relatório e demonstração.**

As camadas têm critérios próprios de conclusão. Corrigir problemas na camada responsável antes de escalar evita gastar tempo de simulação ou treinamento com dados incorretos.

Este plano consolida o escopo e as decisões vigentes do projeto; os materiais pessoais de apresentação e os PDFs de apoio não fazem parte do repositório público.

Referências técnicas consultadas:

- [LibriSpeech — corpus, partições e licença](https://www.openslr.org/12).
- [Pyroomacoustics — simulação de salas e ressalva sobre RT60 realizado](https://pyroomacoustics.readthedocs.io/en/pypi-release/pyroomacoustics.room.html).
- [Pyroomacoustics — SRP-PHAT e parametrização de direção/distância](https://pyroomacoustics.readthedocs.io/en/pypi-release/pyroomacoustics.doa.srp.html).
- [Liu et al. — localização bidimensional em etapas](https://arxiv.org/abs/2210.10265).
- [Grinstein et al. — Neural-SRP](https://arxiv.org/abs/2403.09455).

A CNN proposta é um baseline próprio de refinamento de mapas, inspirado pelo tema dos artigos; não deve ser apresentada como reprodução das arquiteturas publicadas.

---

## Nova rodada de pesquisa — reduzir falhas grandes em até 12 horas de máquina

**Data da revisão bibliográfica e do código: 2026-11-08. Status: planejada, ainda não executada.** Esta seção orienta a próxima rodada e preserva o histórico das anteriores. O orçamento cobre todas as computações: preparação, download, simulação, extração, treino, avaliação e análises. Implementação manual não entra no relógio de máquina.

### 1. Problema concreto e objetivo

O melhor resultado da rodada 2 foi 1,116 m de erro médio, mediana de 0,256 m, P90 de 3,370 m e 55,4% das fontes até 0,50 m. Portanto, há muitos acertos razoáveis junto com falhas de vários metros. A prioridade é reduzir fontes perdidas, picos de reflexão e duplicações de fonte, em vez de apenas deslocar alguns centímetros um pico já correto. Esses resultados são de validação reutilizada para seleção, não de um novo teste independente.

Manter sala 8×6×3 m, altura 1,5 m, N conhecido (1–3), M=4/6/8 e as duas geometrias. Manter associação húngara e erro ponderado por fonte. Não facilitar a tarefa removendo M=4, N=3, reverberação alta ou cenas difíceis.

**Meta prática de pesquisa, sem promessa de desempenho:** erro médio ≤0,75 m, P90 ≤2,0 m e ≥65% das fontes até 0,50 m. **Gate mínimo para promover um candidato:** reduzir a média em pelo menos 0,15 m frente à fusão atual, reduzir P90 em pelo menos 0,25 m e reduzir em pelo menos 20% a proporção de erros acima de 1 m; não diminuir acertos até 0,50 m. Um ganho menor continua sendo registrado, mas não será chamado de solução suficiente para o problema levantado.

**Critério adicional solicitado para a entrega:** apresentar pelo menos 100 estimativas de fonte com erro estritamente **<0,10 m**, com seus IDs de cena/fonte e posições estimada/referência. Contabilizar primeiro no holdout independente congelado e reportar também o total sobre todas as fontes, além da taxa. Reportar separadamente quantas cenas têm todas as fontes abaixo de 0,10 m. Não selecionar só os casos fáceis para compor o denominador nem usar exemplos de validação como se fossem confirmação independente. A contagem não substitui as métricas de média, cauda e regressão; não é promessa de que um método atingirá esse limiar.

### 2. Diagnóstico fundamentado no projeto

- `spatial.py` calcula GCC por janela e depois toma uma média temporal por par. O mapa final soma todos os pares, reduzindo os dois segundos de áudio a um único mapa. Uma CNN posterior não recupera toda a informação temporal/espectral descartada.
- Os atrasos são calculados por diferenças de distância, adequados ao campo próximo. Bibliotecas/papers que retornam apenas direção não substituem diretamente essa localização (x,y).
- O arranjo de perímetro tem distâncias máximas entre sensores de 7,071–8,062 m: limites físicos de atraso de 20,6–23,5 ms. O círculo tem máximo de 4 m/11,7 ms. A janela atual é Hann de 512 amostras/32 ms. Na validação, o P99 dos atrasos verdadeiros absolutos foi 256,88 amostras/16,06 ms e o máximo foi 337,06/21,07 ms. Isso justifica testar sensibilidade ao janelamento; não demonstra que ele seja o principal erro. Só 1,04% dos atrasos analisados ultrapassam metade da janela.
- A grade é de 0,10 m. A distância máxima ao ponto mais próximo de uma grade quadrada é aproximadamente 0,071 m no interior. Diminuir o passo não explica nem resolve sozinho erros médios de um metro.
- A fusão implementada combina percentil do SRP com sigmoid da CNN; os escores têm escalas com significados diferentes. O peso 25% não é uma probabilidade calibrada. Testar normalização comum é uma ablação barata, não a aposta central.
- Na triagem anterior cada configuração teve seed diferente (3100+índice). Na próxima comparação usar a mesma seed e a mesma ordem de minibatches para todas as configurações; depois repetir o finalista com outra seed.
- Correção documental feita nesta revisão: no estrato de 20 cenas com RT60 realizado >0,8 s, a CNN ampliada foi 0,260 m **pior** que SRP (1,750 contra 1,490 m), não melhor. A fusão também piorou. Não ajustar uma regra específica a partir dessas 20 cenas.

### 3. Métodos pesquisados, adequação e prioridade

| Linha | Evidência e adaptação ao projeto | Decisão para 12 h |
|---|---|---|
| SRP com informação tempo/frequência preservada | SRP-NMF explora esparsidade da fala em grupos espectrais; o SRP multidimensional/SBL conserva informação tempo-frequência para resolver múltiplas fontes. Esses trabalhos sustentam investigar o descarte de informação, sem garantir os mesmos ganhos aqui. | **Prioridade alta:** mapas por blocos/bandas e consenso espacial; NMF/SBL completos ficam para evolução posterior. [R1, R2] |
| Remoção da influência de fonte dominante | De-emphasis e a extensão multifuente de SVD-PHAT buscam uma fonte e reduzem sua contribuição antes da próxima. Isso enfrenta uma limitação que aumentar apenas o raio de NMS não resolve. | **Prioridade alta:** adaptação leve no domínio GCC, preservando atrasos de campo próximo; não executar SVD gigante nem chamar a adaptação de reprodução exata. [R3, R4] |
| Neural-SRP / SRP-DNN | Aprendem evidência por pares e/ou fase de caminho direto, usando a estrutura física antes da agregação espacial. SRP-DNN também trata interação entre fontes. | **Prioridade intermediária:** ablação pequena com mapas multibanda como entrada; reprodução integral das redes fica fora deste teto. [R5, R6] |
| Seleção de regiões de caminho direto | DPD tenta selecionar bins menos dominados por reverberação. O método de diretividade consultado depende de harmônicos esféricos e arranjo esférico/volumétrico. | Usar como motivação para seleção de confiabilidade, sem aplicar sua fórmula diretamente aos nossos arranjos planares distribuídos. Coerência alta isolada não prova caminho direto. [R7] |
| MUSIC de campo próximo / MVDR | MUSIC esférico mostra por que modelar curvatura da frente de onda importa. Métodos de subespaço podem discriminar fontes, mas reverberação coerente, aliasing e M=4/N=3 deixam pouca margem. | Reserva comparativa se as duas primeiras linhas falharem cedo. Exigir steering esférico (x,y) e comparar com o mesmo N. Não usar MUSIC angular como se produzisse posição. [R8] |
| Refinamento hierárquico da grade | SRP density/hierarchical refinement é uma alternativa publicada de busca multifuente. | Só refinar em 0,025 m ao redor de candidatos corretos. Baixa prioridade para a cauda de erros em metros. [R9] |

Kalman/partículas podem ajudar a acompanhar fontes móveis, mas estas cenas têm fontes estáticas: simples suavização não distingue uma reflexão estável da fonte real. Wave-U-Net, ASR e grandes modelos genéricos não recebem novo treino nesta rodada.

### 4. Protocolo para descobrir a causa antes de treinar

Usar os WAVs existentes e preservar os caches congelados. Criar `artifacts/round3/` com configurações, IDs, hashes e tempos. Não avaliar o teste antigo durante esta rodada.

1. Fixar 360 cenas de triagem da validação (20 por combinação N/M/geometria), seed 20261108, com seleção distribuída nas faixas de SNR e RT60. Fixar 6.000 cenas de treino da mesma maneira, sem interseção de IDs. A triagem continua exploratória.
2. Reproduzir exatamente a fusão atual nas 360 cenas e depois na validação completa, incluindo percentil SRP + sigmoid CNN, peso CNN 0,25 e máximos locais/raio 0,60 m. Esse resultado é a referência de promoção.
3. Salvar erros por fonte e cena; reportar média, mediana, P90, frações <0,10 m e ≤0,25/0,50 m, frações >1/2 m, proporção de cenas com **todas** as fontes abaixo de 0,10 m e até 0,50 m e latência. Examinar N/M/geometria/SNR/RT60 e separação entre fontes.
4. Extrair até 32 máximos locais por mapa. Fazer uma análise oracle por associação húngara entre esses candidatos distintos e as referências. Medir cobertura até 0,25/0,50 m e comparar com as N posições efetivamente selecionadas. Essa análise usa rótulos apenas para diagnóstico e nunca conta como desempenho do localizador.
5. Ressimular 18 receitas de treino, uma por combinação, nas condições: caminho direto sem ruído, reverberante sem ruído e original. Manter recortes, posições e ganhos; salvar áudio de fonte individual apenas nesse piloto. Se falhar no caso direto de fonte única, interromper a busca de hiperparâmetros e corrigir sinal de atraso, alinhamento e steering primeiro. A entrada limpa reverberante somada existente permite uma ablação sem ruído nas 360 cenas sem gerar toda a base novamente.
6. Se a cobertura oracle for alta mas a seleção das N fontes for ruim, priorizar consenso/de-emphasis. Se a cobertura cair com reverberação ou mistura de fontes, priorizar os recursos tempo/frequência. Se nem os mapas de caminho direto forem corretos, não atribuir a falha à rede.

### 5. Hipóteses e ensaios fechados

**H1 — janelamento e escores:** comparar a configuração atual com Hann de 2.048 e 4.096 amostras (saltos de metade da janela, FFT 2×janela) e janela retangular de 2.048 amostras. Fixar expoente PHAT=1 para isolar janelamento; não cruzar uma grande grade de hiperparâmetros. Registrar efeito por comprimento dos pares. Em paralelo, comparar a fusão atual com percentis em ambos os mapas, mantendo peso 0,25. Não alimentar checkpoints antigos com representação acústica nova: avaliar essas alterações diretamente como localizadores físicos.

**H2 — consenso tempo/frequência:** usar a melhor janela de H1 e conservar quatro blocos consecutivos de 0,5 s e quatro bandas [300,800), [800,1600), [1600,3000), [3000,6000] Hz. Calcular GCC por bloco/banda e projetar atrasos por geometria exata. Não normalizar bins sem energia para terem o mesmo peso que fala. Rejeitar blocos abaixo de 10% da energia máxima da cena e reduzir o peso de bandas por sua fração de energia observada. Se todos forem rejeitados, voltar ao SRP completo.

Produzir no máximo 16 mapas bloco/banda, extrair até três máximos por mapa e agrupar candidatos com DBSCAN (`min_samples=2`, raios 0,25 e 0,50 m). Ordenar grupos por soma de evidência, obter a posição por mediana ponderada dos votos e selecionar N grupos respeitando distância de 0,60 m; completar grupos faltantes com os máximos do SRP global. Comparar os dois raios com média simples de mapas e com o baseline atual. Esta é uma adaptação motivada por R1/R2, não SRP-NMF nem SBL completos.

**H3 — fonte dominante:** em vez de simplesmente remover um disco no mapa, localizar o primeiro máximo, calcular seus atrasos para cada par e atenuar a GCC por um notch gaussiano nesses atrasos. Testar larguras de 1 e 3 amostras e profundidades 0,5 e 1,0, total de quatro configurações; reconstruir o mapa para procurar as próximas fontes até N. Controlar duplicações e comparar fontes próximas/separadas. Rejeitar o método se o erro da primeira fonte se propagar ou apagar fontes distintas. A não linearidade PHAT torna isso uma heurística de de-emphasis a validar, não uma separação exata do áudio. Combinar H2 e H3 somente se ambos passarem individualmente e ainda houver tempo; no máximo uma combinação pré-fixada dos vencedores.

**H4 — rede com mais evidência:** executar somente se H2 elevar a cobertura oracle em ≥10 pontos percentuais ou reduzir o erro médio em ≥0,10 m nas 360 cenas, e o custo projetado couber. Entrada de seis canais: SRP global, quatro mapas agregados por banda e plano N/3. CNN com dilatações [1,2,4,8], canais 16/32/32/16, batch 16, AdamW, lr=0,001, weight_decay=0,0001. Ajustar normalizadores e peso BCE apenas no treino; peso positivo igual a metade da razão negativos/positivos do treino. Treinar do zero nas 6.000 cenas, seed 42, até 12 épocas, parada após três épocas sem melhora depois de cinco épocas mínimas; repetir só a configuração escolhida com seed 43. Comparar com uma CNN de dois canais treinada no mesmo subconjunto/orçamento para isolar o ganho das entradas. Reservar 30 minutos para preparar recursos e 30 minutos para cada uma das três execuções: controle seed 42, candidato seed 42 e candidato seed 43; total máximo de duas horas. Não comparar apenas com o treino antigo mais longo.

**Refinamento local:** só se o oracle/diagnóstico mostrar que o pico correto já é selecionado. Reavaliar o SRP físico em janelas ±0,20 m dos N candidatos com passo 0,025 m, mantendo candidatos distintos. Registrar ganho de centímetros separadamente da redução de falhas >1 m.

**H5 — refinamento subgrade:** comparar (a) coordenada na grade de 0,10 m, (b) interpolação quadrática local 2-D em torno do pico e (c) busca do SRP físico em ±0,20 m, passo 0,025 m, usando as GCCs já calculadas para a janela selecionada. Aplicar o mesmo refinamento a toda estimativa, sem consultar verdade-terreno. Medir especificamente a fração <0,10 m e as regressões >1 m; só manter refinamento que aumente a primeira sem piorar materialmente a cauda. Isso pode corrigir quantização quando o pico já está correto, mas não deve ser apresentado como solução para fontes perdidas.

### 6. Orçamento rígido e promoção

| Etapa | Teto de máquina |
|---|---:|
| Reprodução e diagnóstico/oracles/piloto físico | 0,75 h |
| H1: janela/escores | 0,75 h |
| H2: consenso tempo/frequência | 1,50 h |
| H3: de-emphasis | 1,00 h |
| H4: recursos/treinos curtos condicionais | 2,00 h |
| No máximo dois finalistas na validação completa | 2,00 h |
| Novo teste independente, geração + avaliação única | 2,00 h |
| Bootstrap, figuras, relatório e auditoria | 1,00 h |
| Reserva operacional | 1,00 h |
| **Total máximo** | **12,00 h** |

Medir inicialmente 30 cenas e 100 minibatches. Projetar o custo pelo caso M=8 e usar fator de segurança 1,5. Guardar espectros/GCC intermediários para reutilizar blocos/bandas e projeções geométricas; não repetir simulação nem todos os FFTs para cada parâmetro. Limitar RAM a 6 GiB e gravar tensores compactos, sem manter todos os mapas tempo/frequência da base em RAM. Tempo gasto em falhas também entra no orçamento. Relógio global de execução e limites de subprocessos devem impedir iniciar etapas sem tempo para avaliação e relatório; interromper treino salvando último checkpoint em limites de minibatch. Não repetir a ultrapassagem de 90 minutos por só verificar o relógio no fim da época.

Descartar candidatos que não superem o baseline da **mesma** triagem por ≥0,05 m sem piorar a fração >1 m, ou que excedam o custo previsto. Levar no máximo dois finalistas à validação completa de 3.000 cenas; reduzir número de candidatos se necessário, sem apresentar uma validação parcial como completa. O bloco H4 é opcional pelo gate e não expande a grade quando é pulado. Se nenhum método melhorar, registrar conclusão negativa e encerrar com o melhor baseline disponível.

### 7. Confirmação independente e critérios de entrega

- Congelar método, pesos, decodificador e normalizadores após a validação completa. Se for rede, exigir melhora nas duas seeds; selecionar a receita pela média, não pelo melhor seed isolado.
- Para a confirmação, buscar primeiro falantes/áudios nunca usados nos manifestos existentes. Se não existirem em quantidade suficiente, usar o corpus LibriSpeech `test-other` [R10], auditando interseções de IDs de falantes e áudios antes de gerar. Isso introduz uma mudança de material de fala; reportá-la explicitamente. Não chamar novas receitas com os mesmos áudios de teste independente de falantes.
- Gerar 600 cenas com 200 para cada N, aproximadamente balanceadas pelas 18 combinações; mesmas faixas acústicas e sala, novas receitas e seed 20261109. Registrar o manifesto antes de extrair mapas; nenhuma inspeção de desempenho até congelar os candidatos. Avaliar baseline atual e candidato nas mesmas cenas, uma única vez; sem retunar após abrir os resultados.
- Bootstrap pareado por **cena**, 10.000 reamostragens. Usar os mesmos índices reamostrados em todos os métodos e seeds, pois as seeds avaliam as mesmas cenas. Informar variação entre as duas seeds separadamente; duas seeds não quantificam toda a variabilidade de treino. ICs da validação usada para seleção continuam exploratórios.
- Além do gate da seção 1, registrar regressões por N/M/geometria e erro >1 m. Não ajustar regras usando RT60 verdadeiro, posições verdadeiras, identidade de fonte ou informações indisponíveis na inferência. Nenhum método pode descartar fontes/cenas ao calcular a média para aparentar melhora; abstenções devem ter cobertura e falhas reportadas.
- Se download/geração/extração do novo teste não couber nas duas horas, priorizar 180 cenas novas completas e declarar menor precisão estatística; se nem isso couber, entregar somente resultado exploratório de validação. Nunca usar o teste antigo como substituto silencioso.
- Entregar `artifacts/round3/`: protocolo congelado, manifestos/IDs, custos por etapa, erros e estimativas por cena, CSVs estratificados, comparações pareadas, pelo menos quatro sucessos/quatro falhas escolhidos por regra determinística e relatório com resultados negativos. Acrescentar ao Streamlit apenas candidatos que passem o gate; exibir erro por fonte, risco de falha grande e mapas temporais/bandas úteis ao diagnóstico.

### 8. Referências consultadas e limites de transferência

- **R1 — Thakallapalli, Gangashetty e Madhu (2021).** [NMF-weighted SRP for multi-speaker direction of arrival estimation](https://link.springer.com/article/10.1186/s13636-021-00201-y). Fala esparsa, agrupamento de frequências e arranjos com diferentes espaçamentos; métricas de direção não equivalem a erro posicional em metros.
- **R2 — Lai et al. (2024).** [Source Localization by Multidimensional SRP Mapping with Sparse Bayesian Learning](https://arxiv.org/abs/2405.11792). Sustenta conservar informação tempo-frequência e inferência esparsa; experimento com 16 sensores não prova desempenho com nossos 4/6/8 sensores.
- **R3 — [Multiple Source Localization Based on Acoustic Map De-Emphasis (2010)](https://link.springer.com/article/10.1155/2010/147495).** Motiva tratar a contribuição de fontes já localizadas, além de suprimir regiões próximas de picos.
- **R4 — Grondin e Glass (2019).** [Multiple Sound Source Localization with SVD-PHAT](https://www.isca-archive.org/interspeech_2019/grondin19_interspeech.pdf). Projeções ortogonais sucessivas para múltiplas fontes; arranjos compactos e resultados angulares. SVD comprime busca; compressão sozinha não garante precisão maior.
- **R5 — Grinstein et al. (Asilomar 2023; preprint 2024).** [The Neural-SRP Method for Positional Sound Source Localization](https://arxiv.org/abs/2403.09455). Evidência primária para redes que incorporam estrutura espacial/geometria. Nossa H4 será uma ablação própria de entradas multibanda, não Neural-SRP reproduzido.
- **R6 — Yang, Liu e Li (2022).** [SRP-DNN: Learning Direct-Path Phase Difference for Multiple Moving Sound Source Localization](https://arxiv.org/abs/2202.07859). Motiva melhorar evidência de caminho direto antes de construir o espectro espacial; treinamento integral não está prometido no bloco de duas horas.
- **R7 — Rafaely e Alhaiany.** [Speaker localization using direct path dominance test based on sound field directivity](https://arxiv.org/abs/2310.03688). Restrição a representação harmônica/arranjos esféricos impede usar a fórmula sem nova derivação.
- **R8 — Choi e Kim (1995).** [Spherical beam-forming and MUSIC methods for estimation of location and strength of spherical sound sources](https://doi.org/10.1006/mssp.1995.0043). Fundamenta steering de ondas esféricas para posição em campo próximo; não comprova robustez em nossa mistura reverberante.
- **R9 — [Multiple Sound Source Localisation with Steered Response Power Density and Hierarchical Grid Refinement (2018)](https://arxiv.org/abs/1803.01339).** O trabalho estima DOA com arranjo esférico rígido e harmônicos esféricos. Inspira a ideia de busca hierárquica, mas nosso refinamento local (x,y) será uma adaptação própria; não tratar resolução de grade como sinônimo de identificação correta de fontes.
- **R10 — [LibriSpeech/OpenSLR 12](https://www.openslr.org/12).** Fonte oficial das partições de áudio para construir holdout novo, sujeito à auditoria explícita de independência.

Os artigos demonstram possibilidades sob seus próprios protocolos. Não há evidência para prometer nesta máquina um erro de 10 cm, nem para transferir diretamente um percentual de ganho publicado ao nosso conjunto. O objetivo das 12 horas é testar as causas mais plausíveis, obter uma melhora material da cauda de erros e medir honestamente a generalização.

### 9. Execução do Round 3 — 2026-11-08

**Status: experimentos e confirmação independente concluídos; nenhum método promovido ao Streamlit.** O limite era 12 h de máquina; o ledger somou **4.531 s = 1,26 h**, incluindo tentativas que abortaram, download, extração, geração de dados, validações completas e teste. Restaram 10,74 h, mas não as usei para retunar após abrir o holdout. O relatório de execução e a auditoria detalhada estão em [`artifacts/round3/round3_report.md`](artifacts/round3/round3_report.md).

#### Implementação e triagem

- A extensão em `spatial.py` permite janela Hann/retangular e avaliação de GCC em pontos arbitrários; a janela padrão, grade e chamadas antigas continuam usando a configuração anterior.
- Baseline congelado em 360 cenas: fusão do Round 2, raio NMS 0,60 m e dois checkpoints de seeds 42/43. A CNN foi inferida separadamente por seed.
- **H1:** a janela retangular de 2.048 amostras foi a melhor das três variantes. Na triagem, a fusão rankeada teve média 1,038 m contra 1,090 m e 237 contra 247 erros >1 m; o ganho ainda era pequeno.
- **H2:** a média ponderada dos mapas tempo/frequência e os dois consensos foram piores, com média 1,37–1,69 m. Rejeitada.
- **H3:** o melhor notch (largura 3 amostras, profundidade 1) ficou quase igual ao baseline em média (1,093 vs. 1,090 m) e aumentou erros >1 m (265 vs. 247). Rejeitada.
- **H5, refinamento local:** no baseline de triagem, os acertos <10 cm subiram de 240/720 para 329/720, média de 1,090 para 1,077 m e mesma contagem >1 m (247). Aplicado depois da janela H1, deixou média 1,029 m, 330 fontes <10 cm e 240 erros >1 m. O ganho de precisão fina não corrigiu a cauda.
- O piloto com 18 receitas/36 fontes teve mediana 0,082 m no campo direto sem ruído, mas 8 erros >1 m; reverberação sem ruído: 16/36 acima de 1 m; mistura original: 16/36. Ruído por si só não explica os grandes desvios.
- H4, nova CNN multibanda, não foi executada: H2 não melhorou a cobertura oracle nem o erro em 360 cenas, condição predefinida para justificá-la. Wave-U-Net também ficou fora da rodada.

#### Validação completa e confirmação

A H1 com refinamento físico foi avaliada nas 3.000 cenas de validação para ambas as seeds. Frente à fusão corrente, o erro médio reduziu 0,076 m/seed42 e 0,077 m/seed43; P90 reduziu 0,091/0,109 m; a fração de erros >1 m caiu só 3,8%/3,1% em termos relativos. Aumentou acertos <10 cm de 1.943 para 2.810 e de 1.928 para 2.802 (de 6.000 fontes). Bootstrap pareado por cena, 10.000 amostras, apoiou a redução média na validação. São resultados exploratórios: a validação já participou da seleção.

Antes da inferência, congelei checkpoints, configuração, decodificador e protocolo do holdout. O conjunto externo contém 600 novas cenas de `LibriSpeech test-other` ([fonte oficial OpenSLR 12](https://www.openslr.org/12)), com 200 cenas por N. Os 2.939 áudios/33 falantes usados não se cruzam com treino, validação ou teste antigo; hash do manifesto: `e36a26a427e1f680ea193eca423fc8946b507387a11b404b7ffe300f981d0c15`. A seleção era speaker-disjoint e o holdout foi medido uma única vez, sem ajuste posterior.

Na confirmação, seed42 passou de média 1,119 para 1,070 m, mediana 0,312 para 0,245 m e 379 para **543 estimativas <0,10 m** entre 1.200 fontes; as falhas >1 m mudaram de 434 para 430. O P90, porém, subiu 0,048 m e o IC95% da diferença média inclui zero. Seed43 passou de média 1,151 para 1,079 m, P90 3,437 para 3,359 m, 385 para 533 acertos <0,10 m e 443 para 435 erros >1 m. A diferença média do seed43 foi −0,072 m (IC95% −0,130 a −0,015); do seed42, −0,049 m (−0,106 a 0,008). Há regressões em estratos externos, apesar do aumento de casos <10 cm.

O requisito adicional de **100 fontes únicas <0,10 m no holdout independente foi atendido**: o arquivo [`independent_lt_0_1m_cases.csv`](artifacts/round3/independent_lt_0_1m_cases.csv) exporta os 543 casos do seed42 com IDs de cena/fonte e coordenadas reais/estimadas. Em 197/600 cenas todas as fontes do seed42 ficaram abaixo de 0,10 m. A lista completa de fontes, incluindo falhas, está em [`independent_holdout_source_errors.csv`](artifacts/round3/independent_holdout_source_errors.csv).

#### Decisão de promoção

O candidato não atingiu o gate congelado: faltam pelo menos 0,074 m no ganho mínimo de média, pelo menos 0,141 m no ganho mínimo de P90 e a redução relativa de erros >1 m ficou muito aquém dos 20%. O seed42 também regrediu P90 no holdout e as tabelas externas têm pioras em alguns grupos N/M/SNR/RT60. **Não alterei o Streamlit.** A precisão local melhorou e o CSV materializa 543 exemplos úteis, mas as evidências não sustentam substituir o método atual enquanto estamos priorizando falhas grandes.

Todos os custos por etapa estão em [`artifacts/round3/machine_budget_ledger.csv`](artifacts/round3/machine_budget_ledger.csv); configurações/hash em `protocol.json`, `round3_finalist_frozen.json` e `holdout_manifest_frozen.json`; métricas, bootstrap, estratos e saídas de H1–H5 no diretório [`artifacts/round3`](artifacts/round3). Quatro exemplos de sucesso e quatro falhas selecionados por regra determinística estão em `independent_success_failure_examples.csv`.

#### Próximo foco recomendado

Não ampliar uma varredura de janelas nem treinar a CNN multibanda sem um novo gate. A evidência desta rodada aponta para identificar candidatos de fonte com consistência conjunta entre pares/canais e tratar a interação entre fontes dominantes/reflexões. Próxima hipótese: representação aprendida de evidência de caminho direto por pares, seguida de associação multi-fonte com N conhecido; comparar com um método de de-emphasis que preserve os atrasos near-field. Seleção e parâmetros devem ficar na validação, com novo corpus/speakers congelado antes da próxima avaliação independente.

## 10. Execução do Round 4 — validação de hipóteses em até 8 horas

**Data: 2026-11-08. Status: concluído; nenhum candidato promovido ao Streamlit.** O ledger estima 4.725 s (aprox. 1,31 h) de máquina contabilizada; as horas restantes não foram gastas em buscas amplas sem evidência. O relatório completo está em [`artifacts/round4/round4_report.md`](artifacts/round4/round4_report.md) e o custo por etapa em [`artifacts/round4/machine_budget_ledger_round4.csv`](artifacts/round4/machine_budget_ledger_round4.csv).

O principal candidato foi uma rede que aprende evidência de caminho direto por par de microfones. Na validação exploratória de 3.000 cenas/6.000 fontes, a média caiu 1,105→1,034 m, a mediana 0,251→0,122 m e os acertos <10 cm aumentaram 1.943→2.470. O P90 ficou praticamente igual (3,330→3,338 m), ainda houve 2.032 fontes >1 m (35,7%→33,9%) e a redução relativa dessa cauda foi só 5,2%. O bootstrap pareado por cena estimou Δ médio −0,071 m (IC95% −0,107 a −0,035 m). A validação já participara da seleção; os resultados são exploratórios.

O modelo melhorou sobretudo M=4; regrediu em M=8. A rota pós-hoc rede para M=4/6 e baseline para M=8 chegou a média 1,002 m, P90 3,240 m e 32,7% >1 m, porém não passou o gate congelado de Δ média ≥0,15 m, ganho P90 ≥0,25 m e redução relativa >1 m ≥20%. Por isso não deve ser tratada como candidata de implantação.

Treinei uma rede especializada em M=8 com 800 cenas. Na triagem congelada de 120 cenas/240 fontes, a média piorou 0,658→0,853 m, o P90 2,381→3,021 m e erros >1 m 20,4%→27,9%; IC95% bootstrap da diferença média por cena: +0,062 a +0,315 m. Rejeitada antes da avaliação completa.

Também foram rejeitados: escore de primeira reflexão (muito pior), fusões adicionais com SRP e decodificação TDOA conjunta (reduziu P90 em um caso mas aumentou >1 m). O teste M=8 não acessou `test-other`; a confirmação independente anterior do Round 3 não foi reaberta. Nenhuma alteração foi feita no Streamlit.

### Próximas hipóteses recomendadas

1. Auditar coordenadas reais, sincronização/offset entre canais, clipping, ganho, sample rate e velocidade do som; validar atrasos com fonte conhecida antes de retreinar.
2. Calcular cobertura oráculo: se a posição verdadeira aparece perto dos top-K máximos locais. Isso separa mapa acústico ruim de erro do decodificador/associação.
3. Testar poucas variantes de consenso robusto em GCC por par/quadro/banda, com confiabilidade e consistência near-field para rejeitar outliers. A decodificação conjunta simples já falhou em reduzir a cauda.
4. Priorizar N=2/3 para uma única hipótese de supressão iterativa/de-emphasis ou SRP esparso. Não iniciar Wave-U-Net sem prova de que separação preserva fase e melhora localização.
5. Se o arranjo puder mudar, comparar pareadamente M=4 com M=6/8 e otimizar posições via observabilidade 2D (Fisher/GDOP). No Round 4, M=4 foi o estrato mais difícil.

O relatório propõe alocação para uma próxima rodada de 8 h, com triagem obrigatória, no máximo um finalista de validação completa e reserva de contingência. Os detalhes e limites das referências científicas já pesquisadas permanecem nas seções 3 e 8 deste plano.

## 11. Execução do Round 5 — auditoria, cobertura e seleção multi-fonte

**Data:** 2026-11-08. **Objetivo:** reduzir simultaneamente média e P90. **Status:** quatro frentes concluídas; nenhum método promovido ao Streamlit. O relatório completo, os protocolos, erros por fonte, bootstraps e estratos estão em [`artifacts/round5/round5_report.md`](artifacts/round5/round5_report.md); os tempos estão em [`artifacts/round5/machine_budget_ledger_round5.csv`](artifacts/round5/machine_budget_ledger_round5.csv). O Round 5 consumiu aproximadamente 13.675 s (3,80 h) de máquina, dentro do orçamento de 12 h. `test-other` não foi aberto nesta rodada.

### Auditoria e cobertura de candidatos

Na triagem congelada de 360 cenas, os canais WAV tinham 16 kHz e 32.000 amostras; contagem de canais e SNR correspondiam ao manifesto. Áudio `FLOAT` acima de amplitude 1 não demonstra clipping. Os canais vêm de um único WAV multicanal simulado, então a auditoria não mede offsets de relógios de hardware reais. O proxy de GDOP aumentou conforme M caiu, e a média baseline foi 1,479 m para M=4, 1,038 m para M=6 e 0,548 m para M=8; dentro de cada M, a correlação GDOP/erro foi fraca (ρ=0,222, −0,055 e 0,025). O contraste M=4/6/8 também mistura geometria e distribuição das cenas.

O diagnóstico oráculo mostrou que o top-50 SRP+CNN contém um candidato a até 0,5 m para **92,2% das fontes** e para todas as fontes em **85,8% das cenas**. O decoder atual acerta ≤0,5 m em 55,5% das fontes. Como o oráculo associa usando a posição verdadeira, ele só demonstra potencial do pool; não é método inferível em uso. A lacuna aponta para seleção, reordenação e associação de candidatos, mais do que resolução de grade.

### Hipóteses avaliadas

- **Consenso robusto de GCC:** Huber em resposta bruta, mediana de ranks por par, Huber em ranks e média geométrica de ranks foram rejeitados. Todas pioraram a média sobre baseline 1,124 m; as médias candidatas ficaram entre 1,135 e 1,230 m e as taxas >1 m entre 35,8% e 39,2%. A menor variante de P90 não compensou a regressão média/cauda.
- **Agregação temporal:** média dos top-2 de oito blocos de 250 ms reduziu a média em validação de 3.000 cenas, 1,105→1,088 m (−1,8 cm; IC95% pareado −3,3 a −0,2 cm). O P90 caiu só 0,57 cm (3,330→3,324 m; IC95% −4,79 a +6,46 cm), portanto não há evidência de melhora confiável da cauda. Erros >1 m reduziram 0,45 pp, com IC incluindo zero.
- **Seleção TDOA conjunta:** top-32 candidatos espaciais com até cinco picos GCC-PHAT por par e N conhecido. Na validação exploratória de 3.000 cenas, a média piorou 1,105→1,115 m (Δ +0,0099 m; IC95% −0,0236 a +0,0431 m), enquanto o P90 melhorou pontualmente 7,24 cm (3,330→3,258 m; IC95% −2,48 a +17,23 cm, inclui zero). A taxa >1 m piorou significativamente 35,73→39,07% (Δ +3,33 pp; IC95% +2,14 a +4,54 pp) e a cobertura ≤0,5 m caiu 1,72 pp (IC95% −2,88 a −0,45 pp). Assim, nem a média nem P90 têm melhora estatisticamente estabelecida, e há regressão clara em erros grandes.

Os cortes exploratórios mostram TDOA melhor em média/P90 para M=4, mas pior em média para M=6 e M=8; para N=1, piorou média em 9,8 cm e P90 em 67,7 cm. Em N=2 a média/P90 melhoraram 3,2/24,1 cm, mas a taxa >1 m subiu. Esses grupos foram inspecionados na validação e não podem ser usados agora para selecionar uma rota condicional sem confirmação em dados novos.

### Decisão e próxima rodada

**Não promover** a agregação temporal ou o decoder TDOA ao Streamlit. Os quatro testes não encontraram uma melhoria simultânea e robusta para média e P90. Não foi feito teste de Wave-U-Net: a cobertura oráculo favorece primeiro melhorar a escolha de candidatos, e não há evidência de que separar fontes preserve a informação de fase e os TDOAs necessários.

Próxima hipótese com melhor custo-benefício: um reordenador leve do pool top-K, treinado para escolher conjuntos de N posições usando score SRP/CNN, resíduos TDOA e consistência entre pares. Criar antes um split novo de cenas/fontes, descontaminado por áudio e falante, sem reabrir `test-other`; congelar a receita antes do holdout. Triagem curta com média, P90 e taxa >1 m; avançar no máximo um finalista se os três objetivos forem promissores. Os achados por N/M deste Round 5 servem apenas para formular a hipótese, não para escolher limiares ou regras de implantação.

**Adendo posterior:** a hipótese foi separada em uma triagem de reordenador local mais simples e registrada no Round 6. Sete regras foram avaliadas no pool top-50 SRP em 3.000 cenas; nenhuma passou o critério de média/P90 sem piora na taxa >1 m. O método e a decisão estão resumidos nas seções seguintes.

## 12. Execução e promoção da Rodada 6 — geometria de microfones

**Data:** 2026-11-09. **Status:** validação independente concluída; opção “Círculo da Rodada 6” promovida à demonstração Streamlit, com M=8 como padrão. A tela usa a orientação exata do círculo testado, conserva o perímetro e o círculo aleatório para comparação e avisa que a evidência de P90 se aplica a M=8. A confirmação para SRP-PHAT não valida separadamente as saídas da CNN ou da Wave-U-Net.

Com 300 cenas por N×M e falantes LibriSpeech dev-other disjuntos, a geometria circular reduziu a média em M=4/6/8. Apenas M=8 teve redução confirmatória de P90 (3,107→2,904 m; ganho de 0,203 m, IC95% 0,042–0,343 m). Em M=4/6, o P90 não melhorou de forma conclusiva. A taxa >1 m caiu nos três M. Os dados gerados e as métricas por cena permanecem locais e não são versionados. O runner está em [`scripts/run_round6.py`](scripts/run_round6.py).

### Próximas hipóteses recomendadas

1. **Busca direta de raio e orientação**, partindo do círculo de 2 m. Otimizar a média e P90 do SRP real em cenas pareadas, não o proxy FIM/entropia usado nesta triagem. Fazer poucas variantes e reservar um split novo para confirmação.
2. **Ajuste generativo do mapa SRP para N fontes**, com esparsidade e rejeição da parte do mapa não explicada pelo modelo. É uma alternativa ao decoder de picos e ao reordenador rejeitado; há evidência publicada de redução de erro frente ao SRP-PHAT, mas precisa ser validada neste projeto e especialmente para N=2/3 ([Velasco et al., 2012](https://doi.org/10.3390/s121013781)).
3. **Diversidade espacial em reverberação alta**, combinando mapas SRP de subarranjos suficientemente separados. Um estudo com simulações e medidas reais relata robustez maior usando várias matrizes pequenas separadas; vale uma triagem estratificada por RT60, porque essa adaptação pode perder resolução ao reduzir os pares por subarranjo ([García-Barrios et al., 2023](https://doi.org/10.1016/j.apacoust.2022.109138)).

O método publicado de entropia para geometria SRP-PHAT usa uma sensibilidade espacial construída a partir de TDOAs/GCC e otimização sem derivadas; o proxy FIM simplificado testado nesta rodada não reproduz esse método. Ele é uma hipótese científica distinta para uma rodada futura ([Salvati, 2026](https://doi.org/10.1109/LSP.2026.3675914)). A otimização PSO de arranjos TDOA também fornece precedente metodológico, embora seu objetivo e cenário experimental difiram deste projeto ([Wang et al., 2019](https://doi.org/10.3390/s19194326)).
