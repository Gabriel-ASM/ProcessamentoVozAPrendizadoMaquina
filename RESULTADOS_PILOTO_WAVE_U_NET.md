# Resultados do piloto Wave-U-Net

**Data:** 04/11/2026  
**Conclusão:** a linha Wave-U-Net é tecnicamente viável como ramo auxiliar, mas este piloto ainda não demonstra um ganho de localização consistente. Mantenha-a como hipótese experimental e decida após o experimento com LibriSpeech.

## Escopo e ambiente

O piloto usou 72 cenas com fala sintética e reverberação do Pyroomacoustics 0.10.1: 36 de treino, 18 de validação e 18 de teste. Cada split cobre N ∈ {1, 2, 3}, M ∈ {4, 6, 8} e duas geometrias. As cenas têm 2 s, 16 kHz e uma sala de 8 × 6 × 3 m. A fala sintética serve para verificar o pipeline; não substitui LibriSpeech nem gravações reais.

O ambiente foi Python 3.11.5, PyTorch 2.14.0+cu126 e CUDA na NVIDIA GeForce MX550 com 2 GB de VRAM. A Wave-U-Net do piloto usou 8 canais máximos, largura-base 8, profundidade 4 e lote 1; pico de memória GPU reportado: 35,8 MiB.

## O que foi implementado e auditado

- A Wave-U-Net recebe os canais de áudio, uma máscara dos microfones presentes e as coordenadas x/y normalizadas. Prevê uma correção residual de waveform. O alvo supervisionado é o áudio limpo **reverberante**, alinhado por canal à mistura; assim, o objetivo remove ruído sem exigir que a rede desfaça a resposta da sala.
- Neste pipeline a rede não separa os falantes: para N fontes, reconstrói a soma limpa reverberante nos canais. A localização continua a cargo de SRP-PHAT ou da CNN espacial que consome seus mapas.
- O ruído é dimensionado para o trecho final de 2 s que vai para o WAV. O SNR realizado é calculado sobre as amostras float32 efetivamente salvas.
- O gerador semeia os geradores interno e legado usados pelo Pyroomacoustics. As amostras de cenas repetidas são iguais; a comparação ignora o timestamp auxiliar que a biblioteca grava no bloco PEAK do cabeçalho WAV.
- As métricas de localização agregam erros por fonte. Os intervalos pareados reamostram cenas inteiras e levam junto todos os erros de fonte daquela cena.
- Os mapas têm shape 61 × 81. A auditoria conjunta aprovou 72/72 cenas, 72 receitas distintas, sem falantes compartilhados entre splits, sem WAVs inválidos e com pares limpo/ruidoso alinhados.

O SNR medido nos WAVs ficou entre 10,12–29,85 dB no treino, 10,75–29,62 dB na validação e 10,58–29,92 dB no teste. O RT60 realizado ficou entre 0,201–0,777 s no treino, 0,250–0,713 s na validação e 0,207–0,785 s no teste. O ajuste RT60 foi feito apenas com treino: versão 0.10.1, slope 1,4914, intercepto −0,1118 e RMSE residual 0,009 s. A configuração aplicada usa coeficientes muito próximos (1,4941 e −0,1130).

O check físico anecoico cobriu 18 casos: duas geometrias, 4/6/8 microfones e três posições conhecidas. O maior erro de TDOA foi 0,983 amostra e o maior erro de posição foi 4,8 × 10⁻⁷ m. Esse caso valida convenções de atraso e coordenadas; não mede robustez a fala, ruído ou reverberação.

Os 14 testes automatizados e o fluxo `smoke-test` completo passaram no ambiente CUDA. O check de aprendizado do CNN sobreajustou um conjunto fixo de 8 cenas em 500 passos: loss 1,660 → 0,361 e erro de treino 2,292 → 1,743 m. Isso valida o caminho de otimização, não a generalização.

## Localização na validação

Todos os métodos usaram as mesmas 18 cenas (36 fontes), posições de microfone, N conhecido e decodificador. O teste permaneceu reservado: seus WAVs e mapas foram auditados, mas suas métricas de localização não foram calculadas.

| Método | Erro médio (m) | Mediana (m) | P90 (m) | Até 0,50 m |
|---|---:|---:|---:|---:|
| SRP-PHAT sobre a mistura | 2,001 | 1,961 | 3,592 | 11,1% |
| CNN sobre a mistura | 2,396 | 2,282 | 4,327 | 11,1% |
| Wave-U-Net + SRP-PHAT | 1,944 | 1,886 | 3,493 | 11,1% |
| Wave-U-Net + CNN | 2,339 | 2,249 | 3,745 | 11,1% |

O Wave-U-Net reduziu o SI-SDR médio de 19,172 para 19,198 dB (+0,026 dB), uma mudança pequena. O L1 normalizado caiu de 0,171 para 0,093 no treino e de 0,124 para 0,093 na validação em cinco épocas. A redução de loss não se converteu em melhoria substancial do SI-SDR ou da localização.

## Comparações pareadas

Bootstrap por cena com 10.000 reamostragens. Δ = candidato − referência; para erro, valores negativos favorecem o candidato.

| Comparação | Δ erro médio (m), IC 95% | Δ P90 (m), IC 95% | Δ até 0,50 m |
|---|---:|---:|---:|
| Wave-U-Net + SRP vs. SRP | −0,058 [−0,193; 0,017] | −0,098 [−0,650; 0,094] | 0,0 pp |
| CNN mistura vs. SRP | +0,395 [0,032; 0,730] | +0,735 [−0,029; 2,011] | 0,0 pp |
| CNN Wave vs. CNN mistura | −0,058 [−0,269; 0,156] | −0,582 [−1,365; 0,780] | 0,0 pp |
| CNN Wave vs. SRP | +0,337 [0,008; 0,672] | +0,153 [−0,266; 1,759] | 0,0 pp |

Wave-U-Net + SRP apresenta um pequeno ganho pontual sobre SRP, mas os intervalos incluem ausência de ganho e a proporção dentro de 0,50 m não mudou. O CNN sobre mapas Wave ficou ligeiramente melhor que o CNN sobre a mistura, também sem diferença clara; ambos ficaram abaixo do SRP neste piloto. Com 18 cenas de validação sintéticas, esses números servem para escolher o próximo experimento, não para concluir desempenho final.

## Aplicação acadêmica e decisão

O Wave-U-Net original usa uma U-Net multiescala no domínio temporal para separação de fontes. Há extensões multicanal de fala que tratam explicitamente a interação entre microfones com atenção entre canais. A implementação deste projeto é uma adaptação compacta condicionada por geometria, não uma reprodução fiel desses artigos. Como a tarefa final é localização, a avaliação precisa medir as coordenadas estimadas: melhorar SI-SDR ou inteligibilidade, sozinho, não prova que as diferenças temporais e espaciais foram preservadas.

O caminho recomendado é prosseguir com Wave-U-Net como linha auxiliar, treinar em dados maiores com LibriSpeech, repetir com seeds 42/43/44 e selecionar pela validação. Só mantenha o ramo no pacote final se produzir uma melhoria pareada de erro de localização com incerteza aceitável sem comprometer as faixas de SNR, RT60, N, M e geometria. Os comandos e a sequência para isso estão em [INSTRUCOES_ETAPAS_6_A_8.md](INSTRUCOES_ETAPAS_6_A_8.md).

## Referências

- Stoller, Ewert e Dixon (2018), [*Wave-U-Net: A Multi-Scale Neural Network for End-to-End Audio Source Separation*](https://ismir2018.ircam.fr/doc/pdfs/205_Paper.pdf), ISMIR 2018.
- Ho et al. (2020), [*A Cross-Channel Attention-Based Wave-U-Net for Multi-Channel Speech Enhancement*](https://www.isca-archive.org/interspeech_2020/ho20_interspeech.html), Interspeech 2020.

## Artefatos para reprodução

- Auditoria dos áudios: `artifacts/wave_pilot/calibrated/scene_audio_audit.json`
- Calibração RT60 somente com treino: `artifacts/wave_pilot/calibrated/rt60_fit_train_only.json`
- Check físico SRP: `artifacts/wave_pilot/srp_physics_calibrated.json`
- Resultados brutos e bootstrap: `artifacts/wave_pilot/calibrated/`
- Checkpoints e históricos: `artifacts/calibrated_cnn_mixture_seed42/`, `artifacts/calibrated_wave_seed42/` e `artifacts/calibrated_cnn_wave_seed42/`
