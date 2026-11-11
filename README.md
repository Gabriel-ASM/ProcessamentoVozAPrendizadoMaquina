# Localização de múltiplas fontes de voz

Projeto experimental de processamento de voz para localizar múltiplas fontes sonoras em cenas simuladas. O pipeline gera cenas com posições conhecidas, calcula mapas espaciais SRP-PHAT e compara esse baseline com modelos de aprendizado de máquina.

## O que o projeto faz

- Simula salas, arranjos de microfones, reverberação e ruído.
- Gera áudio multicanal e rótulos espaciais reproduzíveis.
- Extrai mapas SRP-PHAT a partir de STFT e GCC-PHAT.
- Treina uma CNN para estimar posições no plano da sala, assumindo conhecido o número de fontes.
- Inclui um ramo experimental Wave-U-Net para aprimoramento de áudio antes da localização.
- Calcula métricas de localização e compara métodos nas mesmas cenas.

A Wave-U-Net é uma linha experimental auxiliar; ela não é, por si só, um localizador. O perfil `smoke` usa sinais sintéticos e um backend simplificado para verificar o pipeline, não para tirar conclusões sobre fala real.

## Estrutura

```text
configs/                 Configurações smoke, piloto e execução completa
scripts/                 Auditorias, análises e automação de experimentos
src/voice_localization/  Código do pipeline e da CLI
tests/                   Testes unitários e de integração
app.py                   Demonstração interativa em Streamlit
```

Dados e saídas não fazem parte do repositório: `data/`, `artifacts/`, `output/` e `tmp/` são ignorados pelo Git. Isso evita publicar corpora de áudio, cenas geradas, checkpoints, logs e relatórios exportados. PDFs e apresentações exportadas também são ignorados.

## Requisitos e instalação

Use Python 3.11. A instalação editável inclui as dependências de desenvolvimento:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

No Linux ou macOS, crie e ative o ambiente com:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

O arquivo `requirements-lock.txt` fixa versões usadas no ambiente de desenvolvimento com CPython 3.11 no Windows. A disponibilidade de versões de PyTorch e Pyroomacoustics pode variar entre sistemas.

## Primeira execução

Valide a configuração e execute o teste rápido do pipeline. Essa etapa usa dados sintéticos e não requer baixar um corpus:

```bash
python -m voice_localization.cli validate-config --config configs/smoke.yaml
python -m voice_localization.cli smoke-test --config configs/smoke.yaml
```

Para rodar a suíte automatizada:

```bash
python -m pytest -q
```

## Reproduzir os experimentos

As configurações em `configs/` incluem `smoke`, `pilot`, `full` e `wave_pilot`. O piloto e a execução completa usam as partições `train-clean-100`, `dev-clean` e `test-clean` do corpus LibriSpeech. Baixe e extraia os arquivos em `data/pilot/LibriSpeech/` ou `data/full/LibriSpeech/`. O corpus não é distribuído neste repositório; a ficha do [OpenSLR](https://www.openslr.org/12) informa licença CC BY 4.0 e disponibiliza os arquivos.

Exemplo de preparação e geração do piloto:

```bash
python -m voice_localization.cli prepare-corpus --root data/pilot/LibriSpeech --output data/pilot/manifests
python -m voice_localization.cli generate-scenes --config configs/pilot.yaml
python -m voice_localization.cli extract-maps --config configs/pilot.yaml
python -m voice_localization.cli validate-data --config configs/pilot.yaml
python -m voice_localization.cli train --config configs/pilot.yaml
```

O perfil `full` gera 30.000 cenas e exige bastante espaço em disco e tempo de processamento. Confira os recursos disponíveis antes de executá-lo. Os resultados de cada etapa ficam em `data/` e `artifacts/`, que são locais e ignorados pelo Git.

## Demonstração interativa

Instale a dependência opcional e inicie a aplicação:

```bash
python -m pip install -e ".[demo]"
streamlit run app.py
```

A demonstração depende da base completa, dos normalizadores e dos checkpoints treinados em `artifacts/`; esses arquivos não estão incluídos no repositório.

Por padrão, a tela usa M=8 e o arranjo circular fixo da Rodada 6. O seletor mantém o perímetro baseline e o círculo com orientação aleatória para comparação. A confirmação independente da Rodada 6 avaliou SRP-PHAT; os valores de CNN e Wave-U-Net exibidos para novas cenas continuam sendo comparações demonstrativas, não uma validação específica dessa geometria.

## Licença

Este repositório ainda não contém um arquivo de licença para o código. O corpus LibriSpeech é externo e mantém seus próprios termos de uso.
