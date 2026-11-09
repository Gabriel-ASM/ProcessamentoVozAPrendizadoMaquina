[CmdletBinding()]
param(
    [string]$EnvironmentName = "voz-gpu",
    [string]$CudaIndexUrl = "https://download.pytorch.org/whl/cu126",
    [switch]$SkipSmoke
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

function Find-Conda {
    $candidates = @()
    if ($env:CONDA_EXE) { $candidates += $env:CONDA_EXE }
    $candidates += @(
        (Join-Path $env:USERPROFILE "anaconda3\Scripts\conda.exe"),
        (Join-Path $env:USERPROFILE "miniconda3\Scripts\conda.exe"),
        "C:\ProgramData\Anaconda3\Scripts\conda.exe",
        "C:\ProgramData\Miniconda3\Scripts\conda.exe"
    )
    $fromPath = Get-Command conda -ErrorAction SilentlyContinue
    if ($fromPath) { $candidates += $fromPath.Source }
    foreach ($candidate in ($candidates | Where-Object { $_ } | Select-Object -Unique)) {
        if (Test-Path -LiteralPath $candidate) {
            return (Resolve-Path -LiteralPath $candidate).Path
        }
    }
    throw "Não encontrei conda. Execute este script no Anaconda Prompt ou ajuste CONDA_EXE."
}

function Invoke-Conda {
    param([Parameter(Mandatory)][string[]]$Arguments)
    & $script:conda @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Comando conda falhou (código $LASTEXITCODE): conda $($Arguments -join ' ')"
    }
}

function Invoke-EnvPython {
    param([Parameter(Mandatory)][string[]]$Arguments)
    Invoke-Conda @(@("run", "--no-capture-output", "-n", $script:EnvironmentName, "python") + $Arguments)
}

$script:conda = Find-Conda
$script:EnvironmentName = $EnvironmentName

Write-Host "Projeto: $projectRoot" -ForegroundColor Cyan
Write-Host "Conda:   $script:conda" -ForegroundColor Cyan
Write-Host "Ambiente: $EnvironmentName" -ForegroundColor Cyan

if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
    throw "nvidia-smi não foi encontrado. Instale/atualize o driver NVIDIA antes de habilitar CUDA."
}
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

$env:PYTHONPATH = Join-Path $projectRoot "src"
# O Anaconda deste computador carregou duas runtimes OpenMP ao combinar
# NumPy/SciPy com PyTorch. Estas variáveis evitam o erro durante esta execução.
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"

$envList = (& $script:conda env list --json | ConvertFrom-Json).envs
$environmentPath = $envList | Where-Object { (Split-Path -Leaf $_) -eq $EnvironmentName } | Select-Object -First 1
if (-not $environmentPath) {
    Write-Host "Criando $EnvironmentName a partir do ambiente base..." -ForegroundColor Yellow
    Invoke-Conda @("create", "-n", $EnvironmentName, "--clone", "base", "-y")
} else {
    Write-Host "Ambiente $EnvironmentName já existe; ele será reutilizado." -ForegroundColor Yellow
}

Write-Host "Atualizando ferramentas e dependências comuns..." -ForegroundColor Yellow
Invoke-EnvPython @("-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel")
Invoke-EnvPython @("-m", "pip", "install", "--upgrade", "numpy>=1.26", "scipy>=1.11", "soundfile>=0.12", "PyYAML>=6.0", "matplotlib>=3.8", "tensorboard>=2.14", "pytest>=8.0")

Write-Host "Instalando o projeto em modo editável..." -ForegroundColor Yellow
Invoke-EnvPython @("-m", "pip", "install", "--editable", $projectRoot, "--no-deps")

Write-Host "Instalando Pyroomacoustics..." -ForegroundColor Yellow
$pyroomInstalled = $true
try {
    Invoke-EnvPython @("-m", "pip", "install", "--upgrade", "pyroomacoustics>=0.7.7")
} catch {
    $pyroomInstalled = $false
    Write-Warning "Pyroomacoustics não foi instalado. O smoke anecoico continuará disponível, mas a simulação reverberante exigirá instalação manual."
}

Write-Host "Trocando PyTorch CPU pelo wheel CUDA..." -ForegroundColor Yellow
Invoke-EnvPython @("-m", "pip", "uninstall", "-y", "torch", "torchvision", "torchaudio")
Invoke-EnvPython @("-m", "pip", "install", "--upgrade", "torch", "torchvision", "torchaudio", "--index-url", $CudaIndexUrl)

Write-Host "Validando CUDA no PyTorch..." -ForegroundColor Yellow
$cudaCheck = 'import torch; assert torch.cuda.is_available(), "PyTorch não detectou CUDA"; print("torch=" + torch.__version__); print("cuda_build=" + str(torch.version.cuda)); print("device=" + torch.cuda.get_device_name(0)); print("vram_gb=" + str(round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)))'
Invoke-EnvPython @("-c", $cudaCheck)

Write-Host "Executando testes do projeto..." -ForegroundColor Yellow
Invoke-EnvPython @("-m", "pytest", "-q", $projectRoot)

if (-not $SkipSmoke) {
    Write-Host "Executando smoke test com treino curto..." -ForegroundColor Yellow
    Invoke-EnvPython @("-m", "voice_localization.cli", "smoke-test", "--config", (Join-Path $projectRoot "configs\smoke.yaml"))
}

Write-Host "" 
Write-Host "Ambiente CUDA pronto: $EnvironmentName" -ForegroundColor Green
Write-Host "Para usar depois: conda activate $EnvironmentName" -ForegroundColor Green
if (-not $pyroomInstalled) {
    Write-Warning "O ambiente GPU foi criado, mas Pyroomacoustics ficou pendente."
}
