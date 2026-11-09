param(
    [string]$Config = "configs/full.yaml",
    [int[]]$Seeds = @(42, 43, 44),
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $ProjectRoot
try {
    $env:PYTHONPATH = Join-Path $ProjectRoot "src"

    function Invoke-Project([string[]]$Arguments) {
        & $Python @Arguments
        if ($LASTEXITCODE -ne 0) {
            throw "Comando falhou (exit $LASTEXITCODE): $($Arguments -join ' ')"
        }
    }

    Invoke-Project @("scripts\freeze_full_dataset.py", "--config", $Config)
    $DatasetFreezePath = Join-Path $ProjectRoot "artifacts\full\frozen_dataset\freeze.json"
    $DatasetFreeze = Get-Content -LiteralPath $DatasetFreezePath -Raw | ConvertFrom-Json
    foreach ($Seed in $Seeds) {
        Write-Host "=== Seed ${Seed}: CNN sobre mistura ==="
        $CnnMixtureDir = Join-Path $ProjectRoot "artifacts\cnn_mixture_seed$Seed"
        $CnnMixtureHistory = Join-Path $CnnMixtureDir "history.json"
        $CnnMixtureLast = Join-Path $CnnMixtureDir "last.pt"
        if ((Test-Path -LiteralPath $CnnMixtureHistory) -or (Test-Path -LiteralPath $CnnMixtureLast)) {
            Invoke-Project @("scripts\check_training_config.py", "--config", $Config, "--saved", (Join-Path $CnnMixtureDir "config.json"), "--seed", "$Seed", "--variant", "mixture")
        }
        if (-not (Test-Path -LiteralPath $CnnMixtureHistory)) {
            $TrainArgs = @("-m", "voice_localization.cli", "train", "--config", $Config, "--variant", "mixture", "--seed", "$Seed", "--run-name", "cnn_mixture_seed$Seed")
            if (Test-Path -LiteralPath $CnnMixtureLast) { $TrainArgs += @("--resume", $CnnMixtureLast) }
            Invoke-Project $TrainArgs
        }

        Write-Host "=== Seed ${Seed}: Wave-U-Net ==="
        $WaveRunDir = Join-Path $ProjectRoot "artifacts\wave_seed$Seed\wave_unet"
        $WaveHistory = Join-Path $WaveRunDir "history.json"
        $WaveLast = Join-Path $WaveRunDir "last.pt"
        if ((Test-Path -LiteralPath $WaveHistory) -or (Test-Path -LiteralPath $WaveLast)) {
            Invoke-Project @("scripts\check_training_config.py", "--config", $Config, "--saved", (Join-Path $WaveRunDir "config.json"), "--seed", "$Seed", "--variant", "mixture")
        }
        if (-not (Test-Path -LiteralPath $WaveHistory)) {
            $WaveTrainArgs = @("-m", "voice_localization.cli", "train-enhancer", "--config", $Config, "--seed", "$Seed", "--run-name", "wave_seed$Seed")
            if (Test-Path -LiteralPath $WaveLast) { $WaveTrainArgs += @("--resume", $WaveLast) }
            Invoke-Project $WaveTrainArgs
        }
        $WaveCheckpoint = Join-Path $ProjectRoot "artifacts\wave_seed$Seed\wave_unet\best.pt"
        if (-not (Test-Path -LiteralPath $WaveCheckpoint)) { throw "Checkpoint Wave-U-Net ausente: $WaveCheckpoint" }
        $WaveHash = (Get-FileHash -LiteralPath $WaveCheckpoint -Algorithm SHA256).Hash.ToLowerInvariant()
        $WaveVariant = "wave_$($WaveHash.Substring(0, 12))"

        foreach ($Split in @("train", "validation")) {
            Write-Host "=== Seed ${Seed}: enhancement de $Split ($WaveVariant) ==="
            Invoke-Project @("-m", "voice_localization.cli", "enhance-audio", "--config", $Config, "--checkpoint", $WaveCheckpoint, "--split", $Split)
            Invoke-Project @("-m", "voice_localization.cli", "extract-maps", "--config", $Config, "--split", $Split, "--variant", $WaveVariant, "--audio-variant", "enhanced")
        }

        Write-Host "=== Seed ${Seed}: CNN sobre mapas Wave ==="
        $CnnWaveDir = Join-Path $ProjectRoot "artifacts\cnn_wave_seed$Seed"
        $CnnWaveHistory = Join-Path $CnnWaveDir "history.json"
        $CnnWaveLast = Join-Path $CnnWaveDir "last.pt"
        if ((Test-Path -LiteralPath $CnnWaveHistory) -or (Test-Path -LiteralPath $CnnWaveLast)) {
            Invoke-Project @("scripts\check_training_config.py", "--config", $Config, "--saved", (Join-Path $CnnWaveDir "config.json"), "--seed", "$Seed", "--variant", $WaveVariant)
        }
        if (-not (Test-Path -LiteralPath $CnnWaveHistory)) {
            $CnnWaveTrainArgs = @("-m", "voice_localization.cli", "train", "--config", $Config, "--variant", $WaveVariant, "--seed", "$Seed", "--run-name", "cnn_wave_seed$Seed")
            if (Test-Path -LiteralPath $CnnWaveLast) { $CnnWaveTrainArgs += @("--resume", $CnnWaveLast) }
            Invoke-Project $CnnWaveTrainArgs
        }

        $ValidationDir = Join-Path $ProjectRoot "artifacts\final\validation\seed$Seed"
        New-Item -ItemType Directory -Force -Path $ValidationDir | Out-Null
        Invoke-Project @("-m", "voice_localization.cli", "evaluate-baseline", "--config", $Config, "--split", "validation", "--variant", "mixture", "--output", (Join-Path $ValidationDir "srp_validation.json"))
        Invoke-Project @("-m", "voice_localization.cli", "evaluate", "--config", $Config, "--checkpoint", (Join-Path $ProjectRoot "artifacts\cnn_mixture_seed$Seed\best.pt"), "--split", "validation", "--variant", "mixture", "--output", (Join-Path $ValidationDir "cnn_mixture_validation.json"))
        Invoke-Project @("-m", "voice_localization.cli", "evaluate-baseline", "--config", $Config, "--split", "validation", "--variant", $WaveVariant, "--output", (Join-Path $ValidationDir "wave_srp_validation.json"))
        Invoke-Project @("-m", "voice_localization.cli", "evaluate", "--config", $Config, "--checkpoint", (Join-Path $ProjectRoot "artifacts\cnn_wave_seed$Seed\best.pt"), "--split", "validation", "--variant", $WaveVariant, "--output", (Join-Path $ValidationDir "cnn_wave_validation.json"))

        $Comparisons = @(
            [PSCustomObject]@{ Reference = "srp_validation.json"; Candidate = "cnn_mixture_validation.json"; Output = "paired_srp_cnn_mixture.json" },
            [PSCustomObject]@{ Reference = "srp_validation.json"; Candidate = "wave_srp_validation.json"; Output = "paired_srp_wave.json" },
            [PSCustomObject]@{ Reference = "srp_validation.json"; Candidate = "cnn_wave_validation.json"; Output = "paired_srp_cnn_wave.json" },
            [PSCustomObject]@{ Reference = "cnn_mixture_validation.json"; Candidate = "cnn_wave_validation.json"; Output = "paired_cnn_wave.json" }
        )
        foreach ($Comparison in $Comparisons) {
            Invoke-Project @("scripts\compare_localization.py", "--reference", (Join-Path $ValidationDir $Comparison.Reference), "--candidate", (Join-Path $ValidationDir $Comparison.Candidate), "--output", (Join-Path $ValidationDir $Comparison.Output), "--samples", "2000", "--seed", "$Seed")
        }
        $FreezeRecord = [ordered]@{
            seed = $Seed
            wave_variant = $WaveVariant
            wave_checkpoint_sha256 = $WaveHash
            cnn_mixture_checkpoint_sha256 = (Get-FileHash -LiteralPath (Join-Path $ProjectRoot "artifacts\cnn_mixture_seed$Seed\best.pt") -Algorithm SHA256).Hash.ToLowerInvariant()
            cnn_wave_checkpoint_sha256 = (Get-FileHash -LiteralPath (Join-Path $ProjectRoot "artifacts\cnn_wave_seed$Seed\best.pt") -Algorithm SHA256).Hash.ToLowerInvariant()
            config_sha256 = (Get-FileHash -LiteralPath (Join-Path $ProjectRoot $Config) -Algorithm SHA256).Hash.ToLowerInvariant()
            dataset_freeze_sha256 = (Get-FileHash -LiteralPath $DatasetFreezePath -Algorithm SHA256).Hash.ToLowerInvariant()
            test_dataset_id = $DatasetFreeze.scene_dataset_ids.test
            mixture_normalizer_sha256 = (Get-FileHash -LiteralPath (Join-Path $ProjectRoot "data\full\maps\mixture\normalizer.npz") -Algorithm SHA256).Hash.ToLowerInvariant()
            wave_normalizer_sha256 = (Get-FileHash -LiteralPath (Join-Path $ProjectRoot "data\full\maps\$WaveVariant\normalizer.npz") -Algorithm SHA256).Hash.ToLowerInvariant()
            validation_directory = $ValidationDir
        }
        $FreezeRecord | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $ValidationDir "frozen_models.json") -Encoding UTF8
    }

    $GateArgs = @("scripts\check_validation_gate.py", "--validation-root", (Join-Path $ProjectRoot "artifacts\final\validation"), "--output", (Join-Path $ProjectRoot "artifacts\final\wave_validation_gate.json"), "--seeds")
    foreach ($Seed in $Seeds) { $GateArgs += "$Seed" }
    Invoke-Project $GateArgs
    Write-Host "Treino e seleção por validação concluídos. O conjunto de teste ainda não foi avaliado."
}
finally {
    Pop-Location
}
