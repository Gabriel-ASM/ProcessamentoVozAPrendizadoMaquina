param(
    [string]$Config = "configs/full.yaml",
    [int[]]$Seeds = @(42, 43, 44),
    [string]$Python = "python",
    [switch]$SkipFreezeCheck
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
    if (-not $SkipFreezeCheck) {
        foreach ($Seed in $Seeds) {
            $FreezePath = Join-Path $ProjectRoot "artifacts\final\validation\seed$Seed\frozen_models.json"
            if (-not (Test-Path -LiteralPath $FreezePath)) {
                throw "Validação/freeze ausente para seed $Seed. Rode run_training_seeds.ps1 primeiro."
            }
            $Freeze = Get-Content -LiteralPath $FreezePath -Raw | ConvertFrom-Json
            $CurrentConfigHash = (Get-FileHash -LiteralPath (Join-Path $ProjectRoot $Config) -Algorithm SHA256).Hash.ToLowerInvariant()
            if ($CurrentConfigHash -ne $Freeze.config_sha256) { throw "Configuração mudou depois do freeze da seed $Seed" }
            $CurrentDatasetFreezeHash = (Get-FileHash -LiteralPath $DatasetFreezePath -Algorithm SHA256).Hash.ToLowerInvariant()
            if ($CurrentDatasetFreezeHash -ne $Freeze.dataset_freeze_sha256) { throw "Manifestos/auditoria congelados mudaram depois do freeze da seed ${Seed}" }
            if ($DatasetFreeze.scene_dataset_ids.test -ne $Freeze.test_dataset_id) { throw "Receitas do teste diferem do freeze da seed ${Seed}" }
            $MixtureNormalizer = Join-Path $ProjectRoot "data\full\maps\mixture\normalizer.npz"
            $CurrentMixtureNormalizerHash = (Get-FileHash -LiteralPath $MixtureNormalizer -Algorithm SHA256).Hash.ToLowerInvariant()
            if ($CurrentMixtureNormalizerHash -ne $Freeze.mixture_normalizer_sha256) { throw "Normalizador de mistura mudou depois do freeze da seed $Seed" }
            $WaveCheckpoint = Join-Path $ProjectRoot "artifacts\wave_seed$Seed\wave_unet\best.pt"
            $MixtureCheckpoint = Join-Path $ProjectRoot "artifacts\cnn_mixture_seed$Seed\best.pt"
            $WaveCnnCheckpoint = Join-Path $ProjectRoot "artifacts\cnn_wave_seed$Seed\best.pt"
            foreach ($Entry in @(
                @{ Path = $WaveCheckpoint; Hash = $Freeze.wave_checkpoint_sha256 },
                @{ Path = $MixtureCheckpoint; Hash = $Freeze.cnn_mixture_checkpoint_sha256 },
                @{ Path = $WaveCnnCheckpoint; Hash = $Freeze.cnn_wave_checkpoint_sha256 }
            )) {
                if (-not (Test-Path -LiteralPath $Entry.Path)) { throw "Checkpoint ausente: $($Entry.Path)" }
                $Current = (Get-FileHash -LiteralPath $Entry.Path -Algorithm SHA256).Hash.ToLowerInvariant()
                if ($Current -ne $Entry.Hash) { throw "Checkpoint da seed $Seed mudou depois do freeze: $($Entry.Path)" }
            }
            $WaveNormalizer = Join-Path $ProjectRoot "data\full\maps\$($Freeze.wave_variant)\normalizer.npz"
            $CurrentWaveNormalizerHash = (Get-FileHash -LiteralPath $WaveNormalizer -Algorithm SHA256).Hash.ToLowerInvariant()
            if ($CurrentWaveNormalizerHash -ne $Freeze.wave_normalizer_sha256) { throw "Normalizador Wave mudou depois do freeze da seed $Seed" }
        }
    }

    foreach ($Seed in $Seeds) {
        $WaveCheckpoint = Join-Path $ProjectRoot "artifacts\wave_seed$Seed\wave_unet\best.pt"
        $WaveHash = (Get-FileHash -LiteralPath $WaveCheckpoint -Algorithm SHA256).Hash.ToLowerInvariant()
        $WaveVariant = "wave_$($WaveHash.Substring(0, 12))"
        $ResultDir = Join-Path $ProjectRoot "artifacts\final\seed$Seed"
        New-Item -ItemType Directory -Force -Path $ResultDir | Out-Null

        Write-Host "=== Seed ${Seed}: transformação de teste congelada ($WaveVariant) ==="
        Invoke-Project @("-m", "voice_localization.cli", "enhance-audio", "--config", $Config, "--checkpoint", $WaveCheckpoint, "--split", "test")
        Invoke-Project @("-m", "voice_localization.cli", "extract-maps", "--config", $Config, "--split", "test", "--variant", $WaveVariant, "--audio-variant", "enhanced")

        Write-Host "=== Seed ${Seed}: avaliação final no teste reservado ==="
        Invoke-Project @("-m", "voice_localization.cli", "evaluate-baseline", "--config", $Config, "--split", "test", "--variant", "mixture", "--output", (Join-Path $ResultDir "srp_test.json"))
        Invoke-Project @("-m", "voice_localization.cli", "evaluate", "--config", $Config, "--checkpoint", (Join-Path $ProjectRoot "artifacts\cnn_mixture_seed$Seed\best.pt"), "--split", "test", "--variant", "mixture", "--output", (Join-Path $ResultDir "cnn_mixture_test.json"))
        Invoke-Project @("-m", "voice_localization.cli", "evaluate-baseline", "--config", $Config, "--split", "test", "--variant", $WaveVariant, "--output", (Join-Path $ResultDir "wave_srp_test.json"))
        Invoke-Project @("-m", "voice_localization.cli", "evaluate", "--config", $Config, "--checkpoint", (Join-Path $ProjectRoot "artifacts\cnn_wave_seed$Seed\best.pt"), "--split", "test", "--variant", $WaveVariant, "--output", (Join-Path $ResultDir "cnn_wave_test.json"))

        $Freeze = Get-Content -LiteralPath (Join-Path $ProjectRoot "artifacts\final\validation\seed$Seed\frozen_models.json") -Raw | ConvertFrom-Json
        if ($Freeze.wave_variant -ne $WaveVariant) { throw "Wave map variant diverge do registro de freeze para seed $Seed" }
        $EvalRecord = [ordered]@{
            seed = $Seed
            wave_variant = $WaveVariant
            wave_checkpoint_sha256 = $WaveHash
            evaluation_split = "test"
            evaluation_timestamp_utc = [DateTime]::UtcNow.ToString("o")
            test_dataset_id = $DatasetFreeze.scene_dataset_ids.test
            dataset_freeze_sha256 = (Get-FileHash -LiteralPath $DatasetFreezePath -Algorithm SHA256).Hash.ToLowerInvariant()
            test_manifest = (Join-Path $ProjectRoot "data\full\manifests\scenes_test.jsonl")
        }
        $EvalRecord | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $ResultDir "evaluation_record.json") -Encoding UTF8
    }

    $ReportArgs = @("scripts\build_final_report.py", "--results-root", (Join-Path $ProjectRoot "artifacts\final"), "--output", (Join-Path $ProjectRoot "artifacts\final\report"), "--artifact-root", (Join-Path $ProjectRoot "artifacts"), "--map-root", (Join-Path $ProjectRoot "data\full\maps"), "--seeds")
    foreach ($Seed in $Seeds) { $ReportArgs += "$Seed" }
    Invoke-Project $ReportArgs
    Write-Host "Teste reservado concluído e relatório gerado em artifacts/final/report."
}
finally {
    Pop-Location
}
