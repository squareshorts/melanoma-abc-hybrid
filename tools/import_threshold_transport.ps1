# PowerShell: commit all 65 threshold-transport files before making a GitHub release.
# Example: .\\tools\\import_threshold_transport.ps1 -ArchivePath "$HOME\\Downloads\\Melanoma_threshold_transport_reproducibility_2026-10-18.zip"
param(
    [Parameter(Mandatory=$true)][string]$ArchivePath,
    [string]$ClonePath = ""
)
$ErrorActionPreference = "Stop"
$expected = "ce47b3c8e6136b1995a2ba0f2677a48dd30f956037dcecd42e7e20161047cf01"
$archive = (Resolve-Path -LiteralPath $ArchivePath).Path
$actual = (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant()
if ($actual -ne $expected) { throw "Archive checksum mismatch; stop before committing." }
if (-not $ClonePath) {
    $ClonePath = Join-Path ([IO.Path]::GetTempPath()) ("melanoma-abc-hybrid-" + [guid]::NewGuid().ToString("N").Substring(0,10))
}
if (Test-Path -LiteralPath $ClonePath) { throw "ClonePath already exists. Supply an unused directory." }
& git clone "https://github.com/squareshorts/melanoma-abc-hybrid.git" $ClonePath
if ($LASTEXITCODE -ne 0) { throw "Git clone failed. Check your Git configuration." }
$repo = (Resolve-Path -LiteralPath $ClonePath).Path
$destination = Join-Path $repo "reproducibility/threshold_transport_v2_1_0"
New-Item -ItemType Directory -Path $destination -Force | Out-Null
Expand-Archive -LiteralPath $archive -DestinationPath $destination -Force
$allFiles = @(Get-ChildItem -LiteralPath $destination -Recurse -File)
if ($allFiles.Count -ne 65) { throw "Expected 65 archive files; found $($allFiles.Count). Stop." }
foreach ($relative in @("README.md", "analysis/frozen_score_transport.py", "analysis/test_frozen_score_transport.py", "inputs/four_model_known_identity_corrected/bcn_external.csv", "comparator/outputs/training_features_float32.npy", "results/four_model_corrected/operating_performance.csv")) {
    if (-not (Test-Path -LiteralPath (Join-Path $destination $relative))) { throw "Required archive file missing: $relative" }
}
& git -C $repo add -- reproducibility/threshold_transport_v2_1_0
if ($LASTEXITCODE -ne 0) { throw "Git add failed." }
& git -C $repo diff --cached --stat
& git -C $repo commit -m "Add lesion-level threshold transport scores, scripts and outputs for v2.1.0"
if ($LASTEXITCODE -ne 0) { throw "Git commit failed. Configure your author name/email and retry in the cloned directory." }
& git -C $repo push origin main
if ($LASTEXITCODE -ne 0) { throw "Git push failed. Authenticate GitHub and retry from $repo. Do NOT create a release." }
$commit = (& git -C $repo rev-parse HEAD).Trim()
Write-Output "PUSHED COMMIT $commit"
Write-Output "Check https://github.com/squareshorts/melanoma-abc-hybrid/tree/main/reproducibility/threshold_transport_v2_1_0"
Write-Output "Then manually create v2.1.0 from that commit to trigger Zenodo."
