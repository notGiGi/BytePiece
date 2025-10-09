# INSTALL_BENCHMARKS.ps1
# Complete automated setup for BytePiece benchmarks
# Run from project root: .\INSTALL_BENCHMARKS.ps1

Write-Host ""
Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host "  BytePiece Benchmark - Complete Installation" -ForegroundColor Cyan
Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host ""

# Verify location
if (-not (Test-Path "pyproject.toml")) {
    Write-Host "ERROR: Run from project root (where pyproject.toml is)" -ForegroundColor Red
    exit 1
}

# Create structure
Write-Host "[1/2] Creating structure..." -ForegroundColor Green
$dirs = @(
    "benchmarks\scripts",
    "benchmarks\datasets\python",
    "benchmarks\datasets\english",
    "benchmarks\models",
    "benchmarks\results\raw",
    "benchmarks\results\plots",
    "benchmarks\results\reports"
)
foreach ($d in $dirs) { 
    New-Item -ItemType Directory -Path $d -Force | Out-Null 
}
Write-Host "  Directories created" -ForegroundColor Gray

# Create empty files to show structure
Write-Host ""
Write-Host "[2/2] Creating files..." -ForegroundColor Green
Write-Host ""
Write-Host "Creating placeholder files..." -ForegroundColor Yellow
Write-Host "You MUST copy the actual Python code from the artifacts I showed you." -ForegroundColor Yellow
Write-Host ""

# Create placeholder Python files
$placeholderContent = @"
#!/usr/bin/env python3
# TODO: Replace this file with the actual code from Claude's artifact
# File: {0}

print("ERROR: This is a placeholder file!")
print("Please replace with actual code from the artifacts")
exit(1)
"@

$files = @{
    "benchmarks\run_all.py" = "run_all.py"
    "benchmarks\scripts\1_prepare_datasets.py" = "1_prepare_datasets.py"
    "benchmarks\scripts\2_train_models.py" = "2_train_models.py"
    "benchmarks\scripts\3_run_benchmarks.py" = "3_run_benchmarks.py"
    "benchmarks\scripts\4_generate_report.py" = "4_generate_report.py"
}

foreach ($path in $files.Keys) {
    $filename = $files[$path]
    ($placeholderContent -f $filename) | Out-File -FilePath $path -Encoding UTF8
    Write-Host "  Created placeholder: $path" -ForegroundColor DarkGray
}

# Create actual config files
@"
# Benchmark dependencies
matplotlib>=3.7.0
pandas>=2.0.0
"@ | Out-File -FilePath "benchmarks\requirements.txt" -Encoding UTF8

@"
datasets/
models/
results/raw/*.csv
!**/.gitkeep
!results/plots/*.png
!results/reports/*.md
"@ | Out-File -FilePath "benchmarks\.gitignore" -Encoding UTF8

@"
# BytePiece Benchmarks

Run: ``python benchmarks/run_all.py``

See BENCHMARK_SETUP_COMPLETE.md for details.
"@ | Out-File -FilePath "benchmarks\README.md" -Encoding UTF8

Write-Host "  Created config files" -ForegroundColor Gray

Write-Host ""
Write-Host "=" * 70 -ForegroundColor Yellow
Write-Host "  IMPORTANT: Copy Python Files from Artifacts!" -ForegroundColor Yellow
Write-Host "=" * 70 -ForegroundColor Yellow
Write-Host ""
Write-Host "Files created are PLACEHOLDERS. You must:" -ForegroundColor Red
Write-Host ""
Write-Host "  1. Scroll up to find the 5 Python code artifacts I created" -ForegroundColor Cyan
Write-Host "  2. Copy each one to the correct location:" -ForegroundColor Cyan
Write-Host ""
Write-Host "     benchmarks\run_all.py" -ForegroundColor White
Write-Host "     benchmarks\scripts\1_prepare_datasets.py" -ForegroundColor White
Write-Host "     benchmarks\scripts\2_train_models.py" -ForegroundColor White
Write-Host "     benchmarks\scripts\3_run_benchmarks.py" -ForegroundColor White
Write-Host "     benchmarks\scripts\4_generate_report.py" -ForegroundColor White
Write-Host ""
Write-Host "  3. Install dependencies (optional):" -ForegroundColor Cyan
Write-Host "     pip install -r benchmarks\requirements.txt" -ForegroundColor White
Write-Host ""
Write-Host "  4. Run:" -ForegroundColor Cyan
Write-Host "     python benchmarks\run_all.py" -ForegroundColor White
Write-Host ""
Write-Host "=" * 70 -ForegroundColor Green
Write-Host "  Structure ready! Now copy the Python code." -ForegroundColor Green
Write-Host "=" * 70 -ForegroundColor Green
Write-Host ""