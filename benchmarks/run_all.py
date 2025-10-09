#!/usr/bin/env python3
"""Run complete benchmark pipeline.

This script runs all benchmark steps in sequence:
1. Prepare datasets
2. Train models
3. Run benchmarks
4. Generate report

Usage:
    python benchmarks/run_all.py
"""

import subprocess
import sys
from pathlib import Path

# Get absolute paths
BENCHMARK_ROOT = Path(__file__).parent.resolve()
SCRIPTS_DIR = BENCHMARK_ROOT / "scripts"


def run_script(script_name: str) -> bool:
    """Run a benchmark script."""
    script_path = SCRIPTS_DIR / script_name
    
    if not script_path.exists():
        print(f"\n❌ Error: Script not found: {script_path}")
        return False
    
    print(f"\n{'=' * 70}")
    print(f"Running: {script_name}")
    print('=' * 70)
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            check=True,
            cwd=str(BENCHMARK_ROOT.parent),  # Run from project root
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error running {script_name}")
        print(f"   Exit code: {e.returncode}")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠️  Interrupted by user")
        return False


def main():
    """Main entrypoint."""
    print("\n" + "=" * 70)
    print("🚀 BytePiece Complete Benchmark Pipeline")
    print("=" * 70)
    print("\nThis will run all benchmark steps:")
    print("  1. Prepare datasets (downloads code from GitHub)")
    print("  2. Train models (4 different tokenizers)")
    print("  3. Run benchmarks (measure quality & performance)")
    print("  4. Generate report (plots & markdown)")
    print("\nEstimated time: 5-10 minutes")
    print("\nPress Ctrl+C to cancel, or Enter to continue...")
    
    try:
        input()
    except KeyboardInterrupt:
        print("\n\nCancelled.")
        return
    
    # Run all steps
    steps = [
        "1_prepare_datasets.py",
        "2_train_models.py",
        "3_run_benchmarks.py",
        "4_generate_report.py",
    ]
    
    for step in steps:
        success = run_script(step)
        if not success:
            print(f"\n❌ Pipeline failed at step: {step}")
            sys.exit(1)
    
    # Success
    print("\n" + "=" * 70)
    print("✅ BENCHMARK PIPELINE COMPLETE!")
    print("=" * 70)
    print("\nResults:")
    print(f"  • Report: {BENCHMARK_ROOT / 'results' / 'reports' / 'benchmark_report.md'}")
    print(f"  • Plots: {BENCHMARK_ROOT / 'results' / 'plots' / '*.png'}")
    print(f"  • Raw data: {BENCHMARK_ROOT / 'results' / 'raw' / 'benchmark_results.csv'}")
    print("\nNext steps:")
    print("  • Review the markdown report")
    print("  • Add results to your README")
    print("  • Share on GitHub/Twitter")
    print()


if __name__ == "__main__":
    main()