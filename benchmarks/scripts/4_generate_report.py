#!/usr/bin/env python3
"""Run tokenization benchmarks.

Benchmarks different tokenization strategies on real code
and measures compression, speed, and quality metrics.

Usage:
    python benchmarks/scripts/3_run_benchmarks.py
"""

import csv
import time
from pathlib import Path
from typing import Dict, List, Tuple
import statistics

import bytepiece

# Configuration
BENCHMARK_ROOT = Path(__file__).parent.parent
DATASET_DIR = BENCHMARK_ROOT / "datasets"
MODELS_DIR = BENCHMARK_ROOT / "models"
RESULTS_DIR = BENCHMARK_ROOT / "results" / "raw"

# Number of runs for latency measurements
NUM_RUNS = 5


def benchmark_file(
    file_path: Path,
    encoder: bytepiece.BPEEncoder,
    model_name: str,
) -> Dict:
    """Benchmark tokenization of a single file."""
    # Read file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    num_chars = len(content)
    
    # Warm-up run
    _ = encoder.encode(content)
    
    # Timed runs
    times = []
    for _ in range(NUM_RUNS):
        start = time.perf_counter()
        tokens = encoder.encode(content)
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)  # Convert to ms
    
    # Get final tokenization
    tokens = encoder.encode(content)
    num_tokens = len(tokens)
    
    # Calculate metrics
    median_time = statistics.median(times)
    p90_time = statistics.quantiles(times, n=10)[8] if len(times) >= 10 else max(times)
    
    return {
        'file': file_path.name,
        'model': model_name,
        'chars': num_chars,
        'tokens': num_tokens,
        'compression_ratio': num_tokens / num_chars if num_chars > 0 else 0,
        'time_ms_median': median_time,
        'time_ms_p90': p90_time,
        'throughput_mb_s': (num_chars / 1024 / 1024) / (median_time / 1000) if median_time > 0 else 0,
    }


def benchmark_model(model_path: Path, dataset_dir: Path) -> List[Dict]:
    """Benchmark a model on a dataset."""
    model_name = model_path.parent.name
    
    print(f"  Benchmarking {model_name}...")
    
    # Load model
    encoder = bytepiece.load_model(str(model_path))
    
    # Get all files
    files = sorted(dataset_dir.glob("*.py"))[:50]  # Limit to 50 files for speed
    
    results = []
    for i, file_path in enumerate(files):
        if (i + 1) % 10 == 0:
            print(f"    Progress: {i+1}/{len(files)} files...")
        
        result = benchmark_file(file_path, encoder, model_name)
        results.append(result)
    
    # Summary statistics
    avg_compression = statistics.mean(r['compression_ratio'] for r in results)
    avg_throughput = statistics.mean(r['throughput_mb_s'] for r in results)
    
    print(f"    ✓ Avg compression: {avg_compression:.3f} tokens/char")
    print(f"    ✓ Avg throughput: {avg_throughput:.1f} MB/s")
    print()
    
    return results


def save_results(results: List[Dict], output_path: Path):
    """Save benchmark results to CSV."""
    if not results:
        return
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fieldnames = [
        'file', 'model', 'chars', 'tokens', 'compression_ratio',
        'time_ms_median', 'time_ms_p90', 'throughput_mb_s'
    ]
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"  ✓ Results saved to {output_path}")


def generate_summary(results: List[Dict]) -> Dict:
    """Generate summary statistics."""
    if not results:
        return {}
    
    # Group by model
    by_model = {}
    for r in results:
        model = r['model']
        if model not in by_model:
            by_model[model] = []
        by_model[model].append(r)
    
    # Calculate summary for each model
    summary = {}
    for model, model_results in by_model.items():
        summary[model] = {
            'files': len(model_results),
            'total_chars': sum(r['chars'] for r in model_results),
            'total_tokens': sum(r['tokens'] for r in model_results),
            'avg_compression': statistics.mean(r['compression_ratio'] for r in model_results),
            'median_compression': statistics.median(r['compression_ratio'] for r in model_results),
            'avg_throughput': statistics.mean(r['throughput_mb_s'] for r in model_results),
            'median_time_ms': statistics.median(r['time_ms_median'] for r in model_results),
            'p90_time_ms': statistics.median(r['time_ms_p90'] for r in model_results),
        }
    
    return summary


def print_summary(summary: Dict):
    """Print summary table."""
    print("\n" + "=" * 70)
    print("📊 BENCHMARK SUMMARY")
    print("=" * 70 + "\n")
    
    # Header
    print(f"{'Model':<20} {'Files':<8} {'Compression':<15} {'Throughput':<15}")
    print("-" * 70)
    
    # Sort by compression (lower is better)
    sorted_models = sorted(summary.items(), key=lambda x: x[1]['avg_compression'])
    
    for model, stats in sorted_models:
        comp = stats['avg_compression']
        throughput = stats['avg_throughput']
        
        print(f"{model:<20} {stats['files']:<8} {comp:.4f} tok/char   {throughput:.1f} MB/s")
    
    print("\n" + "=" * 70)
    print("\n💡 Lower compression ratio = better (fewer tokens per character)")
    print()


def main():
    """Main entrypoint."""
    print("\n" + "=" * 70)
    print("🏁 BytePiece Benchmarking")
    print("=" * 70 + "\n")
    
    # Check models exist
    models = list(MODELS_DIR.glob("*/model.json"))
    if not models:
        print("❌ Error: No trained models found!")
        print("   Run: python benchmarks/scripts/2_train_models.py")
        return
    
    print(f"Found {len(models)} trained models")
    print(f"Dataset: {DATASET_DIR / 'python'}")
    print()
    
    # Run benchmarks
    print("🔥 Running benchmarks...\n")
    
    all_results = []
    
    for model_path in sorted(models):
        results = benchmark_model(model_path, DATASET_DIR / "python")
        all_results.extend(results)
    
    # Save results
    print("\n💾 Saving results...")
    output_path = RESULTS_DIR / "benchmark_results.csv"
    save_results(all_results, output_path)
    print()
    
    # Generate and print summary
    summary = generate_summary(all_results)
    print_summary(summary)
    
    # Save summary
    summary_path = RESULTS_DIR / "summary.txt"
    with open(summary_path, 'w') as f:
        f.write("BytePiece Benchmark Summary\n")
        f.write("=" * 50 + "\n\n")
        
        for model, stats in sorted(summary.items(), key=lambda x: x[1]['avg_compression']):
            f.write(f"{model}:\n")
            f.write(f"  Files: {stats['files']}\n")
            f.write(f"  Total chars: {stats['total_chars']:,}\n")
            f.write(f"  Total tokens: {stats['total_tokens']:,}\n")
            f.write(f"  Avg compression: {stats['avg_compression']:.4f} tok/char\n")
            f.write(f"  Avg throughput: {stats['avg_throughput']:.1f} MB/s\n")
            f.write(f"  Median latency: {stats['median_time_ms']:.2f} ms\n")
            f.write("\n")
    
    print(f"✓ Summary saved to {summary_path}")
    
    print("\nNext steps:")
    print("  python benchmarks/scripts/4_generate_report.py")
    print()


if __name__ == "__main__":
    main()