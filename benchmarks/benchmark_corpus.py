"""
Benchmark Script: Test Entropy-Aware BPE on Real Python Corpus
Path: bytepiece/benchmarks/benchmark_corpus.py

Tests on 100+ Python files and compares:
- Standard BPE vs Entropy-Aware BPE
- Compression, operator preservation, speed
"""

import time
import json
from pathlib import Path
from typing import List, Dict
from collections import Counter
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from bytepiece.algorithms.bpe import BPEEncoder, train_bpe
from bytepiece.algorithms.entropy_bpe import EntropyAwareBPE


def collect_python_files(directory: Path, max_files: int = 100) -> List[str]:
    """
    Collect Python files from directory.
    
    Args:
        directory: Root directory to search
        max_files: Maximum number of files to collect
        
    Returns:
        List of file contents
    """
    print(f"Collecting Python files from {directory}...")
    
    python_files = []
    for py_file in directory.rglob("*.py"):
        if len(python_files) >= max_files:
            break
        
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
                if len(content) > 100:  # Skip very small files
                    python_files.append(content)
        except Exception as e:
            print(f"  Warning: Could not read {py_file}: {e}")
            continue
    
    print(f"✅ Collected {len(python_files)} Python files")
    return python_files


def measure_operator_preservation(tokens: List[str]) -> float:
    """
    Calculate % of operators preserved as single tokens.
    """
    operators = {
        '>=', '<=', '==', '!=', '->', '=>', '+=', '-=',
        '*=', '/=', '//', '<<', '>>', '||', '&&'
    }
    
    preserved_count = sum(1 for tok in tokens if tok in operators)
    
    # Count tokens that contain operators (possibly fragmented)
    total_operator_tokens = sum(1 for tok in tokens 
                                if any(op in tok for op in operators))
    
    return preserved_count / total_operator_tokens if total_operator_tokens > 0 else 1.0


def measure_keyword_preservation(tokens: List[str]) -> float:
    """Calculate % of keywords preserved as single tokens"""
    keywords = {
        'def', 'class', 'if', 'else', 'for', 'while',
        'return', 'import', 'from', 'try', 'except', 'with',
        'as', 'lambda', 'yield', 'async', 'await'
    }
    
    preserved_count = sum(1 for tok in tokens if tok in keywords)
    total_keywords = preserved_count  # Simplified
    
    return preserved_count / total_keywords if total_keywords > 0 else 1.0


def run_benchmark(train_corpus: List[str],
                  test_corpus: List[str],
                  vocab_size: int = 5000) -> Dict:
    """
    Run comprehensive benchmark comparing tokenizers.
    
    Args:
        train_corpus: Training data
        test_corpus: Test data
        vocab_size: Target vocabulary size
        
    Returns:
        Dictionary with results for both tokenizers
    """
    results = {
        'standard_bpe': {},
        'entropy_bpe': {},
        'comparison': {}
    }
    
    print("\n" + "=" * 80)
    print("BENCHMARK: Standard BPE vs Entropy-Aware BPE")
    print("=" * 80)
    print(f"\nCorpus size:")
    print(f"  Training: {len(train_corpus)} files")
    print(f"  Testing: {len(test_corpus)} files")
    print(f"  Target vocab: {vocab_size}")
    
    # ===== STANDARD BPE =====
    print("\n" + "=" * 80)
    print("[1/2] STANDARD BPE")
    print("=" * 80)
    
    print("\nTraining...")
    start_time = time.time()
    vocab, merge_rules, normalizer = train_bpe(
        train_corpus, 
        vocab_size=vocab_size,
        verbose=True
    )
    standard_bpe = BPEEncoder(vocab, merge_rules, normalizer)
    train_time_standard = time.time() - start_time
    
    print(f"✅ Training time: {train_time_standard:.2f}s")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    standard_results = evaluate_tokenizer(standard_bpe, test_corpus)
    standard_results['train_time'] = train_time_standard
    standard_results['vocab_size'] = len(vocab)
    
    results['standard_bpe'] = standard_results
    
    # ===== ENTROPY-AWARE BPE =====
    print("\n" + "=" * 80)
    print("[2/2] ENTROPY-AWARE BPE")
    print("=" * 80)
    
    entropy_bpe = EntropyAwareBPE(vocab_size=vocab_size)
    
    print("\nTraining...")
    start_time = time.time()
    entropy_bpe.train(train_corpus, verbose=True)
    train_time_entropy = time.time() - start_time
    
    print(f"✅ Training time: {train_time_entropy:.2f}s")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    entropy_results = evaluate_tokenizer(entropy_bpe, test_corpus)
    entropy_results['train_time'] = train_time_entropy
    entropy_results['vocab_size'] = len(entropy_bpe.vocab)
    
    results['entropy_bpe'] = entropy_results
    
    # ===== COMPARISON =====
    print("\n" + "=" * 80)
    print("RESULTS COMPARISON")
    print("=" * 80)
    
    metrics = [
        'compression_ratio',
        'operator_preservation',
        'keyword_preservation',
        'avg_tokens_per_file',
        'train_time'
    ]
    
    comparison = {}
    for metric in metrics:
        if metric not in standard_results or metric not in entropy_results:
            continue
        
        std_val = standard_results[metric]
        ent_val = entropy_results[metric]
        
        # Determine if lower or higher is better
        if metric in ['compression_ratio', 'train_time', 'avg_tokens_per_file']:
            # Lower is better
            improvement = (std_val - ent_val) / std_val * 100
        else:
            # Higher is better
            improvement = (ent_val - std_val) / std_val * 100 if std_val > 0 else 0
        
        comparison[metric] = {
            'standard': std_val,
            'entropy': ent_val,
            'improvement': improvement
        }
        
        print(f"\n📊 {metric.upper().replace('_', ' ')}:")
        print(f"  Standard BPE:      {std_val:.4f}")
        print(f"  Entropy-Aware BPE: {ent_val:.4f}")
        
        if improvement > 0:
            print(f"  Improvement:       +{improvement:.2f}% ✅")
        else:
            print(f"  Difference:        {improvement:.2f}%")
    
    results['comparison'] = comparison
    
    return results


def evaluate_tokenizer(tokenizer, test_corpus: List[str]) -> Dict:
    """
    Evaluate a tokenizer on test corpus.
    
    Returns:
        Dictionary with metrics
    """
    all_compressions = []
    all_op_preservations = []
    all_kw_preservations = []
    all_token_counts = []
    
    total_encode_time = 0
    
    for text in test_corpus:
        # Measure encoding time
        start = time.time()
        tokens = tokenizer.encode(text)
        total_encode_time += time.time() - start
        
        # Metrics
        all_compressions.append(len(tokens) / len(text) if len(text) > 0 else 0)
        all_op_preservations.append(measure_operator_preservation(tokens))
        all_kw_preservations.append(measure_keyword_preservation(tokens))
        all_token_counts.append(len(tokens))
    
    return {
        'compression_ratio': sum(all_compressions) / len(all_compressions),
        'operator_preservation': sum(all_op_preservations) / len(all_op_preservations),
        'keyword_preservation': sum(all_kw_preservations) / len(all_kw_preservations),
        'avg_tokens_per_file': sum(all_token_counts) / len(all_token_counts),
        'total_encode_time': total_encode_time,
        'files_per_second': len(test_corpus) / total_encode_time if total_encode_time > 0 else 0,
    }


def main():
    """Main benchmark runner"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Benchmark entropy-aware BPE on Python corpus')
    parser.add_argument('--corpus-dir', type=str, default='.',
                       help='Directory containing Python files')
    parser.add_argument('--max-files', type=int, default=100,
                       help='Maximum number of files to use')
    parser.add_argument('--vocab-size', type=int, default=5000,
                       help='Target vocabulary size')
    parser.add_argument('--output', type=str, default='benchmark_results.json',
                       help='Output file for results')
    
    args = parser.parse_args()
    
    # Collect corpus
    corpus_dir = Path(args.corpus_dir)
    all_files = collect_python_files(corpus_dir, max_files=args.max_files)
    
    if len(all_files) < 10:
        print("❌ Not enough Python files found. Need at least 10 files.")
        print(f"   Found {len(all_files)} files in {corpus_dir}")
        return
    
    # Split train/test (80/20)
    split_idx = int(0.8 * len(all_files))
    train_corpus = all_files[:split_idx]
    test_corpus = all_files[split_idx:]
    
    print(f"\n✅ Split: {len(train_corpus)} train, {len(test_corpus)} test")
    
    # Run benchmark
    results = run_benchmark(train_corpus, test_corpus, vocab_size=args.vocab_size)
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"✅ Results saved to {args.output}")
    print('='*80)
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    wins = sum(1 for data in results['comparison'].values() 
              if data['improvement'] > 0)
    total = len(results['comparison'])
    
    if wins > total / 2:
        print(f"\n🏆 Entropy-Aware BPE wins on {wins}/{total} metrics!")
    else:
        print(f"\n⚖️  Mixed results: {wins}/{total} metrics improved")
    
    print("\nTop improvements:")
    sorted_metrics = sorted(results['comparison'].items(), 
                           key=lambda x: x[1]['improvement'], 
                           reverse=True)
    for metric, data in sorted_metrics[:3]:
        if data['improvement'] > 0:
            print(f"  • {metric}: +{data['improvement']:.1f}%")


if __name__ == '__main__':
    main()