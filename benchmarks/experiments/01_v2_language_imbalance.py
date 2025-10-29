#!/usr/bin/env python3
"""Experiment 01 v2: Language Imbalance (CORRECTED)."""

import sys
import json
import time
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import bytepiece
from bytepiece.algorithms.bpe import train_bpe, BPEEncoder
from bytepiece.core.fairness import (
    compute_coverage_metrics,
    compute_fairness_summary,
    print_fairness_report,
)

# Config
LANGUAGES = ['English', 'Spanish', 'French', 'German', 'Portuguese', 
             'Italian', 'Russian', 'Arabic', 'Chinese', 'Hindi']
DATA_DIR = PROJECT_ROOT / "benchmarks" / "data" / "multilingual"
RESULTS_DIR = PROJECT_ROOT / "benchmarks" / "results" / "language_balance"
VOCAB_SIZES = [1000, 5000, 10000]


def load_corpus(split='train'):
    """Load corpus from files."""
    split_dir = DATA_DIR / split
    
    if not split_dir.exists():
        raise FileNotFoundError(
            f"Corpus not found at {split_dir}\n"
            "Run: python benchmarks/data/build_corpus.py"
        )
    
    corpus = {}
    for lang in LANGUAGES:
        lang_file = split_dir / f"{lang.lower()}.txt"
        if lang_file.exists():
            with open(lang_file, 'r', encoding='utf-8') as f:
                corpus[lang] = [line.strip() for line in f if line.strip()]
    
    return corpus


def train_bpe_on_corpus(train_corpus, vocab_size, verbose=False):
    """Train BPE on combined corpus."""
    all_samples = []
    for samples in train_corpus.values():
        all_samples.extend(samples)
    
    if verbose:
        print(f"  Training on {len(all_samples)} samples...")
    
    vocab, merge_rules, normalizer = train_bpe(
        texts=all_samples,
        vocab_size=vocab_size,
        byte_fallback=True,
        verbose=False,
    )
    
    return BPEEncoder(vocab, merge_rules, normalizer)


def evaluate_on_test(encoder, test_corpus):
    """Evaluate on test set."""
    results = {}
    
    for lang, samples in test_corpus.items():
        combined_text = ' '.join(samples)
        metrics = compute_coverage_metrics(encoder, combined_text, lang)
        results[lang] = metrics
    
    return results


def run_experiment(vocab_size):
    """Run experiment for one vocab size."""
    print(f"\n{'='*80}")
    print(f"EXPERIMENT: vocab_size={vocab_size}")
    print(f"{'='*80}\n")
    
    # Load
    print("[1/3] Loading corpus...")
    train_corpus = load_corpus('train')
    test_corpus = load_corpus('test')
    print(f"  ✓ Train: {sum(len(s) for s in train_corpus.values())} samples")
    print(f"  ✓ Test: {sum(len(s) for s in test_corpus.values())} samples")
    
    # Train
    print("\n[2/3] Training BPE...")
    start_time = time.time()
    encoder = train_bpe_on_corpus(train_corpus, vocab_size, verbose=True)
    train_time = time.time() - start_time
    print(f"  ✓ Training complete ({train_time:.2f}s)")
    print(f"  ✓ Vocab size: {len(encoder.vocab)}\n")
    
    # Evaluate
    print("[3/3] Evaluating on test set...")
    results = evaluate_on_test(encoder, test_corpus)
    print(f"  ✓ Evaluation complete\n")
    
    # Results
    fertilities = {lang: m['fertility'] for lang, m in results.items()}
    summary = compute_fairness_summary(fertilities)
    print_fairness_report(results, summary)
    
    # Interpretation
    gini = summary['gini']
    if gini > 0.20:
        print("🚨 SIGNIFICANT BIAS DETECTED!")
        print(f"   Gini = {gini:.3f} > 0.20 threshold")
        print(f"   → Fair BPE is recommended\n")
    else:
        print("✅ Relatively balanced tokenization")
        print(f"   Gini = {gini:.3f} < 0.20 threshold")
        print(f"   → Consider imbalanced corpus or downstream eval\n")
    
    return {
        'vocab_size': vocab_size,
        'train_time': train_time,
        'results': {lang: {k: float(v) for k, v in m.items()} 
                    for lang, m in results.items()},
        'summary': {k: float(v) for k, v in summary.items()},
    }


def main():
    """Run all experiments."""
    print("\n" + "="*80)
    print("LANGUAGE BALANCE EXPERIMENT v2 (CORRECTED)")
    print(f"bytepiece v{bytepiece.__version__}")
    print("="*80)
    
    # Check corpus
    if not DATA_DIR.exists():
        print(f"\n❌ Corpus not found at {DATA_DIR}")
        print("Run: python benchmarks/data/build_corpus.py\n")
        return 1
    
    # Run experiments
    all_results = []
    for vocab_size in VOCAB_SIZES:
        result = run_experiment(vocab_size)
        all_results.append(result)
    
    # Save
    output = {
        'experiment': 'language_imbalance_v2',
        'bytepiece_version': bytepiece.__version__,
        'languages': LANGUAGES,
        'vocab_sizes': VOCAB_SIZES,
        'results_by_vocab_size': all_results,
    }
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "01_v2_results.json"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"{'='*80}")
    print(f"✅ Results saved to: {output_path}")
    print(f"{'='*80}\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())