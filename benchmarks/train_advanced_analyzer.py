"""
Train Advanced Entropy Analyzer
Collects statistics (PMI, L/R entropy) from corpus

Run: python benchmarks/train_advanced_analyzer.py
"""

import json
from pathlib import Path
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bytepiece.algorithms.entropy.advanced_analyzer import AdvancedEntropyAnalyzer


def load_corpus(path: str = "benchmarks/data/sample_code.txt") -> list:
    """Load training corpus"""
    corpus_path = Path(path)
    
    if not corpus_path.exists():
        # Fallback corpus
        base = [
            'def calculate(): return x >= 10',
            'if status >= 200: return True',
            'url = "https://api.example.com/users"',
            'for i in range(100): print(i)',
            'result += value if value > 0 else 0',
            'data = {"key": "value", "count": 42}',
            'class Calculator: pass',
            'async def fetch(): await api.get()',
            'lambda x: x * 2 if x > 0 else 0',
            'try: process() except Exception: pass',
            'while True: time.sleep(1)',
            'import os, sys, json',
            '@decorator def func(): return None',
            'with open("file") as f: data = f.read()',
            'assert x == y, "values must match"',
            'raise ValueError("invalid input")',
            'from typing import List, Dict, Optional',
            'config = {"host": "localhost", "port": 8080}',
            'response = requests.get(url, timeout=30)',
            'logger.info("Processing started")',
        ]
        return base * 20
    
    with open(corpus_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def simple_code_tokenizer(text: str) -> list:
    """
    Simple tokenizer that splits on whitespace + preserves operators
    (Similar to entropy pretokenizer but simpler)
    """
    import re
    
    # Pattern that splits but keeps operators
    pattern = re.compile(
        r'>=|<=|==|!=|\+=|-=|->|=>|//|<<|>>|\|\||&&|'  # Operators
        r'[a-zA-Z_][a-zA-Z0-9_]*|'  # Identifiers
        r'\d+\.?\d*|'  # Numbers
        r'"[^"]*"|'  # Strings
        r"'[^']*'|"  # Strings
        r'\S'  # Single chars
    )
    
    tokens = pattern.findall(text)
    return [t for t in tokens if t and not t.isspace()]


def main():
    print("="*80)
    print(" "*20 + "TRAIN ADVANCED ANALYZER")
    print("="*80)
    
    # Load corpus
    print("\n[1/4] Loading corpus...")
    corpus = load_corpus()
    print(f"  Loaded {len(corpus)} code samples")
    
    # Initialize analyzer
    print("\n[2/4] Initializing advanced analyzer...")
    analyzer = AdvancedEntropyAnalyzer(
        default_threshold=4.0,
        string_threshold=4.5,
        identifier_threshold=3.0,
        pmi_threshold=2.0
    )
    print("  ✅ Analyzer initialized")
    
    # Train on corpus
    print("\n[3/4] Training analyzer (collecting PMI + L/R entropy statistics)...")
    stats = analyzer.train_on_corpus(corpus, simple_code_tokenizer)
    
    print(f"\n  Statistics collected:")
    print(f"    Total tokens: {stats['total_tokens']}")
    print(f"    Unique tokens: {stats['unique_unigrams']}")
    print(f"    Unique bigrams: {stats['unique_bigrams']}")
    
    # Test the trained analyzer
    print("\n[4/4] Testing trained analyzer on sample tokens...")
    
    # Create context for testing
    all_tokens = []
    for text in corpus:
        all_tokens.extend(simple_code_tokenizer(text))
    
    test_cases = [
        ('def', 'calculate', '('),
        ('>=', '10', None),
        ('return', 'True', None),
        ('status', '>=', '200'),
        ('x', '*', '2'),
    ]
    
    print("\n" + "-"*80)
    print("SAMPLE DECISIONS (with multi-signal analysis)")
    print("-"*80)
    
    for i, (token, prev, next_tok) in enumerate(test_cases, 1):
        should_frag, reason, scores = analyzer.should_fragment(
            token, prev, next_tok, all_tokens
        )
        
        decision = "🔀 FRAGMENT" if should_frag else "✅ PRESERVE"
        
        print(f"\n[{i}] Token: '{token}' (prev: {prev}, next: {next_tok})")
        print(f"    Decision: {decision}")
        print(f"    Reason: {reason}")
        
        # Show all signals
        if 'shannon_entropy' in scores:
            print(f"    Shannon: {scores['shannon_entropy']:.3f}")
        if 'pmi_left' in scores:
            print(f"    PMI left: {scores['pmi_left']:.3f}")
        if 'pmi_right' in scores:
            print(f"    PMI right: {scores['pmi_right']:.3f}")
        if 'left_entropy' in scores:
            print(f"    L/R entropy: {scores['left_entropy']:.3f} / {scores['right_entropy']:.3f}")
    
    # Save trained analyzer
    output_dir = Path("benchmarks/models")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "advanced_analyzer_stats.json"
    
    # Save statistics
    save_data = {
        'stats': stats,
        'bigram_counts': {f"{k[0]}||{k[1]}": v for k, v in analyzer.bigram_counts.most_common(100)},
        'unigram_counts': {k: v for k, v in analyzer.unigram_counts.most_common(100)},
        'config': {
            'pmi_threshold': analyzer.pmi_threshold,
            'thresholds': analyzer.thresholds,
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(save_data, f, indent=2)
    
    print(f"\n✅ Trained analyzer statistics saved to {output_file}")
    print("="*80)
    print("\nNext steps:")
    print("  1. Run: python benchmarks/comparative_benchmark.py")
    print("  2. Compare 5 configurations (Standard, Entropy, PMI, etc.)")
    print("="*80)


if __name__ == '__main__':
    main()