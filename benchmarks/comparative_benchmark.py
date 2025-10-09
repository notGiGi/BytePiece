"""
Comparative Benchmark: 5 Tokenization Approaches

Compares:
1. Standard BPE (baseline)
2. Entropy-only (Shannon entropy, no syntax rules)
3. Syntax-only (preserve operators/keywords, no entropy)
4. Hybrid (Shannon + Syntax) - current implementation
5. Advanced (PMI + L/R + Syntax) - multi-signal

Run: python benchmarks/comparative_benchmark.py
"""

import time
import json
from pathlib import Path
from typing import List, Dict
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from bytepiece.core.normalizer import Normalizer, SpacerMode, PreTokenizationMode
from bytepiece.algorithms.bpe import train_bpe, BPEEncoder
from bytepiece.algorithms.entropy_bpe import EntropyAwareBPE


def load_corpus():
    """Load corpus"""
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


def evaluate_tokenizer(tokenizer, test_corpus: List[str], name: str) -> Dict:
    """Evaluate a tokenizer"""
    
    all_tokens = []
    total_chars = 0
    
    start = time.time()
    for code in test_corpus:
        tokens = tokenizer.encode(code)
        all_tokens.extend(tokens)
        total_chars += len(code)
    encode_time = time.time() - start
    
    # Decode byte tokens if needed
    import re
    def decode_byte(token):
        pattern = re.compile(r'<0x([0-9A-Fa-f]{2})>')
        matches = pattern.findall(token)
        if not matches:
            return token
        try:
            return bytes(int(h, 16) for h in matches).decode('utf-8', errors='ignore')
        except:
            return token
    
    decoded_tokens = [decode_byte(t) if '<0x' in t else t for t in all_tokens]
    
    # Metrics
    operators = ['>=', '<=', '==', '!=', '+=', '-=', '->', '=>', '||', '&&']
    operators_preserved = sum(1 for op in operators if op in decoded_tokens)
    
    keywords = ['def', 'class', 'if', 'else', 'for', 'while', 'return', 
                'lambda', 'async', 'await', 'try', 'except', 'import', 'from']
    keywords_preserved = sum(1 for kw in keywords if kw in decoded_tokens)
    
    return {
        'name': name,
        'total_tokens': len(all_tokens),
        'total_chars': total_chars,
        'compression_ratio': len(all_tokens) / total_chars,
        'tokens_per_second': len(all_tokens) / encode_time if encode_time > 0 else 0,
        'operators_preserved': operators_preserved,
        'operator_rate': operators_preserved / len(operators),
        'keywords_preserved': keywords_preserved,
        'keyword_rate': keywords_preserved / len(keywords),
        'unique_tokens': len(set(all_tokens)),
        'encode_time': encode_time,
    }


def print_comparison(results: Dict[str, Dict]):
    """Print comparison table"""
    
    print(f"\n{'='*100}")
    print("COMPARATIVE RESULTS: 5 CONFIGURATIONS")
    print(f"{'='*100}")
    
    # Metrics to compare
    metrics = [
        ('Total Tokens', 'total_tokens', 'lower'),
        ('Compression Ratio', 'compression_ratio', 'lower'),
        ('Tokens/Second', 'tokens_per_second', 'higher'),
        ('Operators Preserved', 'operators_preserved', 'higher'),
        ('Operator Rate', 'operator_rate', 'higher'),
        ('Keywords Preserved', 'keywords_preserved', 'higher'),
        ('Keyword Rate', 'keyword_rate', 'higher'),
        ('Unique Tokens', 'unique_tokens', 'context'),
    ]
    
    # Header
    configs = list(results.keys())
    print(f"\n{'Metric':<25}", end='')
    for config in configs:
        print(f"{config[:15]:>15}", end='')
    print("  Winner")
    print("-" * 100)
    
    # Winners tally
    winner_counts = {config: 0 for config in configs}
    
    for metric_name, key, direction in metrics:
        print(f"{metric_name:<25}", end='')
        
        values = [results[config][key] for config in configs]
        
        # Find winner
        if direction == 'lower':
            winner_idx = values.index(min(values))
        elif direction == 'higher':
            winner_idx = values.index(max(values))
        else:
            winner_idx = -1
        
        # Print values
        for i, val in enumerate(values):
            if isinstance(val, float):
                val_str = f"{val:.4f}"
            else:
                val_str = str(val)
            
            if i == winner_idx:
                print(f"{val_str:>15}", end='')
            else:
                print(f"{val_str:>15}", end='')
        
        if winner_idx >= 0:
            winner = configs[winner_idx]
            winner_counts[winner] += 1
            print(f"  🏆 {winner[:12]}")
        else:
            print("  -")
    
    print("\n" + "="*100)
    print("WINNER SUMMARY:")
    for config, count in sorted(winner_counts.items(), key=lambda x: -x[1]):
        print(f"  {config}: {count} wins")
    print("="*100)


def main():
    print("="*100)
    print(" "*30 + "COMPARATIVE BENCHMARK")
    print(" "*25 + "5 Tokenization Configurations")
    print("="*100)
    
    # Load corpus
    print("\n[1/6] Loading corpus...")
    corpus = load_corpus()
    
    split_idx = int(len(corpus) * 0.8)
    train_corpus = corpus[:split_idx]
    test_corpus = corpus[split_idx:]
    
    print(f"  Train: {len(train_corpus)}, Test: {len(test_corpus)}")
    
    vocab_size = 400  # Reasonable for this corpus
    
    results = {}
    
    # ========================================================================
    # CONFIG 1: Standard BPE (baseline)
    # ========================================================================
    print("\n[2/6] Training Standard BPE...")
    
    normalizer = Normalizer(
        spacer_mode=SpacerMode.NONE,
        pre_tokenization=PreTokenizationMode.WHITESPACE
    )
    
    start = time.time()
    vocab, merges, _ = train_bpe(
        train_corpus,
        vocab_size=vocab_size,
        normalizer=normalizer,
        byte_fallback=False,
        verbose=False
    )
    train_time = time.time() - start
    
    tokenizer_standard = BPEEncoder(vocab, merges, normalizer)
    
    print(f"  ✅ Trained in {train_time:.2f}s | Vocab: {len(vocab)}")
    
    results['Standard BPE'] = evaluate_tokenizer(tokenizer_standard, test_corpus, 'Standard BPE')
    results['Standard BPE']['train_time'] = train_time
    
    # ========================================================================
    # CONFIG 2: Entropy-only (no syntax rules)
    # ========================================================================
    # ========================================================================
    # CONFIG 2: Entropy-only
    # ========================================================================
    print("\n[3/6] Training Entropy-only (Shannon, no syntax)...")

    from bytepiece.algorithms.bpe_variants import EntropyOnlyBPE

    start = time.time()
    tokenizer_entropy = EntropyOnlyBPE(vocab_size=vocab_size, min_frequency=2)
    tokenizer_entropy.train(train_corpus, verbose=False)
    train_time = time.time() - start

    print(f"  ✅ Trained in {train_time:.2f}s | Vocab: {len(tokenizer_entropy.vocab)}")

    results['Entropy-Only'] = evaluate_tokenizer(tokenizer_entropy, test_corpus, 'Entropy-Only')
    results['Entropy-Only']['train_time'] = train_time

    # ========================================================================
    # CONFIG 3: Syntax-only
    # ========================================================================
    print("\n[4/6] Training Syntax-only...")

    from bytepiece.algorithms.bpe_variants import SyntaxOnlyBPE

    start = time.time()
    tokenizer_syntax = SyntaxOnlyBPE(vocab_size=vocab_size, min_frequency=2)
    tokenizer_syntax.train(train_corpus, verbose=False)
    train_time = time.time() - start

    print(f"  ✅ Trained in {train_time:.2f}s | Vocab: {len(tokenizer_syntax.vocab)}")

    results['Syntax-Only'] = evaluate_tokenizer(tokenizer_syntax, test_corpus, 'Syntax-Only')
    results['Syntax-Only']['train_time'] = train_time
    
    # TODO: This would require a variant without syntax rules
    # For now, skip or implement a simplified version
    print("  ⏭️  Skipped (requires entropy-only variant)")
    
    # ========================================================================
    # CONFIG 3: Syntax-only (no entropy)
    # ========================================================================
    print("\n[4/6] Training Syntax-only (operators/keywords preserved)...")
    
    # TODO: This would require a variant without entropy checks
    print("  ⏭️  Skipped (requires syntax-only variant)")
    
    # ========================================================================
    # CONFIG 4: Hybrid (Shannon + Syntax) - CURRENT IMPLEMENTATION
    # ========================================================================
    print("\n[5/6] Training Hybrid (Shannon + Syntax)...")
    
    start = time.time()
    tokenizer_hybrid = EntropyAwareBPE(vocab_size=vocab_size, min_frequency=2)
    tokenizer_hybrid.train(train_corpus, verbose=False)
    train_time = time.time() - start
    
    print(f"  ✅ Trained in {train_time:.2f}s | Vocab: {len(tokenizer_hybrid.vocab)}")
    
    results['Hybrid (Current)'] = evaluate_tokenizer(tokenizer_hybrid, test_corpus, 'Hybrid')
    results['Hybrid (Current)']['train_time'] = train_time
    
    # ========================================================================
    # CONFIG 5: Advanced (PMI + L/R + Syntax) - NEW
    # ========================================================================
    print("\n[6/6] Training Advanced (PMI + L/R + Syntax)...")
    
    # TODO: Implement EntropyAwareBPE variant with AdvancedAnalyzer
    print("  ⏭️  Coming soon (requires integration)")
    
    # ========================================================================
    # COMPARISON
    # ========================================================================
    print_comparison(results)
    
    # Save results
    output_dir = Path("benchmarks/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "comparative_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n✅ Results saved to benchmarks/results/comparative_results.json")
    print("\nNext steps:")
    print("  1. Implement entropy-only variant")
    print("  2. Implement syntax-only variant")
    print("  3. Integrate AdvancedAnalyzer into EntropyAwareBPE")
    print("  4. Re-run full comparison")


if __name__ == '__main__':
    main()