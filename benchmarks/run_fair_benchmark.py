"""
Fair Benchmark: Standard BPE vs Entropy-Aware BPE
Both at character-level (no byte-fallback for normal chars)

Run: python benchmarks/run_fair_benchmark.py
"""

import time
from pathlib import Path
from bytepiece.core.normalizer import Normalizer, SpacerMode, PreTokenizationMode
from bytepiece.algorithms.bpe import train_bpe
from bytepiece.algorithms.entropy_bpe import EntropyAwareBPE


def load_corpus(path: str = "benchmarks/data/sample_code.txt") -> list:
    """Load code corpus"""
    corpus_path = Path(path)
    if not corpus_path.exists():
        # Expanded corpus with MORE diversity
        base_corpus = [
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
        # Less repetition = more vocab diversity
        return base_corpus * 10
    
    with open(corpus_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def estimate_vocab_size(corpus: list) -> int:
    """
    Estimate appropriate vocab size based on corpus.
    
    Rule of thumb: ~0.5x unique tokens after whitespace split
    """
    all_tokens = set()
    for line in corpus:
        tokens = line.split()
        all_tokens.update(tokens)
    
    # Target: Allow some merges but prevent full line memorization
    estimated = int(len(all_tokens) * 1.5)
    
    # Clamp to reasonable range
    return max(300, min(estimated, 2000))


def decode_byte_token(token: str) -> str:
    """Decode byte tokens like '<0x3E><0x3D>' → '>='"""
    import re
    byte_pattern = re.compile(r'<0x([0-9A-Fa-f]{2})>')
    matches = byte_pattern.findall(token)
    
    if not matches:
        return token
    
    try:
        byte_values = bytes(int(h, 16) for h in matches)
        return byte_values.decode('utf-8', errors='ignore')
    except:
        return token


def evaluate_tokenizer(tokenizer, test_corpus: list, name: str) -> dict:
    """Evaluate tokenizer with correct metrics"""
    
    print(f"\n{'='*80}")
    print(f"Evaluating {name}...")
    print(f"{'='*80}")
    
    all_tokens = []
    total_chars = 0
    
    # Encode all test samples
    for code in test_corpus:
        tokens = tokenizer.encode(code)
        all_tokens.extend(tokens)
        total_chars += len(code)
    
    # Decode tokens for analysis
    decoded_tokens = [decode_byte_token(t) if '<0x' in t else t for t in all_tokens]
    
    # Operators to check
    operators = ['>=', '<=', '==', '!=', '+=', '-=', '->', '=>', '||', '&&']
    
    # Count operators as single tokens
    operators_as_tokens = sum(1 for op in operators if op in decoded_tokens)
    
    # Count keywords
    keywords = ['def', 'class', 'if', 'else', 'for', 'while', 'return', 'lambda', 'async', 'await', 'try', 'except', 'import', 'from', 'with', 'assert', 'raise']
    keywords_as_tokens = sum(1 for kw in keywords if kw in decoded_tokens)
    
    # Calculate metrics
    results = {
        'name': name,
        'total_tokens': len(all_tokens),
        'total_chars': total_chars,
        'compression_ratio': len(all_tokens) / total_chars if total_chars > 0 else 0,
        'avg_token_length': total_chars / len(all_tokens) if all_tokens else 0,
        'operators_as_single_tokens': operators_as_tokens,
        'operator_preservation_rate': operators_as_tokens / len(operators),
        'keywords_as_single_tokens': keywords_as_tokens,
        'keyword_preservation_rate': keywords_as_tokens / len(keywords),
        'unique_tokens': len(set(all_tokens)),
        'sample_tokens': decoded_tokens[:20],
    }
    
    return results


def print_results(results_standard: dict, results_entropy: dict):
    """Print comparison results"""
    
    print(f"\n{'='*80}")
    print("FAIR COMPARISON RESULTS")
    print(f"{'='*80}")
    
    metrics = [
        ('Total Tokens', 'total_tokens', 'lower_is_better'),
        ('Compression Ratio', 'compression_ratio', 'lower_is_better'),
        ('Avg Token Length (chars)', 'avg_token_length', 'higher_is_better'),
        ('Operators as Single Tokens', 'operators_as_single_tokens', 'higher_is_better'),
        ('Operator Preservation Rate', 'operator_preservation_rate', 'higher_is_better'),
        ('Keywords as Single Tokens', 'keywords_as_single_tokens', 'higher_is_better'),
        ('Keyword Preservation Rate', 'keyword_preservation_rate', 'higher_is_better'),
        ('Unique Tokens in Output', 'unique_tokens', 'context'),
    ]
    
    print(f"\n{'Metric':<35} {'Standard BPE':>15} {'Entropy BPE':>15} {'Winner':>10}")
    print("-" * 85)
    
    winners = {'Standard': 0, 'Entropy': 0}
    
    for metric_name, key, direction in metrics:
        val_std = results_standard[key]
        val_ent = results_entropy[key]
        
        # Determine winner
        if direction == 'lower_is_better':
            winner = 'Entropy' if val_ent < val_std else 'Standard'
        elif direction == 'higher_is_better':
            winner = 'Entropy' if val_ent > val_std else 'Standard'
        else:
            winner = '-'
        
        if winner in winners:
            winners[winner] += 1
        
        # Format values
        if isinstance(val_std, float):
            val_std_str = f"{val_std:.4f}"
            val_ent_str = f"{val_ent:.4f}"
        else:
            val_std_str = f"{val_std}"
            val_ent_str = f"{val_ent}"
        
        winner_emoji = " 🏆" if winner != '-' else ""
        print(f"{metric_name:<35} {val_std_str:>15} {val_ent_str:>15} {winner:>10}{winner_emoji}")
    
    # Summary
    print("\n" + "="*85)
    print(f"SUMMARY: Entropy wins {winners['Entropy']}/7 metrics, Standard wins {winners['Standard']}/7")
    print("="*85)
    
    # Sample tokens comparison
    print(f"\n{'='*80}")
    print("SAMPLE TOKENS (first 10)")
    print(f"{'='*80}")
    
    print(f"\nStandard BPE:")
    for i, tok in enumerate(results_standard['sample_tokens'][:10], 1):
        preview = tok[:40] + "..." if len(tok) > 40 else tok
        print(f"  [{i}] '{preview}'")
    
    print(f"\nEntropy BPE:")
    for i, tok in enumerate(results_entropy['sample_tokens'][:10], 1):
        preview = tok[:40] + "..." if len(tok) > 40 else tok
        print(f"  [{i}] '{preview}'")


def main():
    """Run fair benchmark"""
    
    print("="*80)
    print(" "*20 + "FAIR BENCHMARK")
    print(" "*15 + "Standard BPE vs Entropy-Aware BPE")
    print("="*80)
    
    # Load corpus
    print("\n[1/5] Loading corpus...")
    corpus = load_corpus()
    print(f"  Loaded {len(corpus)} code samples")
    
    # Estimate appropriate vocab size
    estimated_vocab = estimate_vocab_size(corpus)
    print(f"  Estimated vocab size: {estimated_vocab}")
    
    # Split train/test
    split_idx = int(len(corpus) * 0.8)
    train_corpus = corpus[:split_idx]
    test_corpus = corpus[split_idx:]
    print(f"  Train: {len(train_corpus)}, Test: {len(test_corpus)}")
    
    # ============================================================================
    # TRAIN STANDARD BPE (WHITESPACE PRE-TOKENIZATION)
    # ============================================================================
    print("\n[2/5] Training Standard BPE (whitespace pre-tokenization)...")
    
    # Create normalizer with WHITESPACE pre-tokenization
    normalizer_whitespace = Normalizer(
        spacer_mode=SpacerMode.NONE,
        pre_tokenization=PreTokenizationMode.WHITESPACE  # Already in your code!
    )
    
    start = time.time()
    vocab_std, merges_std, normalizer_std = train_bpe(
        train_corpus,
        vocab_size=estimated_vocab,  # ← Use estimated vocab
        normalizer=normalizer_whitespace,
        verbose=False
    )
    time_std = time.time() - start
    
    print(f"  ✅ Trained in {time_std:.2f}s")
    print(f"  Vocab size: {len(vocab_std)}")
    print(f"  Merges learned: {len(merges_std)}")
    
    # Create encoder using BPEEncoder directly
    from bytepiece.algorithms.bpe import BPEEncoder
    tokenizer_std = BPEEncoder(
        vocab=vocab_std,
        merge_rules=merges_std,
        normalizer=normalizer_std
    )
    
    # ============================================================================
    # TRAIN ENTROPY-AWARE BPE
    # ============================================================================
    print("\n[3/5] Training Entropy-Aware BPE...")
    
    start = time.time()
    tokenizer_ent = EntropyAwareBPE(vocab_size=estimated_vocab, min_frequency=2)
    tokenizer_ent.train(train_corpus, verbose=False)
    time_ent = time.time() - start
    
    print(f"  ✅ Trained in {time_ent:.2f}s")
    print(f"  Vocab size: {len(tokenizer_ent.vocab)}")
    
    # ============================================================================
    # QUICK SANITY CHECK
    # ============================================================================
    print("\n[4/5] Sanity check on single sample...")
    test_line = "def calculate(): return x >= 10"
    
    tokens_std = tokenizer_std.encode(test_line)
    tokens_ent = tokenizer_ent.encode(test_line)
    
    print(f"  Input: '{test_line}'")
    print(f"  Standard BPE tokens: {tokens_std[:10]}... (total: {len(tokens_std)})")
    print(f"  Entropy BPE tokens: {tokens_ent[:10]}... (total: {len(tokens_ent)})")
    
    if len(tokens_std) == 1:
        print("  ⚠️  WARNING: Standard BPE memorized entire line as 1 token!")
        print("  This means vocab_size is still too large for corpus diversity.")
    
    # ============================================================================
    # EVALUATE BOTH
    # ============================================================================
    print("\n[5/5] Evaluating on test set...")
    
    results_std = evaluate_tokenizer(tokenizer_std, test_corpus, "Standard BPE")
    results_entropy = evaluate_tokenizer(tokenizer_ent, test_corpus, "Entropy-Aware BPE")
    
    # Add training time
    results_std['train_time'] = time_std
    results_entropy['train_time'] = time_ent
    
    # Print comparison
    print_results(results_std, results_entropy)
    
    # Save results
    import json
    output = {
        'standard_bpe': results_std,
        'entropy_bpe': results_entropy,
        'config': {
            'estimated_vocab_size': estimated_vocab,
            'train_samples': len(train_corpus),
            'test_samples': len(test_corpus),
            'standard_pre_tokenization': 'WHITESPACE',
            'standard_uses_spacer': False,
            'entropy_thresholds': tokenizer_ent.analyzer.thresholds,
        }
    }
    
    results_dir = Path('benchmarks/results')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    with open(results_dir / 'fair_comparison.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\n✅ Results saved to benchmarks/results/fair_comparison.json")
    print("="*80)


if __name__ == '__main__':
    main()