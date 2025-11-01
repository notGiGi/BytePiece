"""
FIXED REALISTIC BENCHMARK with reduced corpus size and progress reporting.

Key changes:
1. Reduced corpus: 1000 samples (10x faster, ~5-10 min total)
2. Reduced vocab: 2000 (5x faster per merge)
3. Progress reporting every 50 merges
4. Memory tracking
5. Uses optimized implementations

Run from project root:
    python benchmarks/benchmark_realistic_fixed.py
"""

import sys
import time
import gc
import psutil
import numpy as np
from collections import Counter
from typing import Dict, List, Tuple
from pathlib import Path
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import optimized implementations
from bytepiece.algorithms.bpe_fast import train_bpe_fast
from bytepiece.algorithms.info_fair_bpe_optimized import train_info_fair_bpe
from bytepiece.algorithms.bpe import BPEEncoder


class RealisticCorpusGenerator:
    """Generate realistic multilingual corpus with authentic morphology."""
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        self.seed = seed
        
        # Vocabularies (compact for faster generation)
        self.bases = {
            'English': ['cat', 'dog', 'house', 'tree', 'book', 'water', 'run', 'walk', 'big', 'small'],
            'German': ['Katze', 'Hund', 'Haus', 'Baum', 'Buch', 'Wasser', 'laufen', 'gehen', 'groß', 'klein'],
            'Turkish': ['kedi', 'köpek', 'ev', 'ağaç', 'kitap', 'su', 'koş', 'yürü', 'büyük', 'küçük'],
            'Finnish': ['kissa', 'koira', 'talo', 'puu', 'kirja', 'vesi', 'juosta', 'kävellä', 'iso', 'pieni'],
            'Russian': ['кот', 'собака', 'дом', 'дерево', 'книга', 'вода', 'бежать', 'идти', 'большой', 'маленький'],
            'Chinese': ['mao1', 'gou3', 'fang2', 'shu4', 'shui3', 'pao3', 'zou3', 'da4', 'xiao3', 'hao3'],
        }
        
        # Morphology patterns
        self.morphology = {
            'English': {'plural': ['s', 'es'], 'past': ['ed'], 'ing': ['ing']},
            'German': {'plural': ['en', 'e'], 'compound': True},
            'Turkish': {
                'plural': ['lar', 'ler'],
                'possessive': ['im', 'in'],
                'case': ['dan', 'de', 'a'],
            },
            'Finnish': {
                'plural': ['t'],
                'case': ['ssa', 'sta', 'han', 'lla'],
                'possessive': ['ni', 'si'],
            },
            'Russian': {'plural': ['ы', 'и'], 'case': ['а', 'у', 'ом']},
            'Chinese': {'particle': ['le', 'de', 'ma']},
        }
    
    def generate_word(self, lang: str) -> str:
        """Generate morphologically complex word."""
        base = np.random.choice(self.bases[lang])
        
        if lang == 'Turkish':
            # Turkish: agglutination
            if np.random.random() < 0.4:
                base += np.random.choice(self.morphology['Turkish']['plural'])
            if np.random.random() < 0.3:
                base += np.random.choice(self.morphology['Turkish']['possessive'])
            if np.random.random() < 0.3:
                base += np.random.choice(self.morphology['Turkish']['case'])
        
        elif lang == 'Finnish':
            # Finnish: cases
            if np.random.random() < 0.2:
                base += np.random.choice(self.morphology['Finnish']['plural'])
            if np.random.random() < 0.4:
                base += np.random.choice(self.morphology['Finnish']['case'])
        
        elif lang == 'German':
            # German: compounds
            if np.random.random() < 0.3 and self.morphology['German']['compound']:
                base += np.random.choice(self.bases['German'])
        
        elif lang == 'English':
            # English: simple suffixes
            if np.random.random() < 0.3:
                morph_type = np.random.choice(list(self.morphology['English'].keys()))
                base += np.random.choice(self.morphology['English'][morph_type])
        
        elif lang == 'Russian':
            # Russian: case
            if np.random.random() < 0.4:
                morph_type = np.random.choice(list(self.morphology['Russian'].keys()))
                base += np.random.choice(self.morphology['Russian'][morph_type])
        
        elif lang == 'Chinese':
            # Chinese: particles
            if np.random.random() < 0.2:
                base += np.random.choice(self.morphology['Chinese']['particle'])
        
        return base
    
    def generate_sentence(self, lang: str, length: int = None) -> str:
        """Generate a sentence."""
        if length is None:
            length = np.random.randint(4, 10)
        
        words = [self.generate_word(lang) for _ in range(length)]
        return ' '.join(words)
    
    def generate_corpus(
        self,
        num_samples: int = 1000,
        languages: List[str] = None
    ) -> Dict[str, List[str]]:
        """Generate corpus for each language."""
        if languages is None:
            languages = list(self.bases.keys())
        
        corpora = {}
        for lang in languages:
            sentences = [self.generate_sentence(lang) for _ in range(num_samples)]
            corpora[lang] = sentences
        
        return corpora


def compute_info_metrics(
    encoder: BPEEncoder,
    texts: List[str],
) -> Dict[str, float]:
    """Compute metrics for a language."""
    total_tokens = 0
    total_words = 0
    total_chars = 0
    
    for text in texts:
        words = text.split()
        tokens = encoder.encode(text)
        
        total_words += len(words)
        total_tokens += len(tokens)
        total_chars += len(text)
    
    # Character entropy
    char_counts = Counter()
    for text in texts:
        for char in text:
            char_counts[char] += 1
    
    total_char_count = sum(char_counts.values())
    char_entropy = 0.0
    for count in char_counts.values():
        p = count / total_char_count
        if p > 0:
            char_entropy -= p * np.log2(p)
    
    fertility = total_tokens / total_words if total_words > 0 else 0
    info_per_token = char_entropy / fertility if fertility > 0 else 0
    
    return {
        'fertility': fertility,
        'char_entropy': char_entropy,
        'info_per_token': info_per_token,
        'total_tokens': total_tokens,
        'total_words': total_words,
    }


def compute_fairness_metrics(metrics_by_lang: Dict[str, Dict]) -> Dict:
    """Compute fairness metrics."""
    fertilities = [m['fertility'] for m in metrics_by_lang.values()]
    infos = [m['info_per_token'] for m in metrics_by_lang.values()]
    
    return {
        'fertility_gap': max(fertilities) - min(fertilities),
        'info_gap': max(infos) - min(infos),
        'fertility_mean': np.mean(fertilities),
        'info_mean': np.mean(infos),
        'cost_index': sum(m['total_tokens'] for m in metrics_by_lang.values()),
    }


def measure_memory() -> float:
    """Memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def run_experiment(
    corpora: Dict[str, List[str]],
    vocab_size: int,
    fairness_weight: float,
    algorithm: str = "standard",
    verbose: bool = True
) -> Dict:
    """Run single experiment."""
    if verbose:
        print(f"\n{'='*80}")
        print(f"Experiment: {algorithm.upper()} | λ={fairness_weight:.1f} | vocab={vocab_size}")
        print(f"{'='*80}")
    
    gc.collect()
    mem_before = measure_memory()
    start_time = time.perf_counter()
    
    if algorithm == "standard":
        # Flatten corpus for standard BPE
        all_texts = []
        for texts in corpora.values():
            all_texts.extend(texts)
        
        vocab, merges, normalizer = train_bpe_fast(
            texts=all_texts,
            vocab_size=vocab_size,
            byte_fallback=False,
            verbose=True,  # Show progress
        )
        
        encoder = BPEEncoder(vocab, merges, normalizer)
        language_stats = None
    
    else:  # info_theoretic
        vocab, merges, normalizer, language_stats = train_info_fair_bpe(
            corpora=corpora,
            vocab_size=vocab_size,
            fairness_weight=fairness_weight,
            byte_fallback=False,
            verbose=True,  # Show progress
        )
        
        encoder = BPEEncoder(vocab, merges, normalizer)
    
    training_time = time.perf_counter() - start_time
    mem_after = measure_memory()
    mem_used = mem_after - mem_before
    
    # Evaluate on each language
    metrics_by_lang = {}
    for lang, texts in corpora.items():
        metrics = compute_info_metrics(encoder, texts)
        metrics_by_lang[lang] = metrics
    
    fairness = compute_fairness_metrics(metrics_by_lang)
    
    if verbose:
        print(f"\n✓ Training time: {training_time:.2f}s")
        print(f"✓ Memory used: {mem_used:.1f} MB")
        print(f"\nFairness metrics:")
        print(f"  Fertility gap: {fairness['fertility_gap']:.4f}")
        print(f"  Info/token gap: {fairness['info_gap']:.2f} bits ← KEY METRIC")
        print(f"  Cost index: {fairness['cost_index']:.0f} tokens")
    
    return {
        'algorithm': algorithm,
        'lambda': fairness_weight,
        'vocab_size': vocab_size,
        'training_time': training_time,
        'memory_mb': mem_used,
        'metrics_by_lang': metrics_by_lang,
        'fairness': fairness,
        'language_stats': language_stats,
    }


def print_detailed_results(result: Dict):
    """Print per-language results."""
    print(f"\nPer-language metrics:")
    print(f"{'Language':<12} | {'Fertility':<10} | {'Info/tok (bits)':<16} | {'Cost':<10}")
    print(f"{'-'*12}-+-{'-'*10}-+-{'-'*16}-+-{'-'*10}")
    
    english_tokens = result['metrics_by_lang'].get('English', {}).get('total_tokens', 1)
    
    for lang, metrics in result['metrics_by_lang'].items():
        rel_cost = metrics['total_tokens'] / english_tokens
        print(f"{lang:<12} | {metrics['fertility']:<10.2f} | "
              f"{metrics['info_per_token']:<16.2f} | {rel_cost:<10.2f}x")


def run_comprehensive_benchmark():
    """Run full benchmark with reduced corpus size."""
    print("="*80)
    print("COMPREHENSIVE REALISTIC BENCHMARK - FAST VERSION")
    print("Information-Theoretic Fair BPE")
    print("="*80)
    
    generator = RealisticCorpusGenerator(seed=42)
    
    print(f"\n{'='*80}")
    print("REALISTIC CORPUS (1000 samples/lang, complex morphology)")
    print(f"{'='*80}")
    
    print("\nGenerating corpus...")
    corpora = generator.generate_corpus(num_samples=1000)  # REDUCED from 10000
    
    print("\nCorpus statistics:")
    for lang, texts in corpora.items():
        total_words = sum(len(t.split()) for t in texts)
        avg_chars = np.mean([len(t) for t in texts])
        print(f"  {lang}: {len(texts)} sentences, {total_words} words, "
              f"avg {avg_chars:.1f} chars/sentence")
    
    vocab_size = 2000  # REDUCED from 5000
    fairness_weights = [0.0, 0.3]  # REDUCED: only test baseline + best
    
    results = []
    
    # Baseline
    print(f"\n{'='*40}")
    print("BASELINE: Standard BPE")
    print(f"{'='*40}")
    result = run_experiment(corpora, vocab_size, 0.0, "standard", verbose=True)
    print_detailed_results(result)
    results.append(result)
    
    # Fair BPE
    for lam in fairness_weights[1:]:
        print(f"\n{'='*40}")
        print(f"INFO-THEORETIC FAIR BPE: λ={lam:.1f}")
        print(f"{'='*40}")
        result = run_experiment(corpora, vocab_size, lam, "info_theoretic", verbose=True)
        print_detailed_results(result)
        results.append(result)
    
    # Analysis
    print(f"\n{'='*80}")
    print("COMPARATIVE ANALYSIS")
    print(f"{'='*80}")
    
    baseline = results[0]
    best = results[1]
    
    print(f"\nBaseline (Standard BPE):")
    print(f"  Info/token gap: {baseline['fairness']['info_gap']:.2f} bits")
    print(f"  Fertility gap: {baseline['fairness']['fertility_gap']:.4f}")
    print(f"  Training time: {baseline['training_time']:.2f}s")
    
    print(f"\nInfo-Theoretic Fair BPE (λ={best['lambda']:.1f}):")
    print(f"  Info/token gap: {best['fairness']['info_gap']:.2f} bits")
    print(f"  Fertility gap: {best['fairness']['fertility_gap']:.4f}")
    print(f"  Training time: {best['training_time']:.2f}s")
    
    info_improvement = (1 - best['fairness']['info_gap']/baseline['fairness']['info_gap']) * 100
    time_overhead = (best['training_time']/baseline['training_time'] - 1) * 100
    
    print(f"\nImprovements:")
    print(f"  ✓ Info gap reduction: {info_improvement:.1f}%")
    print(f"  ⚠ Time overhead: {time_overhead:+.1f}%")
    
    # Save results
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "benchmark_realistic_fast.json"
    
    with open(output_file, 'w') as f:
        # Convert numpy types for JSON
        def convert(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(item) for item in obj]
            return obj
        
        json.dump({'results': convert(results)}, f, indent=2)
    
    print(f"\n✓ Results saved to {output_file}")
    print(f"\n{'='*80}")
    print("BENCHMARK COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    run_comprehensive_benchmark()