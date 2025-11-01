"""
Benchmark: Standard BPE vs Parity-Aware BPE

Métricas estándar:
- Fertility (tokens/word)
- Gini coefficient (0-1, lower is better)
- Compression rate (tokens/char)

Compara DESPUÉS de training, no durante.
"""

import sys
import time
import numpy as np
from collections import Counter
from typing import Dict, List
from pathlib import Path
import json

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from bytepiece.algorithms.bpe_fast import train_bpe_fast
from bytepiece.algorithms.parity_aware_bpe import train_parity_aware_bpe, compute_gini_coefficient
from bytepiece.algorithms.bpe import BPEEncoder


class SimpleCorpusGenerator:
    """Generate simple multilingual corpus."""
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        
        self.bases = {
            'English': ['cat', 'dog', 'house', 'book', 'water', 'run', 'big', 'small'],
            'German': ['Katze', 'Hund', 'Haus', 'Buch', 'Wasser', 'laufen', 'groß', 'klein'],
            'Turkish': ['kedi', 'köpek', 'ev', 'kitap', 'su', 'koş', 'büyük', 'küçük'],
            'Finnish': ['kissa', 'koira', 'talo', 'kirja', 'vesi', 'juosta', 'iso', 'pieni'],
        }
        
        self.morphology = {
            'Turkish': {'plural': ['lar', 'ler'], 'case': ['dan', 'de']},
            'Finnish': {'case': ['ssa', 'sta'], 'plural': ['t']},
        }
    
    def generate_word(self, lang: str) -> str:
        base = np.random.choice(self.bases[lang])
        
        if lang == 'Turkish' and np.random.random() < 0.4:
            base += np.random.choice(self.morphology['Turkish']['plural'])
            if np.random.random() < 0.3:
                base += np.random.choice(self.morphology['Turkish']['case'])
        
        elif lang == 'Finnish' and np.random.random() < 0.3:
            base += np.random.choice(self.morphology['Finnish']['case'])
        
        return base
    
    def generate_corpus(
        self,
        num_samples: int = 1000,
        languages: List[str] = None
    ) -> Dict[str, List[str]]:
        if languages is None:
            languages = list(self.bases.keys())
        
        corpora = {}
        for lang in languages:
            sentences = []
            for _ in range(num_samples):
                length = np.random.randint(4, 8)
                words = [self.generate_word(lang) for _ in range(length)]
                sentences.append(' '.join(words))
            corpora[lang] = sentences
        
        return corpora


def evaluate_encoder(
    encoder: BPEEncoder,
    corpora: Dict[str, List[str]]
) -> Dict:
    """Evaluate encoder on corpora and compute metrics."""
    lang_stats = {}
    
    for lang, texts in corpora.items():
        total_tokens = 0
        total_words = 0
        total_chars = 0
        
        for text in texts:
            words = text.split()
            tokens = encoder.encode(text)
            
            total_words += len(words)
            total_tokens += len(tokens)
            total_chars += len(text)
        
        fertility = total_tokens / total_words if total_words > 0 else 0.0
        compression_rate = total_tokens / total_chars if total_chars > 0 else 0.0
        
        lang_stats[lang] = {
            'fertility': fertility,
            'total_tokens': total_tokens,
            'total_words': total_words,
            'compression_rate': compression_rate,
        }
    
    # Compute fairness metrics
    fertilities = [s['fertility'] for s in lang_stats.values()]
    gini = compute_gini_coefficient(fertilities)
    
    return {
        'per_language': lang_stats,
        'gini_coefficient': gini,
        'fertility_gap': max(fertilities) - min(fertilities),
        'fertility_mean': np.mean(fertilities),
        'total_cost': sum(s['total_tokens'] for s in lang_stats.values()),
    }


def run_experiment(
    corpora: Dict[str, List[str]],
    vocab_size: int,
    algorithm: str = "baseline"
) -> Dict:
    """Run single experiment."""
    print(f"\n{'='*60}")
    print(f"{algorithm.upper()}")
    print(f"{'='*60}")
    
    start = time.perf_counter()
    
    if algorithm == "baseline":
        # Standard BPE
        all_texts = []
        for texts in corpora.values():
            all_texts.extend(texts)
        
        vocab, merges, normalizer = train_bpe_fast(
            texts=all_texts,
            vocab_size=vocab_size,
            byte_fallback=False,
            verbose=True,
        )
    
    else:  # parity_aware
        vocab, merges, normalizer, train_stats = train_parity_aware_bpe(
            corpora=corpora,
            vocab_size=vocab_size,
            byte_fallback=False,
            verbose=True,
        )
    
    training_time = time.perf_counter() - start
    
    # Evaluate AFTER training
    encoder = BPEEncoder(vocab, merges, normalizer)
    eval_stats = evaluate_encoder(encoder, corpora)
    
    print(f"\n✓ Training time: {training_time:.2f}s")
    print(f"\nFairness metrics (evaluated):")
    print(f"  Gini coefficient: {eval_stats['gini_coefficient']:.4f}")
    print(f"  Fertility gap: {eval_stats['fertility_gap']:.4f}")
    print(f"  Mean fertility: {eval_stats['fertility_mean']:.4f}")
    print(f"  Total cost: {eval_stats['total_cost']:.0f} tokens")
    
    return {
        'algorithm': algorithm,
        'vocab_size': vocab_size,
        'training_time': training_time,
        'stats': eval_stats,
    }


def print_comparison(baseline: Dict, parity: Dict):
    """Print comparison table."""
    print(f"\n{'='*80}")
    print("COMPARISON")
    print(f"{'='*80}")
    
    print(f"\n{'Language':<12} | {'Baseline Fert':<14} | {'Parity Fert':<14} | {'Change':<10}")
    print(f"{'-'*12}-+-{'-'*14}-+-{'-'*14}-+-{'-'*10}")
    
    for lang in baseline['stats']['per_language'].keys():
        b_fert = baseline['stats']['per_language'][lang]['fertility']
        p_fert = parity['stats']['per_language'][lang]['fertility']
        change = ((p_fert - b_fert) / b_fert * 100) if b_fert > 0 else 0
        
        print(f"{lang:<12} | {b_fert:<14.4f} | {p_fert:<14.4f} | {change:+.1f}%")
    
    print(f"\n{'Metric':<20} | {'Baseline':<14} | {'Parity-Aware':<14} | {'Improvement':<10}")
    print(f"{'-'*20}-+-{'-'*14}-+-{'-'*14}-+-{'-'*10}")
    
    b_gini = baseline['stats']['gini_coefficient']
    p_gini = parity['stats']['gini_coefficient']
    gini_improvement = (1 - p_gini / b_gini) * 100 if b_gini > 0 else 0
    
    print(f"{'Gini coefficient':<20} | {b_gini:<14.4f} | {p_gini:<14.4f} | {gini_improvement:+.1f}%")
    
    b_gap = baseline['stats']['fertility_gap']
    p_gap = parity['stats']['fertility_gap']
    gap_improvement = (1 - p_gap / b_gap) * 100 if b_gap > 0 else 0
    
    print(f"{'Fertility gap':<20} | {b_gap:<14.4f} | {p_gap:<14.4f} | {gap_improvement:+.1f}%")
    
    b_cost = baseline['stats']['total_cost']
    p_cost = parity['stats']['total_cost']
    cost_change = ((p_cost - b_cost) / b_cost * 100) if b_cost > 0 else 0
    
    print(f"{'Total tokens':<20} | {b_cost:<14.0f} | {p_cost:<14.0f} | {cost_change:+.1f}%")


def main():
    print("="*80)
    print("BENCHMARK: Standard BPE vs Parity-Aware BPE")
    print("="*80)
    
    # Generate corpus
    generator = SimpleCorpusGenerator(seed=42)
    corpora = generator.generate_corpus(num_samples=1000)
    
    print("\nCorpus statistics:")
    for lang, texts in corpora.items():
        total_words = sum(len(t.split()) for t in texts)
        print(f"  {lang}: {len(texts)} sentences, {total_words} words")
    
    vocab_size = 2000
    
    # Run experiments
    baseline = run_experiment(corpora, vocab_size, "baseline")
    parity = run_experiment(corpora, vocab_size, "parity_aware")
    
    # Comparison
    print_comparison(baseline, parity)
    
    # Save results
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "parity_aware_comparison.json", 'w') as f:
        def convert(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(item) for item in obj]
            return obj
        
        json.dump({
            'baseline': convert(baseline),
            'parity_aware': convert(parity)
        }, f, indent=2)
    
    print(f"\n✓ Results saved to {output_dir / 'parity_aware_comparison.json'}")
    print(f"\n{'='*80}")
    print("BENCHMARK COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()