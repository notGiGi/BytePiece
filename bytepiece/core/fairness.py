"""Fairness metrics for multilingual tokenization."""

from typing import Dict, List
import numpy as np


def compute_fertility(tokens: List[str], text: str) -> float:
    """Tokens per word ratio."""
    words = text.split()
    if not words:
        return 0.0
    return len(tokens) / len(words)


def compute_gini_coefficient(values: List[float]) -> float:
    """
    Gini coefficient: 0=perfect equality, 1=maximum inequality.
    Target: < 0.20 for fair tokenization.
    """
    if len(values) <= 1:
        return 0.0
    
    sorted_values = np.sort(np.array(values))
    n = len(sorted_values)
    cumsum = np.cumsum(sorted_values)
    
    numerator = 2 * np.sum((np.arange(1, n + 1)) * sorted_values)
    denominator = n * np.sum(sorted_values)
    
    if denominator == 0:
        return 0.0
    
    gini = (numerator / denominator) - (n + 1) / n
    return float(gini)


def compute_coverage_metrics(encoder, text: str, language: str = None) -> Dict[str, float]:
    """Compute comprehensive metrics."""
    tokens = encoder.encode(text)
    
    # Fertility
    fertility = compute_fertility(tokens, text)
    
    # Byte fallback rate
    byte_tokens = [t for t in tokens if '<0x' in t]
    fallback_rate = len(byte_tokens) / len(tokens) if tokens else 0.0
    
    # Character coverage
    chars_in_tokens = sum(len(t.replace('▁', '')) for t in tokens if '<0x' not in t)
    char_coverage = chars_in_tokens / len(text) if text else 0.0
    
    # Token length
    token_lengths = [len(t.replace('▁', '')) for t in tokens if '<0x' not in t]
    avg_token_length = np.mean(token_lengths) if token_lengths else 0.0
    
    return {
        'fertility': fertility,
        'fallback_rate': fallback_rate,
        'char_coverage': char_coverage,
        'avg_token_length': avg_token_length,
        'num_tokens': len(tokens),
        'num_words': len(text.split()),
    }


def compute_fairness_summary(language_fertilities: Dict[str, float]) -> Dict[str, float]:
    """Compute summary statistics."""
    if not language_fertilities:
        return {}
    
    values = list(language_fertilities.values())
    
    mean_val = np.mean(values)
    std_val = np.std(values)
    median_val = np.median(values)
    min_val = np.min(values)
    max_val = np.max(values)
    
    gini = compute_gini_coefficient(values)
    cv = std_val / mean_val if mean_val > 0 else 0.0
    max_min_ratio = max_val / min_val if min_val > 0 else float('inf')
    max_median_ratio = max_val / median_val if median_val > 0 else float('inf')
    
    return {
        'gini': float(gini),
        'cv': float(cv),
        'max_min_ratio': float(max_min_ratio),
        'max_median_ratio': float(max_median_ratio),
        'mean': float(mean_val),
        'std': float(std_val),
        'median': float(median_val),
        'min': float(min_val),
        'max': float(max_val),
    }


def print_fairness_report(language_metrics: Dict[str, Dict], summary: Dict[str, float]):
    """Print formatted report."""
    print("\n" + "=" * 80)
    print("FAIRNESS REPORT")
    print("=" * 80)
    
    sorted_langs = sorted(language_metrics.items(), key=lambda x: x[1]['fertility'])
    
    print(f"\n{'Language':<15} {'Fertility':<12} {'Fallback':<12} {'Coverage':<12}")
    print("-" * 80)
    
    for lang, metrics in sorted_langs:
        fert = metrics['fertility']
        fallback = metrics.get('fallback_rate', 0.0) * 100
        coverage = metrics.get('char_coverage', 0.0) * 100
        
        indicator = "✅" if fert < summary['median'] * 1.2 else ("⚠️ " if fert < summary['median'] * 1.5 else "❌")
        print(f"{indicator} {lang:<13} {fert:<12.2f} {fallback:<11.1f}% {coverage:<11.1f}%")
    
    print("-" * 80)
    print(f"\nSummary:")
    print(f"  Mean fertility:     {summary['mean']:.2f} tokens/word")
    print(f"  Gini coefficient:   {summary['gini']:.3f} ({'FAIR' if summary['gini'] < 0.20 else 'BIASED'})")
    print(f"  Max/Min ratio:      {summary['max_min_ratio']:.2f}x")
    print("=" * 80 + "\n")