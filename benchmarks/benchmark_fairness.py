"""
Comprehensive benchmark for Fair BPE vs Standard BPE.

Tests language fairness, compression efficiency, and training performance.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

# Import from bytepiece package
from bytepiece.algorithms.fair_bpe import train_fair_bpe
from bytepiece import train_bpe, BPEEncoder, Normalizer


def generate_synthetic_corpus() -> Dict[str, List[str]]:
    """
    Generate synthetic multilingual corpus with varying morphological complexity.
    
    Simulates:
    - English: Simple morphology (low fertility)
    - Spanish: Moderate inflection (medium fertility)
    - German: Compound words (high fertility)
    - Turkish: Agglutinative (very high fertility)
    
    Returns:
        Dictionary mapping language -> list of sentences
    """
    
    # English: Simple morphology
    english_base = [
        "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
        "hello", "world", "test", "program", "computer", "science",
        "artificial", "intelligence", "machine", "learning", "neural", "network"
    ]
    english_suffixes = ["", "s", "ed", "ing", "er", "est"]
    
    # Spanish: More inflections
    spanish_base = [
        "el", "rápido", "marrón", "zorro", "salta", "sobre", "perro", "perezoso",
        "hola", "mundo", "prueba", "programa", "computadora", "ciencia",
        "artificial", "inteligencia", "máquina", "aprendizaje", "red", "neuronal"
    ]
    spanish_suffixes = ["", "o", "a", "os", "as", "ado", "ada", "ados", "adas", "ando", "iendo"]
    
    # German: Compound words
    german_base = [
        "der", "schnell", "braun", "fuchs", "springt", "über", "hund", "faul",
        "hallo", "welt", "test", "programm", "computer", "wissenschaft",
        "künstlich", "intelligenz", "maschine", "lernen", "neuron", "netzwerk"
    ]
    german_compounds = ["", "s", "es", "en", "er", "ung", "heit", "keit", "lich", "isch"]
    
    # Turkish: Agglutinative (many suffixes)
    turkish_base = [
        "hızlı", "kahverengi", "tilki", "atlar", "üzerinde", "tembel", "köpek",
        "merhaba", "dünya", "test", "program", "bilgisayar", "bilim",
        "yapay", "zeka", "makine", "öğrenme", "sinir", "ağ"
    ]
    turkish_suffixes = [
        "", "lar", "ler", "im", "in", "i", "de", "den", "e", "dan",
        "lik", "lık", "luk", "lük", "ci", "cı", "cu", "cü"
    ]
    
    def generate_sentences(base: List[str], suffixes: List[str], count: int) -> List[str]:
        """Generate sentences by combining base words with suffixes."""
        sentences = []
        for i in range(count):
            # Pick 5-10 random words
            sentence_length = 5 + (i % 6)
            words = []
            for j in range(sentence_length):
                word_idx = (i * j + j) % len(base)
                suffix_idx = (i + j * 3) % len(suffixes)
                word = base[word_idx] + suffixes[suffix_idx]
                words.append(word)
            sentences.append(" ".join(words))
        return sentences
    
    return {
        "English": generate_sentences(english_base, english_suffixes, 500),
        "Spanish": generate_sentences(spanish_base, spanish_suffixes, 500),
        "German": generate_sentences(german_base, german_compounds, 500),
        "Turkish": generate_sentences(turkish_base, turkish_suffixes, 500),
    }


def compute_fertility(texts: List[str], encoder: BPEEncoder) -> float:
    """
    Compute average fertility (tokens per word) for a set of texts.
    
    Args:
        texts: List of text strings
        encoder: BPE encoder
        
    Returns:
        Average fertility
    """
    total_tokens = 0
    total_words = 0
    
    for text in texts:
        tokens = encoder.encode(text)
        words = text.split()
        
        total_tokens += len(tokens)
        total_words += len(words)
    
    return total_tokens / total_words if total_words > 0 else 0.0


def compute_gini_coefficient(fertilities: List[float]) -> float:
    """
    Compute Gini coefficient for fertility distribution.
    
    Gini = 0 means perfect equality
    Gini = 1 means maximum inequality
    
    Args:
        fertilities: List of fertility values
        
    Returns:
        Gini coefficient (0-1)
    """
    if not fertilities or len(fertilities) < 2:
        return 0.0
    
    # Sort fertilities
    sorted_fert = sorted(fertilities)
    n = len(sorted_fert)
    
    # Compute Gini coefficient
    cumsum = 0.0
    for i, fert in enumerate(sorted_fert):
        cumsum += (2 * (i + 1) - n - 1) * fert
    
    gini = cumsum / (n * sum(sorted_fert))
    
    return abs(gini)


def compute_compression_ratio(texts: List[str], encoder: BPEEncoder) -> float:
    """
    Compute compression ratio (tokens / characters).
    
    Lower is better (fewer tokens needed).
    
    Args:
        texts: List of text strings
        encoder: BPE encoder
        
    Returns:
        Compression ratio
    """
    total_tokens = 0
    total_chars = 0
    
    for text in texts:
        tokens = encoder.encode(text)
        total_tokens += len(tokens)
        total_chars += len(text)
    
    return total_tokens / total_chars if total_chars > 0 else 0.0


def format_table(headers: List[str], rows: List[List[str]]) -> str:
    """Format a simple ASCII table."""
    # Compute column widths
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))
    
    # Format header
    header_line = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    separator = "-+-".join("-" * w for w in col_widths)
    
    # Format rows
    row_lines = [
        " | ".join(str(cell).ljust(w) for cell, w in zip(row, col_widths))
        for row in rows
    ]
    
    return "\n".join([header_line, separator] + row_lines)


def benchmark_fairness(
    corpora: Dict[str, List[str]],
    vocab_size: int = 2000,
    fairness_weights: List[float] = None,
) -> None:
    """
    Comprehensive benchmark comparing Standard BPE vs Fair BPE.
    
    Args:
        corpora: Dictionary mapping language -> texts
        vocab_size: Target vocabulary size
        fairness_weights: List of λ values to test
    """
    if fairness_weights is None:
        fairness_weights = [0.0, 0.1, 0.3, 0.5, 0.7]
    
    print("=" * 80)
    print("FAIR BPE BENCHMARK")
    print("=" * 80)
    print(f"\nCorpus statistics:")
    for lang, texts in corpora.items():
        total_words = sum(len(text.split()) for text in texts)
        print(f"  {lang}: {len(texts)} sentences, {total_words} words")
    print(f"\nVocabulary size: {vocab_size}")
    print(f"Fairness weights (λ): {fairness_weights}")
    print()
    
    results = []
    
    # Test each fairness weight
    for idx, lambda_val in enumerate(fairness_weights):
        print(f"\n{'=' * 80}")
        print(f"Experiment {idx + 1}/{len(fairness_weights)}: λ = {lambda_val:.1f}")
        if lambda_val == 0.0:
            print("(Standard BPE - frequency-only)")
        elif lambda_val < 0.3:
            print("(Low fairness - mostly frequency-based)")
        elif lambda_val < 0.7:
            print("(Balanced fairness)")
        else:
            print("(High fairness - heavily penalize inequality)")
        print("=" * 80)
        
        # Train model
        start_time = time.perf_counter()
        
        if lambda_val == 0.0:
            # Standard BPE (train on combined corpus)
            all_texts = []
            for texts in corpora.values():
                all_texts.extend(texts)
            
            vocab, merges, normalizer = train_bpe(
                texts=all_texts,
                vocab_size=vocab_size,
                byte_fallback=False,
                verbose=False,
            )
            
            # Compute stats manually
            language_stats = {}
            for lang, texts in corpora.items():
                encoder = BPEEncoder(vocab, merges, normalizer)
                fertility = compute_fertility(texts, encoder)
                language_stats[lang] = {'fertility': fertility}
        else:
            # Fair BPE
            vocab, merges, normalizer, language_stats = train_fair_bpe(
                corpora=corpora,
                vocab_size=vocab_size,
                fairness_weight=lambda_val,
                byte_fallback=False,
                verbose=False,
            )
        
        training_time = time.perf_counter() - start_time
        
        # Create encoder
        encoder = BPEEncoder(vocab, merges, normalizer)
        
        # Compute metrics
        print(f"\nTraining time: {training_time:.2f}s")
        print(f"\nFertility by language:")
        
        fertilities = []
        for lang in sorted(corpora.keys()):
            if lang not in language_stats:
                # Compute for standard BPE
                fertility = compute_fertility(corpora[lang], encoder)
                language_stats[lang] = {'fertility': fertility}
            
            fertility = language_stats[lang]['fertility']
            fertilities.append(fertility)
            print(f"  {lang}: {fertility:.4f} tokens/word")
        
        # Compute gap and Gini
        max_fert = max(fertilities)
        min_fert = min(fertilities)
        gap = max_fert - min_fert
        gini = compute_gini_coefficient(fertilities)
        
        print(f"\nFertility gap (max - min): {gap:.4f}")
        print(f"Gini coefficient: {gini:.4f}")
        
        # Compute overall compression
        all_texts = []
        for texts in corpora.values():
            all_texts.extend(texts)
        compression = compute_compression_ratio(all_texts, encoder)
        
        print(f"Compression ratio: {compression:.4f} tokens/char")
        
        # Store results
        results.append({
            'lambda': lambda_val,
            'training_time': training_time,
            'gap': gap,
            'gini': gini,
            'compression': compression,
            'fertilities': {lang: language_stats[lang]['fertility'] for lang in sorted(corpora.keys())},
        })
    
    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    
    headers = ["λ", "Gap", "Gini", "Compression", "Time (s)"] + sorted(corpora.keys())
    rows = []
    
    for result in results:
        row = [
            f"{result['lambda']:.1f}",
            f"{result['gap']:.4f}",
            f"{result['gini']:.4f}",
            f"{result['compression']:.4f}",
            f"{result['training_time']:.2f}",
        ]
        for lang in sorted(corpora.keys()):
            row.append(f"{result['fertilities'][lang]:.4f}")
        rows.append(row)
    
    print("\n" + format_table(headers, rows))
    
    # Print analysis
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    
    baseline = results[0]  # λ=0 (Standard BPE)
    best_fair = min(results[1:], key=lambda x: x['gini'])  # Best fairness
    
    gap_reduction = (baseline['gap'] - best_fair['gap']) / baseline['gap'] * 100
    gini_reduction = (baseline['gini'] - best_fair['gini']) / baseline['gini'] * 100
    compression_increase = (best_fair['compression'] - baseline['compression']) / baseline['compression'] * 100
    time_increase = (best_fair['training_time'] - baseline['training_time']) / baseline['training_time'] * 100
    
    print(f"\nStandard BPE (λ=0.0):")
    print(f"  Gap: {baseline['gap']:.4f}")
    print(f"  Gini: {baseline['gini']:.4f}")
    
    print(f"\nBest Fair BPE (λ={best_fair['lambda']:.1f}):")
    print(f"  Gap: {best_fair['gap']:.4f} ({gap_reduction:+.1f}%)")
    print(f"  Gini: {best_fair['gini']:.4f} ({gini_reduction:+.1f}%)")
    print(f"  Compression: {best_fair['compression']:.4f} ({compression_increase:+.1f}%)")
    print(f"  Training time: {best_fair['training_time']:.2f}s ({time_increase:+.1f}%)")
    
    print("\nKey findings:")
    if gap_reduction > 20:
        print(f"  ✓ Fair BPE significantly reduces fertility gap ({gap_reduction:.1f}% reduction)")
    else:
        print(f"  ⚠ Fair BPE provides modest gap reduction ({gap_reduction:.1f}%)")
    
    if gini_reduction > 20:
        print(f"  ✓ Fair BPE significantly reduces inequality (Gini {gini_reduction:.1f}% reduction)")
    else:
        print(f"  ⚠ Fair BPE provides modest Gini reduction ({gini_reduction:.1f}%)")
    
    if abs(compression_increase) < 5:
        print(f"  ✓ Compression efficiency maintained ({compression_increase:+.1f}% change)")
    else:
        print(f"  ⚠ Compression efficiency impacted ({compression_increase:+.1f}% change)")
    
    if time_increase < 50:
        print(f"  ✓ Training overhead is acceptable ({time_increase:+.1f}%)")
    else:
        print(f"  ⚠ Training overhead is significant ({time_increase:+.1f}%)")
    
    print("\n" + "=" * 80)


def test_correctness():
    """Test that Fair BPE implementation is correct."""
    print("=" * 80)
    print("CORRECTNESS TESTS")
    print("=" * 80)
    
    # Test 1: λ=0 should behave like standard BPE
    print("\nTest 1: λ=0 should give same results as standard BPE")
    
    simple_corpus = {
        "Lang1": ["hello world", "hello"],
        "Lang2": ["hola mundo", "hola"],
    }
    
    # Fair BPE with λ=0
    vocab_fair, merges_fair, _, _ = train_fair_bpe(
        corpora=simple_corpus,
        vocab_size=100,
        fairness_weight=0.0,
        byte_fallback=False,
        verbose=False,
    )
    
    # Standard BPE
    all_texts = []
    for texts in simple_corpus.values():
        all_texts.extend(texts)
    
    vocab_std, merges_std, _ = train_bpe(
        texts=all_texts,
        vocab_size=100,
        byte_fallback=False,
        verbose=False,
    )
    
    print(f"  Fair BPE vocab size: {len(vocab_fair)}")
    print(f"  Standard BPE vocab size: {len(vocab_std)}")
    print(f"  Fair BPE merges: {len(merges_fair)}")
    print(f"  Standard BPE merges: {len(merges_std)}")
    
    if len(vocab_fair) == len(vocab_std):
        print("  ✓ Vocabulary sizes match")
    else:
        print("  ✗ Vocabulary sizes differ")
    
    # Test 2: Higher λ should reduce gap
    print("\nTest 2: Higher λ should reduce fertility gap")
    
    _, _, _, stats_low = train_fair_bpe(
        corpora=simple_corpus,
        vocab_size=100,
        fairness_weight=0.1,
        byte_fallback=False,
        verbose=False,
    )
    
    _, _, _, stats_high = train_fair_bpe(
        corpora=simple_corpus,
        vocab_size=100,
        fairness_weight=0.7,
        byte_fallback=False,
        verbose=False,
    )
    
    gap_low = max(s['fertility'] for s in stats_low.values()) - min(s['fertility'] for s in stats_low.values())
    gap_high = max(s['fertility'] for s in stats_high.values()) - min(s['fertility'] for s in stats_high.values())
    
    print(f"  Gap with λ=0.1: {gap_low:.4f}")
    print(f"  Gap with λ=0.7: {gap_high:.4f}")
    
    if gap_high <= gap_low:
        print("  ✓ Higher λ reduces gap")
    else:
        print("  ✗ Higher λ increased gap (unexpected!)")
    
    # Test 3: Fertility should decrease over training
    print("\nTest 3: Training should reduce overall fertility")
    
    vocab, _, _, stats = train_fair_bpe(
        corpora=simple_corpus,
        vocab_size=100,
        fairness_weight=0.3,
        byte_fallback=False,
        verbose=False,
    )
    
    avg_fertility = sum(s['fertility'] for s in stats.values()) / len(stats)
    print(f"  Average fertility: {avg_fertility:.4f}")
    
    if avg_fertility < 5.0:  # Should be much less than character-level
        print("  ✓ Fertility is reasonable")
    else:
        print("  ✗ Fertility is too high")
    
    print("\n" + "=" * 80)


def main():
    """Run all benchmarks."""
    print("\n" + "=" * 80)
    print("FAIR BPE COMPREHENSIVE BENCHMARK SUITE")
    print("=" * 80)
    
    # Run correctness tests
    test_correctness()
    
    print("\n")
    
    # Generate synthetic corpus
    print("Generating synthetic multilingual corpus...")
    corpora = generate_synthetic_corpus()
    print("Done.\n")
    
    # Run fairness benchmark
    benchmark_fairness(
        corpora=corpora,
        vocab_size=1000,
        fairness_weights=[0.0, 0.1, 0.3, 0.5, 0.7],
    )
    
    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()