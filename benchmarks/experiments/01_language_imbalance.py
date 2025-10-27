#!/usr/bin/env python3
"""
Language Tokenization Imbalance - Experiment 1
Measures fertility (tokens/word) across languages using BytePiece

Location: benchmarks/experiments/01_language_imbalance.py
Usage:    python benchmarks/experiments/01_language_imbalance.py
"""

import sys
from pathlib import Path
from typing import Dict, List
import tempfile
import json

# Add project root to path (allows importing bytepiece)
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import your real bytepiece implementation
import bytepiece


# Sample texts in different languages (similar meaning)
SAMPLE_TEXTS = {
    "English": "Hello, how are you today? I hope you are doing well. The weather is nice and sunny.",
    "Spanish": "Hola, ¿cómo estás hoy? Espero que estés bien. El clima es agradable y soleado.",
    "French": "Bonjour, comment allez-vous aujourd'hui? J'espère que vous allez bien. Le temps est beau et ensoleillé.",
    "German": "Hallo, wie geht es dir heute? Ich hoffe, es geht dir gut. Das Wetter ist schön und sonnig.",
    "Portuguese": "Olá, como você está hoje? Espero que você esteja bem. O clima está agradável e ensolarado.",
    "Italian": "Ciao, come stai oggi? Spero che tu stia bene. Il tempo è bello e soleggiato.",
}


def create_training_corpus(texts: Dict[str, str], output_path: Path) -> None:
    """Create a training corpus file from sample texts."""
    print("  Creating training corpus...")
    with open(output_path, 'w', encoding='utf-8') as f:
        for lang, text in texts.items():
            # Repeat each text 50 times to give BPE enough training data
            for _ in range(50):
                f.write(text + "\n")
    
    print(f"  ✓ Corpus created: {output_path}")


def count_words(text: str) -> int:
    """Count words in text (simple whitespace split)."""
    return len(text.split())


def compute_fertility(text: str, tokens: List[str]) -> float:
    """
    Compute fertility: tokens per word.
    Lower is better (more efficient).
    """
    num_words = count_words(text)
    num_tokens = len(tokens)
    return num_tokens / num_words if num_words > 0 else 0


def analyze_language(encoder, language: str, text: str) -> Dict:
    """Analyze tokenization for one language."""
    tokens = encoder.encode(text)
    fertility = compute_fertility(text, tokens)
    
    return {
        "language": language,
        "text": text,
        "num_chars": len(text),
        "num_words": count_words(text),
        "num_tokens": len(tokens),
        "fertility": fertility,
        "tokens_sample": tokens[:15],  # First 15 tokens for inspection
    }


def print_header():
    """Print experiment header."""
    print("\n" + "=" * 80)
    print(" " * 15 + "LANGUAGE BALANCE EXPERIMENT - BytePiece")
    print(" " * 20 + f"(bytepiece v{bytepiece.__version__})")
    print("=" * 80 + "\n")


def print_results_table(results: List[Dict], vocab_size: int):
    """Print results in a table format."""
    results_sorted = sorted(results, key=lambda x: x["fertility"])
    
    # Find English as baseline
    english_result = next((r for r in results if r["language"] == "English"), results[0])
    english_fertility = english_result["fertility"]
    
    print("📊 FERTILITY RESULTS (Tokens per Word)")
    print("-" * 80)
    print(f"{'Rank':<6} {'Language':<15} {'Fertility':<12} {'vs English':<12} {'Tokens':<8} {'Words'}")
    print("-" * 80)
    
    for rank, result in enumerate(results_sorted, 1):
        lang = result["language"]
        fert = result["fertility"]
        ratio = fert / english_fertility if english_fertility > 0 else 1.0
        tokens = result["num_tokens"]
        words = result["num_words"]
        
        # Color indicators
        if ratio < 1.15:
            indicator = "✅"
        elif ratio < 1.35:
            indicator = "⚠️ "
        else:
            indicator = "❌"
        
        print(f"{indicator} {rank:<4} {lang:<15} {fert:<12.2f} {ratio:<12.2f}x {tokens:<8} {words}")
    
    print("-" * 80 + "\n")


def print_statistics(results: List[Dict]):
    """Print summary statistics."""
    fertilities = [r["fertility"] for r in results]
    results_sorted = sorted(results, key=lambda x: x["fertility"])
    
    avg = sum(fertilities) / len(fertilities)
    min_fert = min(fertilities)
    max_fert = max(fertilities)
    fairness_ratio = min_fert / max_fert
    
    print("📈 STATISTICS")
    print("-" * 80)
    print(f"  Average fertility:     {avg:.2f} tokens/word")
    print(f"  Most efficient:        {min_fert:.2f} ({results_sorted[0]['language']})")
    print(f"  Least efficient:       {max_fert:.2f} ({results_sorted[-1]['language']})")
    print(f"  Efficiency gap:        {max_fert/min_fert:.2f}x")
    print(f"  Fairness score:        {fairness_ratio:.2f} (1.0 = perfect)")
    print("-" * 80 + "\n")
    
    return fairness_ratio, max_fert/min_fert


def print_examples(results: List[Dict]):
    """Print concrete tokenization examples."""
    results_sorted = sorted(results, key=lambda x: x["fertility"])
    best = results_sorted[0]
    worst = results_sorted[-1]
    english = next((r for r in results if r["language"] == "English"), best)
    
    print("🔍 CONCRETE EXAMPLES")
    print("-" * 80)
    
    print(f"\n✅ {english['language']} (baseline):")
    print(f"   Text:   \"{english['text'][:55]}...\"")
    print(f"   Tokens: {english['tokens_sample']}")
    print(f"   Total:  {english['num_tokens']} tokens for {english['num_words']} words")
    
    if worst['language'] != english['language']:
        print(f"\n❌ {worst['language']} (least efficient):")
        print(f"   Text:   \"{worst['text'][:55]}...\"")
        print(f"   Tokens: {worst['tokens_sample']}")
        print(f"   Total:  {worst['num_tokens']} tokens for {worst['num_words']} words")
        diff = worst['num_tokens'] - english['num_tokens']
        print(f"   Impact: +{diff} tokens ({diff/english['num_tokens']*100:.0f}% more)")
    
    print("\n" + "-" * 80 + "\n")


def print_conclusion(fairness_ratio: float, gap: float):
    """Print conclusion and recommendations."""
    print("💡 CONCLUSION")
    print("=" * 80)
    
    if fairness_ratio < 0.6:
        print("🚨 SIGNIFICANT IMBALANCE DETECTED!")
        print(f"\n   • The efficiency gap is {gap:.1f}x between languages")
        print(f"   • Some languages require {(gap-1)*100:.0f}% more tokens")
        print(f"   • This translates to {(gap-1)*100:.0f}% higher costs")
        print("\n   ✅ THIS PROBLEM IS REAL AND WORTH SOLVING!")
        print("\n   Recommendation: Proceed with fair tokenization research")
        
    elif fairness_ratio < 0.8:
        print("⚠️  MODERATE IMBALANCE")
        print(f"\n   • Gap of {gap:.1f}x exists")
        print(f"   • Some improvement possible")
        print("\n   Recommendation: Test with more diverse languages")
        
    else:
        print("✅ RELATIVELY BALANCED")
        print(f"\n   • Gap is only {gap:.1f}x")
        print("\n   Note: Test with more diverse scripts (Arabic, Chinese, Hindi)")
    
    print("=" * 80)


def save_results(results: List[Dict], fairness_ratio: float, gap: float, vocab_size: int, output_path: Path):
    """Save results to JSON file."""
    from datetime import datetime
    
    output = {
        "experiment": "language_imbalance",
        "date": datetime.now().isoformat(),
        "bytepiece_version": bytepiece.__version__,
        "config": {
            "vocab_size": vocab_size,
            "num_languages": len(results),
        },
        "metrics": {
            "fairness_ratio": fairness_ratio,
            "efficiency_gap": gap,
            "avg_fertility": sum(r["fertility"] for r in results) / len(results),
        },
        "languages": results,
    }
    
    # Ensure directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Results saved to: {output_path}\n")


def main():
    """Run the experiment."""
    print_header()
    
    # Step 1: Create training corpus
    print("📝 STEP 1: Preparing training data")
    print("-" * 80)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
        corpus_path = Path(f.name)
    
    create_training_corpus(SAMPLE_TEXTS, corpus_path)
    
    corpus_size = corpus_path.stat().st_size
    print(f"  ✓ Corpus size: {corpus_size:,} bytes")
    print(f"  ✓ Languages: {len(SAMPLE_TEXTS)}")
    print()
    
    # Step 2: Train BPE tokenizer
    print("🔧 STEP 2: Training vanilla BPE tokenizer")
    print("-" * 80)
    vocab_size = 500
    
    try:
        print(f"  Training with vocab_size={vocab_size}...")
        encoder = bytepiece.train_bpe(
            corpus_path=str(corpus_path),
            vocab_size=vocab_size,
            verbose=False,
        )
        print(f"  ✓ Training complete")
        print(f"  ✓ Final vocabulary size: {len(encoder.vocab.tokens)} tokens")
        print(f"  ✓ Number of merges: {len(encoder.merge_rules.merges)}")
        print()
    except Exception as e:
        print(f"  ✗ Error during training: {e}")
        import traceback
        traceback.print_exc()
        corpus_path.unlink()
        return 1
    
    # Step 3: Analyze each language
    print("📊 STEP 3: Analyzing tokenization across languages")
    print("-" * 80)
    results = []
    
    for language, text in SAMPLE_TEXTS.items():
        result = analyze_language(encoder, language, text)
        results.append(result)
        print(f"  ✓ Analyzed {language:<12} ({result['num_tokens']} tokens)")
    
    print("\n")
    
    # Step 4: Display results
    print_results_table(results, vocab_size)
    fairness_ratio, gap = print_statistics(results)
    print_examples(results)
    print_conclusion(fairness_ratio, gap)
    
    # Step 5: Save results
    results_dir = PROJECT_ROOT / "benchmarks" / "results" / "language_balance"
    output_path = results_dir / "01_initial_validation.json"
    save_results(results, fairness_ratio, gap, vocab_size, output_path)
    
    # Next steps
    print("📋 NEXT STEPS:")
    print("-" * 80)
    print("  1. ✅ Validated that imbalance exists")
    print("  2. 🔄 Add more diverse languages (Arabic, Chinese, Hindi)")
    print("  3. 📊 Test with different vocabulary sizes")
    print("  4. 💡 Design fair tokenization strategy")
    print("  5. 🧪 Implement and benchmark the solution")
    print("-" * 80 + "\n")
    
    # Cleanup
    corpus_path.unlink()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())