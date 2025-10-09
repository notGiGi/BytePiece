#!/usr/bin/env python3
"""Train tokenizer models for benchmarking.

Trains BytePiece models with different pre-tokenization strategies
and saves them for benchmarking.

Usage:
    python benchmarks/scripts/2_train_models.py
"""

from pathlib import Path
import time
from typing import List

import bytepiece
from bytepiece import PreTokenizationMode, Normalizer

# Configuration
BENCHMARK_ROOT = Path(__file__).parent.parent
DATASET_DIR = BENCHMARK_ROOT / "datasets"
MODELS_DIR = BENCHMARK_ROOT / "models"
VOCAB_SIZE = 10000
SEED = 42


def load_corpus(directory: Path) -> List[str]:
    """Load all text files from a directory."""
    texts = []
    
    for file_path in sorted(directory.glob("*")):
        if file_path.is_file():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    texts.append(f.read())
            except Exception as e:
                print(f"    Warning: Could not read {file_path}: {e}")
    
    return texts


def train_model(
    name: str,
    corpus: List[str],
    pre_tokenization: PreTokenizationMode,
    vocab_size: int = VOCAB_SIZE,
) -> None:
    """Train a BPE model and save it."""
    print(f"  Training {name}...")
    print(f"    Corpus: {len(corpus)} files, {sum(len(t) for t in corpus):,} chars")
    print(f"    Pre-tokenization: {pre_tokenization.value}")
    print(f"    Vocab size: {vocab_size}")
    
    start = time.time()
    
    # Create normalizer with pre-tokenization mode
    normalizer = bytepiece.Normalizer(
        pre_tokenization=pre_tokenization,
    )
    
    # Train
    vocab, merges, normalizer = bytepiece.train_bpe(
        texts=corpus,
        vocab_size=vocab_size,
        normalizer=normalizer,
        byte_fallback=True,
        verbose=False,
    )
    
    # Create encoder
    encoder = bytepiece.BPEEncoder(vocab, merges, normalizer)
    
    # Save model
    model_path = MODELS_DIR / name / "model.json"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    bytepiece.save_model(encoder, str(model_path))
    
    elapsed = time.time() - start
    
    print(f"    ✓ Trained in {elapsed:.1f}s")
    print(f"    ✓ Saved to {model_path}")
    print(f"    Vocab size: {len(vocab)}, Merges: {len(merges)}")
    print()


def main():
    """Main entrypoint."""
    print("\n" + "=" * 70)
    print("🏋️ BytePiece Model Training")
    print("=" * 70 + "\n")
    
    # Check datasets exist
    if not (DATASET_DIR / "python").exists():
        print("❌ Error: Datasets not found!")
        print("   Run: python benchmarks/scripts/1_prepare_datasets.py")
        return
    
    # Create models directory
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load Python corpus
    print("📂 Loading Python corpus...")
    python_corpus = load_corpus(DATASET_DIR / "python")
    print(f"   Loaded {len(python_corpus)} Python files\n")
    
    # Train models with different pre-tokenization strategies
    print("🔨 Training models...\n")
    
    # 1. Character-level (no pre-tokenization)
    train_model(
        name="bytepiece_none",
        corpus=python_corpus,
        pre_tokenization=PreTokenizationMode.NONE,
    )
    
    # 2. Whitespace pre-tokenization
    train_model(
        name="bytepiece_whitespace",
        corpus=python_corpus,
        pre_tokenization=PreTokenizationMode.WHITESPACE,
    )
    
    # 3. GPT2-style pre-tokenization
    train_model(
        name="bytepiece_gpt2",
        corpus=python_corpus,
        pre_tokenization=PreTokenizationMode.GPT2,
    )
    
    # 4. Python code-aware pre-tokenization
    train_model(
        name="bytepiece_code",
        corpus=python_corpus,
        pre_tokenization=PreTokenizationMode.CODE,
    )
    
    # Summary
    print("=" * 70)
    print("✅ Model training complete!")
    print("=" * 70)
    
    models = list(MODELS_DIR.glob("*/model.json"))
    print(f"\nTrained {len(models)} models:")
    for model_path in sorted(models):
        info = bytepiece.get_model_info(str(model_path))
        print(f"  • {model_path.parent.name}")
        print(f"    Pre-tokenization: {info['normalizer']['pre_tokenization']}")
        print(f"    Vocab size: {info['vocab_size']}")
    
    print("\nNext steps:")
    print("  python benchmarks/scripts/3_run_benchmarks.py")
    print()


if __name__ == "__main__":
    main()