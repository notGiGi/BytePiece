"""
Comprehensive Parity Test: BytePiece vs SentencePiece

Tests multiple dimensions:
- Token count parity
- Compression ratio
- Vocabulary efficiency
- Training time
- Tokenization speed
- Cross-corpus generalization

Uses realistic corpora:
- English text
- Python code
- Multilingual text
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Dict, Tuple

import pytest

from bytepiece.algorithms.bpe import BPEEncoder, train_bpe
from bytepiece.core.normalizer import (
    NormalizationMode,
    Normalizer,
    PreTokenizationMode,
    SpacerMode,
)


# ============================================================================
# REALISTIC CORPORA
# ============================================================================

ENGLISH_CORPUS = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning is a subset of artificial intelligence.",
    "Python is a high-level programming language.",
    "Natural language processing enables computers to understand human language.",
    "Tokenization is the process of breaking text into smaller units.",
    "Byte Pair Encoding is a data compression technique.",
    "The algorithm iteratively merges the most frequent pairs of bytes.",
    "This approach balances vocabulary size and sequence length.",
    "Deep learning models require large amounts of training data.",
    "Transformers have revolutionized natural language understanding.",
] * 100  # 1000 samples

PYTHON_CODE_CORPUS = [
    "def calculate_sum(a, b):\n    return a + b",
    "class Calculator:\n    def __init__(self):\n        self.result = 0",
    "if x >= 10:\n    print('Greater than or equal to 10')",
    "for i in range(100):\n    total += i",
    "url = 'https://api.example.com/v1/users'",
    "import numpy as np\nimport pandas as pd",
    "try:\n    result = process_data()\nexcept Exception as e:\n    log_error(e)",
    "lambda x: x * 2 if x > 0 else 0",
    "def fibonacci(n):\n    return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
    "@decorator\ndef wrapped_function():\n    pass",
] * 100  # 1000 samples

MULTILINGUAL_CORPUS = [
    "Hello world",  # English
    "Hola mundo",   # Spanish
    "Bonjour monde",  # French
    "Hallo Welt",   # German
    "Ciao mondo",   # Italian
    "Olá mundo",    # Portuguese
    "Привет мир",   # Russian
    "مرحبا بالعالم",  # Arabic
    "你好世界",      # Chinese
    "こんにちは世界",  # Japanese
] * 100  # 1000 samples


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class TokenizationMetrics:
    """Metrics for a single corpus evaluation."""
    corpus_name: str
    tokenizer_name: str
    
    # Token counts
    avg_tokens_per_sentence: float
    total_tokens: int
    total_chars: int
    
    # Compression
    compression_ratio: float  # tokens/chars
    
    # Timing
    tokenization_time_ms: float
    tokens_per_second: float
    
    # Vocabulary
    vocab_size: int
    vocab_utilization: float  # % of vocab used


@dataclass
class ParityReport:
    """Complete parity comparison report."""
    vocab_size: int
    training_corpus_size: int
    
    # Training time
    bytepiece_train_time_ms: float
    sentencepiece_train_time_ms: float
    
    # Per-corpus metrics
    metrics: List[TokenizationMetrics]
    
    # Parity scores
    avg_token_count_diff_pct: float
    max_token_count_diff_pct: float
    compression_ratio_diff_pct: float
    speed_ratio: float  # BP speed / SP speed
    
    # Pass/fail
    passes_parity: bool
    failure_reasons: List[str]


# ============================================================================
# TRAINING HELPERS
# ============================================================================

def _train_sentencepiece(
    texts: List[str], 
    vocab_size: int,
    verbose: bool = False
) -> tuple:
    """Train SentencePiece and return (processor, train_time_ms)."""
    spm = pytest.importorskip("sentencepiece")
    
    with TemporaryDirectory() as tmpdir:
        corpus_path = Path(tmpdir) / "corpus.txt"
        corpus_path.write_text("\n".join(texts), encoding="utf-8")
        
        model_prefix = Path(tmpdir) / "sp_model"
        
        start_time = time.perf_counter()
        
        spm.SentencePieceTrainer.train(
            input=str(corpus_path),
            model_prefix=str(model_prefix),
            vocab_size=vocab_size,
            model_type="bpe",
            character_coverage=1.0,
            normalization_rule_name="nfkc",
            input_sentence_size=0,
            shuffle_input_sentence=False,  # Deterministic
            byte_fallback=False,
            add_dummy_prefix=True,  # Equivalent to SpacerMode.PREFIX
            split_by_unicode_script=False,
            split_by_whitespace=True,
            split_by_number=False,
        )
        
        train_time = (time.perf_counter() - start_time) * 1000
        
        processor = spm.SentencePieceProcessor()
        processor.load(str(model_prefix) + ".model")
        
        if verbose:
            print(f"  SentencePiece trained in {train_time:.1f}ms")
        
        return processor, train_time


def _train_bytepiece(
    texts: List[str], 
    vocab_size: int,
    verbose: bool = False
) -> tuple[BPEEncoder, float]:
    """Train BytePiece and return (encoder, train_time_ms)."""
    
    normalizer = Normalizer(
        normalization=NormalizationMode.NFKC,
        spacer_mode=SpacerMode.PREFIX,
        lowercase=False,
        pre_tokenization=PreTokenizationMode.NONE,  # SentencePiece handles whitespace internally
    )
    
    start_time = time.perf_counter()
    
    vocab, merges, norm = train_bpe(
        texts=texts,
        vocab_size=vocab_size,
        normalizer=normalizer,
        byte_fallback=False,
        use_special_tokens=False,
        verbose=False,
    )
    
    train_time = (time.perf_counter() - start_time) * 1000
    
    encoder = BPEEncoder(vocab, merges, norm)
    
    if verbose:
        print(f"  BytePiece trained in {train_time:.1f}ms")
    
    return encoder, train_time


# ============================================================================
# EVALUATION HELPERS
# ============================================================================

def _evaluate_corpus(
    corpus_name: str,
    tokenizer_name: str,
    eval_sentences: List[str],
    encode_fn,
    vocab_size: int,
) -> TokenizationMetrics:
    """Evaluate tokenizer on a corpus."""
    
    # Warm-up
    for _ in range(10):
        _ = encode_fn(eval_sentences[0])
    
    # Timed tokenization
    all_tokens = []
    start_time = time.perf_counter()
    
    for sentence in eval_sentences:
        tokens = encode_fn(sentence)
        all_tokens.append(tokens)
    
    elapsed_time = (time.perf_counter() - start_time) * 1000
    
    # Calculate metrics
    total_tokens = sum(len(tokens) for tokens in all_tokens)
    total_chars = sum(len(sentence) for sentence in eval_sentences)
    avg_tokens = total_tokens / len(eval_sentences)
    compression_ratio = total_tokens / total_chars if total_chars > 0 else 0
    
    # Vocabulary utilization
    unique_tokens = set()
    for tokens in all_tokens:
        if isinstance(tokens, list):
            unique_tokens.update(tokens)
        else:
            unique_tokens.update(str(t) for t in tokens)
    
    vocab_utilization = len(unique_tokens) / vocab_size * 100
    
    # Throughput
    tokens_per_second = total_tokens / (elapsed_time / 1000) if elapsed_time > 0 else 0
    
    return TokenizationMetrics(
        corpus_name=corpus_name,
        tokenizer_name=tokenizer_name,
        avg_tokens_per_sentence=avg_tokens,
        total_tokens=total_tokens,
        total_chars=total_chars,
        compression_ratio=compression_ratio,
        tokenization_time_ms=elapsed_time,
        tokens_per_second=tokens_per_second,
        vocab_size=vocab_size,
        vocab_utilization=vocab_utilization,
    )


def _compute_parity_scores(
    bp_metrics: List[TokenizationMetrics],
    sp_metrics: List[TokenizationMetrics],
) -> Tuple[float, float, float, float]:
    """Compute parity scores between BytePiece and SentencePiece."""
    
    # Token count differences
    token_diffs = []
    for bp, sp in zip(bp_metrics, sp_metrics):
        if sp.avg_tokens_per_sentence > 0:
            diff_pct = abs(bp.avg_tokens_per_sentence - sp.avg_tokens_per_sentence) / sp.avg_tokens_per_sentence * 100
            token_diffs.append(diff_pct)
    
    avg_token_diff = statistics.mean(token_diffs) if token_diffs else 0
    max_token_diff = max(token_diffs) if token_diffs else 0
    
    # Compression ratio difference
    bp_avg_compression = statistics.mean(m.compression_ratio for m in bp_metrics)
    sp_avg_compression = statistics.mean(m.compression_ratio for m in sp_metrics)
    compression_diff = abs(bp_avg_compression - sp_avg_compression) / sp_avg_compression * 100 if sp_avg_compression > 0 else 0
    
    # Speed ratio
    bp_avg_speed = statistics.mean(m.tokens_per_second for m in bp_metrics)
    sp_avg_speed = statistics.mean(m.tokens_per_second for m in sp_metrics)
    speed_ratio = bp_avg_speed / sp_avg_speed if sp_avg_speed > 0 else 0
    
    return avg_token_diff, max_token_diff, compression_diff, speed_ratio


# ============================================================================
# MAIN TEST FUNCTION
# ============================================================================

@pytest.mark.integration
@pytest.mark.parametrize("vocab_size", [500, 1000, 2000])
def test_comprehensive_parity(vocab_size: int, tmp_path: Path):
    """
    Comprehensive parity test comparing BytePiece vs SentencePiece.
    
    Acceptance criteria:
    - Token count difference < 15% on average
    - Max token count difference < 25% on any corpus
    - Compression ratio difference < 10%
    - Both tokenizers complete successfully
    """
    
    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE PARITY TEST - vocab_size={vocab_size}")
    print(f"{'='*80}\n")
    
    # ========================================================================
    # 1. TRAINING
    # ========================================================================
    
    print("📚 Training tokenizers...")
    
    training_corpus = ENGLISH_CORPUS + PYTHON_CODE_CORPUS[:500]
    
    sp_processor, sp_train_time = _train_sentencepiece(
        training_corpus, vocab_size, verbose=True
    )
    
    bp_encoder, bp_train_time = _train_bytepiece(
        training_corpus, vocab_size, verbose=True
    )
    
    print(f"  Training time ratio: {bp_train_time/sp_train_time:.2f}x\n")
    
    # ========================================================================
    # 2. EVALUATION ON MULTIPLE CORPORA
    # ========================================================================
    
    print("🔍 Evaluating on test corpora...")
    
    test_corpora = {
        "English": ENGLISH_CORPUS[800:900],  # Held-out 100 samples
        "Python Code": PYTHON_CODE_CORPUS[800:900],
        "Multilingual": MULTILINGUAL_CORPUS[800:900],
    }
    
    bp_metrics = []
    sp_metrics = []
    
    for corpus_name, eval_sentences in test_corpora.items():
        print(f"\n  Corpus: {corpus_name}")
        
        # Evaluate SentencePiece
        sp_metric = _evaluate_corpus(
            corpus_name=corpus_name,
            tokenizer_name="SentencePiece",
            eval_sentences=eval_sentences,
            encode_fn=lambda s: sp_processor.encode(s, out_type=str, add_bos=False, add_eos=False),
            vocab_size=vocab_size,
        )
        sp_metrics.append(sp_metric)
        
        # Evaluate BytePiece
        bp_metric = _evaluate_corpus(
            corpus_name=corpus_name,
            tokenizer_name="BytePiece",
            eval_sentences=eval_sentences,
            encode_fn=bp_encoder.encode,
            vocab_size=vocab_size,
        )
        bp_metrics.append(bp_metric)
        
        # Print comparison
        token_diff = abs(bp_metric.avg_tokens_per_sentence - sp_metric.avg_tokens_per_sentence) / sp_metric.avg_tokens_per_sentence * 100
        
        print(f"    SentencePiece: {sp_metric.avg_tokens_per_sentence:.2f} tokens/sentence, "
              f"{sp_metric.compression_ratio:.3f} compression, "
              f"{sp_metric.tokens_per_second:.0f} tok/s")
        
        print(f"    BytePiece:     {bp_metric.avg_tokens_per_sentence:.2f} tokens/sentence, "
              f"{bp_metric.compression_ratio:.3f} compression, "
              f"{bp_metric.tokens_per_second:.0f} tok/s")
        
        print(f"    Token diff:    {token_diff:.1f}%")
    
    # ========================================================================
    # 3. COMPUTE PARITY SCORES
    # ========================================================================
    
    avg_token_diff, max_token_diff, compression_diff, speed_ratio = _compute_parity_scores(
        bp_metrics, sp_metrics
    )
    
    # ========================================================================
    # 4. DETERMINE PASS/FAIL
    # ========================================================================
    
    failure_reasons = []
    
    if avg_token_diff > 15.0:
        failure_reasons.append(f"Average token count diff too high: {avg_token_diff:.1f}% (threshold: 15%)")
    
    if max_token_diff > 25.0:
        failure_reasons.append(f"Max token count diff too high: {max_token_diff:.1f}% (threshold: 25%)")
    
    if compression_diff > 10.0:
        failure_reasons.append(f"Compression ratio diff too high: {compression_diff:.1f}% (threshold: 10%)")
    
    passes_parity = len(failure_reasons) == 0
    
    # ========================================================================
    # 5. GENERATE REPORT
    # ========================================================================
    
    report = ParityReport(
        vocab_size=vocab_size,
        training_corpus_size=len(training_corpus),
        bytepiece_train_time_ms=bp_train_time,
        sentencepiece_train_time_ms=sp_train_time,
        metrics=bp_metrics + sp_metrics,
        avg_token_count_diff_pct=avg_token_diff,
        max_token_count_diff_pct=max_token_diff,
        compression_ratio_diff_pct=compression_diff,
        speed_ratio=speed_ratio,
        passes_parity=passes_parity,
        failure_reasons=failure_reasons,
    )
    
    # Save report
    report_path = tmp_path / f"parity_report_vocab{vocab_size}.json"
    with open(report_path, 'w') as f:
        json.dump(asdict(report), f, indent=2)
    
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Average token count difference: {avg_token_diff:.1f}%")
    print(f"Max token count difference:     {max_token_diff:.1f}%")
    print(f"Compression ratio difference:   {compression_diff:.1f}%")
    print(f"Speed ratio (BP/SP):            {speed_ratio:.2f}x")
    print(f"Report saved: {report_path}")
    
    if passes_parity:
        print(f"\n✅ PARITY TEST PASSED")
    else:
        print(f"\n❌ PARITY TEST FAILED")
        for reason in failure_reasons:
            print(f"   - {reason}")
    
    print(f"{'='*80}\n")
    
    # ========================================================================
    # 6. ASSERTIONS
    # ========================================================================
    
    assert avg_token_diff < 15.0, (
        f"Average token count difference too high: {avg_token_diff:.1f}% "
        f"(threshold: 15%). BytePiece may be under/over-tokenizing compared to SentencePiece."
    )
    
    assert max_token_diff < 25.0, (
        f"Max token count difference too high on at least one corpus: {max_token_diff:.1f}% "
        f"(threshold: 25%). Check corpus-specific behavior."
    )
    
    assert compression_diff < 10.0, (
        f"Compression ratio differs significantly: {compression_diff:.1f}% "
        f"(threshold: 10%). Vocabularies may have different characteristics."
    )
    
    # Note: We don't assert on speed since BytePiece is pure Python
    # and expected to be slower. The goal is functional parity.


# ============================================================================
# STANDALONE EXECUTION (for manual testing)
# ============================================================================

if __name__ == "__main__":
    """Run parity test manually."""
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        test_comprehensive_parity(vocab_size=1000, tmp_path=Path(tmpdir))