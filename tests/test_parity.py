"""Integration parity test comparing BytePiece with SentencePiece."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from bytepiece.algorithms.bpe import BPEEncoder, train_bpe
from bytepiece.core.normalizer import (
    NormalizationMode,
    Normalizer,
    PreTokenizationMode,
    SpacerMode,
)


def _train_sentencepiece(texts: list[str], vocab_size: int):
    """Train a SentencePiece BPE model and return the loaded processor."""

    spm = pytest.importorskip("sentencepiece")

    with TemporaryDirectory() as tmpdir:
        corpus_path = Path(tmpdir) / "corpus.txt"
        corpus_path.write_text("\n".join(texts), encoding="utf-8")

        model_prefix = Path(tmpdir) / "sp_model"

        spm.SentencePieceTrainer.train(
            input=str(corpus_path),
            model_prefix=str(model_prefix),
            vocab_size=vocab_size,
            model_type="bpe",
            character_coverage=1.0,
            input_sentence_size=0,
            shuffle_input_sentence=True,
            byte_fallback=False,
        )

        processor = spm.SentencePieceProcessor()
        processor.load(str(model_prefix) + ".model")
        return processor


def _train_bytepiece(texts: list[str], vocab_size: int) -> BPEEncoder:
    """Train BytePiece BPE encoder with SentencePiece-aligned defaults."""

    normalizer = Normalizer(
        normalization=NormalizationMode.NFKC,
        spacer_mode=SpacerMode.PREFIX,
        lowercase=False,
        pre_tokenization=PreTokenizationMode.NONE,
    )

    vocab, merges, norm = train_bpe(
        texts=texts,
        vocab_size=vocab_size,
        normalizer=normalizer,
        byte_fallback=False,
        use_special_tokens=False,
        verbose=False,
    )
    return BPEEncoder(vocab, merges, norm)


def _percentage_diff(count_a: int, count_b: int) -> float:
    """Compute absolute percentage difference between counts."""

    if count_a == 0:
        return 0.0 if count_b == 0 else 100.0
    return abs(count_a - count_b) * 100.0 / count_a


@pytest.mark.integration
def test_bytepiece_sentencepiece_parity():
    """BytePiece should stay within 20% token count of SentencePiece."""

    corpus = [
        "hello world",
        "hello there",
        "test text",
        "another test",
        "simple example",
    ] * 200  # 1000 samples

    vocab_size = 94

    sp_processor = _train_sentencepiece(corpus, vocab_size)
    bytepiece_encoder = _train_bytepiece(corpus, vocab_size)

    eval_sentences = [
        "hello world",
        "test example sentence",
        "another simple test",
        "hello hello test test",
    ]

    diffs = []

    for sentence in eval_sentences:
        sp_tokens = sp_processor.encode(sentence, out_type=str, add_bos=False, add_eos=False)
        bp_tokens = bytepiece_encoder.encode(sentence)
        diffs.append(_percentage_diff(len(sp_tokens), len(bp_tokens)))

    avg_diff = sum(diffs) / len(diffs)

    assert max(diffs) < 20.0, f"Parity gap too high; diffs={diffs}"
    assert avg_diff < 10.0, f"Average gap too high; diffs={diffs}"
