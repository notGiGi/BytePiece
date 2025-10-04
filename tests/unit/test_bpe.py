"""Unit tests for BPE algorithm."""

import pytest

from bytepiece import train_bpe, BPEEncoder, Normalizer, NormalizationMode, SpacerMode


def test_bpe_basic_training():
    """Test basic BPE training."""
    texts = ["hello world", "hello", "world"]
    
    vocab, merges, normalizer = train_bpe(
        texts=texts,
        vocab_size=300,
        byte_fallback=True,
    )
    
    assert len(vocab) > 0
    assert len(merges) > 0


def test_bpe_encode_decode_identity():
    """Test that encode -> decode returns original text."""
    texts = ["hello world", "test text", "unicode: 你好"]
    
    vocab, merges, normalizer = train_bpe(
        texts=texts,
        vocab_size=300,
        byte_fallback=True,
    )
    
    encoder = BPEEncoder(vocab, merges, normalizer)
    
    for text in texts:
        tokens = encoder.encode(text)
        decoded = encoder.decode(tokens)
        # After normalization and denormalization
        normalized = normalizer.normalize(text)
        expected = normalizer.denormalize(normalized)
        assert decoded == expected


def test_bpe_byte_fallback():
    """Test byte-fallback for unseen characters."""
    # Train on English only
    texts = ["hello world", "test"]
    
    vocab, merges, normalizer = train_bpe(
        texts=texts,
        vocab_size=300,
        byte_fallback=True,
    )
    
    encoder = BPEEncoder(vocab, merges, normalizer)
    
    # Encode text with Chinese characters (not in training)
    text = "你好"
    tokens = encoder.encode(text)
    
    # Should have byte tokens
    assert any('<0x' in token for token in tokens)
    
    # Should decode back correctly
    decoded = encoder.decode(tokens)
    normalized = normalizer.normalize(text)
    expected = normalizer.denormalize(normalized)
    assert decoded == expected


def test_bpe_with_emoji():
    """Test handling of emoji and complex Unicode."""
    texts = ["hello 🚀 world", "test 🎉"]
    
    vocab, merges, normalizer = train_bpe(
        texts=texts,
        vocab_size=400,
        byte_fallback=True,
    )
    
    encoder = BPEEncoder(vocab, merges, normalizer)
    
    text = "hello 🚀"
    tokens = encoder.encode(text)
    decoded = encoder.decode(tokens)
    
    normalized = normalizer.normalize(text)
    expected = normalizer.denormalize(normalized)
    assert decoded == expected


def test_bpe_batch_encoding():
    """Test batch encoding and decoding."""
    texts = ["hello", "world", "test"]
    
    vocab, merges, normalizer = train_bpe(
        texts=texts,
        vocab_size=300,
        byte_fallback=True,
    )
    
    encoder = BPEEncoder(vocab, merges, normalizer)
    
    batch = ["hello world", "test case"]
    token_lists = encoder.encode_batch(batch)
    
    assert len(token_lists) == len(batch)
    assert all(isinstance(tokens, list) for tokens in token_lists)
    
    decoded_batch = encoder.decode_batch(token_lists)
    assert len(decoded_batch) == len(batch)


def test_bpe_empty_text():
    """Test handling of empty text."""
    texts = ["hello", "world"]
    
    vocab, merges, normalizer = train_bpe(
        texts=texts,
        vocab_size=300,
        byte_fallback=True,
    )
    
    encoder = BPEEncoder(vocab, merges, normalizer)
    
    tokens = encoder.encode("")
    decoded = encoder.decode(tokens)
    assert decoded == "" or decoded.strip() == ""


def test_bpe_deterministic():
    """Test that BPE training is deterministic."""
    texts = ["hello world", "test"]
    
    vocab1, merges1, _ = train_bpe(texts=texts, vocab_size=300, byte_fallback=True)
    vocab2, merges2, _ = train_bpe(texts=texts, vocab_size=300, byte_fallback=True)
    
    # Should produce identical results
    assert len(vocab1) == len(vocab2)
    assert len(merges1) == len(merges2)
    assert merges1.merges == merges2.merges
