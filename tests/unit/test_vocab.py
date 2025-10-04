"""Unit tests for vocabulary."""

import pytest

from bytepiece.core.vocab import Vocabulary, MergeRules


def test_vocabulary_byte_fallback_init():
    """Test vocabulary initializes with byte tokens."""
    vocab = Vocabulary(byte_fallback=True)
    
    # Should have 256 byte tokens
    assert len(vocab) == 256
    
    # Check some byte tokens
    assert "<0x00>" in vocab
    assert "<0xFF>" in vocab


def test_vocabulary_decode_bytes():
    """Test decoding byte tokens."""
    vocab = Vocabulary(byte_fallback=True)
    
    # Create byte tokens for "你" (UTF-8: E4 BD A0)
    tokens = ["<0xE4>", "<0xBD>", "<0xA0>"]
    
    decoded = vocab.decode_bytes(tokens)
    assert decoded == "你"
