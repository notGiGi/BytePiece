"""Unit tests for model I/O."""

import tempfile
from pathlib import Path

import pytest

from bytepiece import train_bpe, save_model, load_model, get_model_info, BPEEncoder


def test_save_and_load_model():
    """Test saving and loading a model."""
    texts = ["hello world", "test"]
    
    vocab, merges, normalizer = train_bpe(
        texts=texts,
        vocab_size=300,
        byte_fallback=True,
    )
    
    encoder = BPEEncoder(vocab, merges, normalizer)
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = f.name
    
    try:
        save_model(encoder, temp_path)
        
        # Load back
        loaded_encoder = load_model(temp_path)
        
        # Test that loaded model works the same
        test_text = "hello test"
        original_tokens = encoder.encode(test_text)
        loaded_tokens = loaded_encoder.encode(test_text)
        
        assert original_tokens == loaded_tokens
    
    finally:
        Path(temp_path).unlink(missing_ok=True)
