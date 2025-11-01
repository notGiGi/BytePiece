"""Tests for model I/O and serialization."""

import pytest
import json
from pathlib import Path

from bytepiece import train_bpe, save_model, load_model, get_model_info
from bytepiece.core.normalizer import Normalizer, NormalizationMode


@pytest.fixture
def trained_model(tmp_path):
    """Create a trained model for testing."""
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(
        "hello world\n"
        "python programming\n"
        "machine learning\n"
    )
    
    encoder = train_bpe(
        corpus_path=str(corpus),
        vocab_size=100,
        seed=42,
        return_encoder=True,
    )
    
    return encoder


class TestModelSaving:
    """Test model saving functionality."""

    def test_save_model(self, trained_model, tmp_path):
        """Can save model to file."""
        model_path = tmp_path / "model.json"
        
        save_model(trained_model, str(model_path))
        
        assert model_path.exists()
        assert model_path.stat().st_size > 0
    
    def test_save_creates_directories(self, trained_model, tmp_path):
        """Save creates parent directories if needed."""
        model_path = tmp_path / "subdir" / "models" / "model.json"
        
        save_model(trained_model, str(model_path))
        
        assert model_path.exists()
    
    def test_saved_model_is_valid_json(self, trained_model, tmp_path):
        """Saved model is valid JSON."""
        model_path = tmp_path / "model.json"
        
        save_model(trained_model, str(model_path))
        
        with open(model_path) as f:
            data = json.load(f)
        
        assert isinstance(data, dict)
        assert "algorithm" in data
        assert "version" in data
        assert "vocab" in data
        assert "merges" in data
    
    def test_save_includes_metadata(self, trained_model, tmp_path):
        """Saved model includes all required metadata."""
        model_path = tmp_path / "model.json"
        
        metadata = {"source": "test", "description": "Test model"}
        save_model(trained_model, str(model_path), metadata=metadata)
        
        with open(model_path) as f:
            data = json.load(f)
        
        assert data["algorithm"] == "bpe"
        assert "version" in data
        assert "created_at" in data
        assert "model_hash" in data
        assert data["metadata"] == metadata
    
    def test_save_includes_normalizer_config(self, trained_model, tmp_path):
        """Saved model includes normalizer configuration."""
        model_path = tmp_path / "model.json"
        
        save_model(trained_model, str(model_path))
        
        with open(model_path) as f:
            data = json.load(f)
        
        assert "normalizer" in data
        assert "normalization_mode" in data["normalizer"]
        assert "spacer_mode" in data["normalizer"]


class TestModelLoading:
    """Test model loading functionality."""

    def test_load_model(self, trained_model, tmp_path):
        """Can load saved model."""
        model_path = tmp_path / "model.json"
        
        save_model(trained_model, str(model_path))
        loaded = load_model(str(model_path))
        
        assert loaded is not None
        assert len(loaded.vocab.tokens) == len(trained_model.vocab.tokens)
    
    def test_load_preserves_vocabulary(self, trained_model, tmp_path):
        """Loading preserves vocabulary."""
        model_path = tmp_path / "model.json"
        
        save_model(trained_model, str(model_path))
        loaded = load_model(str(model_path))
        
        assert loaded.vocab.tokens == trained_model.vocab.tokens
    
    def test_load_preserves_merge_rules(self, trained_model, tmp_path):
        """Loading preserves merge rules."""
        model_path = tmp_path / "model.json"
        
        save_model(trained_model, str(model_path))
        loaded = load_model(str(model_path))
        
        assert loaded.merge_rules.merges == trained_model.merge_rules.merges
    
    def test_load_preserves_normalizer_config(self, trained_model, tmp_path):
        """Loading preserves normalizer configuration."""
        model_path = tmp_path / "model.json"
        
        save_model(trained_model, str(model_path))
        loaded = load_model(str(model_path))
        
        assert loaded.normalizer.normalization_mode == trained_model.normalizer.normalization_mode
        assert loaded.normalizer.spacer_mode == trained_model.normalizer.spacer_mode
    
    def test_load_nonexistent_file_raises(self, tmp_path):
        """Loading nonexistent file raises error."""
        model_path = tmp_path / "nonexistent.json"
        
        with pytest.raises(FileNotFoundError):
            load_model(str(model_path))
    
    def test_load_invalid_json_raises(self, tmp_path):
        """Loading invalid JSON raises error."""
        model_path = tmp_path / "invalid.json"
        model_path.write_text("not valid json {]")
        
        with pytest.raises(json.JSONDecodeError):
            load_model(str(model_path))


class TestSaveLoadRoundtrip:
    """Test save/load roundtrip."""

    def test_roundtrip_preserves_functionality(self, trained_model, tmp_path):
        """Save/load roundtrip preserves encoding functionality."""
        model_path = tmp_path / "model.json"
        
        # Save and load
        save_model(trained_model, str(model_path))
        loaded = load_model(str(model_path))
        
        # Test on same text
        text = "hello world python"
        
        original_tokens = trained_model.encode(text)
        loaded_tokens = loaded.encode(text)
        
        assert original_tokens == loaded_tokens
    
    def test_roundtrip_preserves_decoding(self, trained_model, tmp_path):
        """Save/load roundtrip preserves decoding functionality."""
        model_path = tmp_path / "model.json"
        
        save_model(trained_model, str(model_path))
        loaded = load_model(str(model_path))
        
        tokens = ["hello", "▁world"]
        
        original_decoded = trained_model.decode(tokens)
        loaded_decoded = loaded.decode(tokens)
        
        assert original_decoded == loaded_decoded
    
    def test_multiple_roundtrips(self, trained_model, tmp_path):
        """Multiple save/load cycles preserve model."""
        model_path1 = tmp_path / "model1.json"
        model_path2 = tmp_path / "model2.json"
        
        # First roundtrip
        save_model(trained_model, str(model_path1))
        loaded1 = load_model(str(model_path1))
        
        # Second roundtrip
        save_model(loaded1, str(model_path2))
        loaded2 = load_model(str(model_path2))
        
        # Should still work the same
        text = "test text"
        assert trained_model.encode(text) == loaded2.encode(text)


class TestModelHash:
    """Test model hash functionality."""

    def test_hash_is_deterministic(self, trained_model, tmp_path):
        """Model hash is deterministic."""
        model_path1 = tmp_path / "model1.json"
        model_path2 = tmp_path / "model2.json"
        
        save_model(trained_model, str(model_path1))
        save_model(trained_model, str(model_path2))
        
        with open(model_path1) as f:
            data1 = json.load(f)
        with open(model_path2) as f:
            data2 = json.load(f)
        
        # Hashes should match (ignoring created_at)
        assert data1["model_hash"] == data2["model_hash"]
    
    def test_hash_changes_with_vocab(self, tmp_path):
        """Hash changes when vocabulary changes."""
        corpus = tmp_path / "corpus.txt"
        corpus.write_text("hello world python")
        
        # Train two models with different vocab sizes
        model1 = train_bpe(str(corpus), vocab_size=50, seed=42, return_encoder=True)
        model2 = train_bpe(str(corpus), vocab_size=100, seed=42, return_encoder=True)
        
        path1 = tmp_path / "model1.json"
        path2 = tmp_path / "model2.json"
        
        save_model(model1, str(path1))
        save_model(model2, str(path2))
        
        with open(path1) as f:
            hash1 = json.load(f)["model_hash"]
        with open(path2) as f:
            hash2 = json.load(f)["model_hash"]
        
        # Different vocabs should have different hashes
        assert hash1 != hash2


class TestGetModelInfo:
    """Test get_model_info functionality."""

    def test_get_model_info(self, trained_model, tmp_path):
        """Can get model info without full load."""
        model_path = tmp_path / "model.json"
        save_model(trained_model, str(model_path))
        
        info = get_model_info(str(model_path))
        
        assert info is not None
        assert info["algorithm"] == "bpe"
        assert "version" in info
        assert "vocab_size" in info
        assert "num_merges" in info
    
    def test_info_includes_metadata(self, trained_model, tmp_path):
        """Model info includes metadata."""
        model_path = tmp_path / "model.json"
        metadata = {"test": "value"}
        save_model(trained_model, str(model_path), metadata=metadata)
        
        info = get_model_info(str(model_path))
        
        assert "metadata" in info
        assert info["metadata"]["test"] == "value"
    
    def test_info_includes_sizes(self, trained_model, tmp_path):
        """Model info includes size information."""
        model_path = tmp_path / "model.json"
        save_model(trained_model, str(model_path))
        
        info = get_model_info(str(model_path))
        
        assert info["vocab_size"] > 0
        assert info["num_merges"] >= 0


class TestEdgeCases:
    """Test edge cases in I/O."""

    def test_save_with_unicode_path(self, trained_model, tmp_path):
        """Can save to path with Unicode characters."""
        model_path = tmp_path / "模型.json"
        
        save_model(trained_model, str(model_path))
        
        assert model_path.exists()
    
    def test_save_overwrites_existing(self, trained_model, tmp_path):
        """Saving overwrites existing file."""
        model_path = tmp_path / "model.json"
        
        # Save first time
        save_model(trained_model, str(model_path))
        first_size = model_path.stat().st_size
        
        # Save again
        save_model(trained_model, str(model_path))
        second_size = model_path.stat().st_size
        
        # Should overwrite (sizes should be similar)
        assert abs(first_size - second_size) < 1000
    
    def test_load_handles_extra_fields(self, trained_model, tmp_path):
        """Loading handles extra fields gracefully."""
        model_path = tmp_path / "model.json"
        
        save_model(trained_model, str(model_path))
        
        # Add extra field
        with open(model_path) as f:
            data = json.load(f)
        data["extra_field"] = "value"
        
        with open(model_path, 'w') as f:
            json.dump(data, f)
        
        # Should still load
        loaded = load_model(str(model_path))
        assert loaded is not None