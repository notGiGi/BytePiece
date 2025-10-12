import pytest
from pathlib import Path
import tempfile
import os

import bytepiece
from bytepiece.algorithms.bpe import BPEEncoder, train_bpe
from bytepiece.core.normalizer import Normalizer, NormalizationMode
from bytepiece.core.vocab import SpecialTokens


@pytest.fixture
def simple_corpus(tmp_path):
    """Create a simple test corpus."""
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(
        "hello world\n"
        "hello python\n"
        "world of python\n"
        "hello hello world\n"
    )
    return str(corpus)


@pytest.fixture
def code_corpus(tmp_path):
    """Create a code corpus for testing."""
    corpus = tmp_path / "code.txt"
    corpus.write_text(
        "def hello():\n"
        "    return 'world'\n"
        "def foo():\n"
        "    return 'bar'\n"
        "print(hello())\n"
    )
    return str(corpus)


class TestBPETraining:
    """Test BPE training functionality."""

    def test_train_simple(self, simple_corpus):
        """BPE training works on simple corpus."""
        encoder = train_bpe(
            corpus_path=simple_corpus,
            vocab_size=100,
        )
        
        assert encoder is not None
        assert len(encoder.vocab.tokens) > 0
        assert len(encoder.merge_rules.merges) > 0
    
    def test_train_with_vocab_size(self, simple_corpus):
        """Training respects vocab_size parameter."""
        encoder = train_bpe(
            corpus_path=simple_corpus,
            vocab_size=50,
        )
        
        
        assert 40 <= len(encoder.vocab.tokens) <= 60
    
    def test_train_deterministic(self, simple_corpus):
        """Training is deterministic with same seed."""
        encoder1 = train_bpe(
            corpus_path=simple_corpus,
            vocab_size=100,
            seed=42,
        )
        
        encoder2 = train_bpe(
            corpus_path=simple_corpus,
            vocab_size=100,
            seed=42,
        )
        
        
        assert encoder1.vocab.tokens == encoder2.vocab.tokens
        assert encoder1.merge_rules.merges == encoder2.merge_rules.merges


class TestBPEEncoding:
    """Test BPE encoding/decoding."""

    def test_encode_decode_roundtrip(self, simple_corpus):
        """decode(encode(x)) == x after normalization."""
        encoder = train_bpe(corpus_path=simple_corpus, vocab_size=100)
        
        text = "hello world"
        tokens = encoder.encode(text)
        decoded = encoder.decode(tokens)
        
        
        normalized_text = encoder.normalizer.normalize(text)
        assert decoded == normalized_text
    
    def test_encode_returns_tokens(self, simple_corpus):
        """Encode returns list of token strings."""
        encoder = train_bpe(corpus_path=simple_corpus, vocab_size=100)
        
        tokens = encoder.encode("hello")
        
        assert isinstance(tokens, list)
        assert all(isinstance(t, str) for t in tokens)
        assert len(tokens) > 0
    
    def test_decode_handles_empty(self, simple_corpus):
        """Decode handles empty token list."""
        encoder = train_bpe(corpus_path=simple_corpus, vocab_size=100)
        
        decoded = encoder.decode([])
        assert decoded == ""
    
    def test_encode_unicode(self, simple_corpus):
        """Encode handles Unicode correctly."""
        encoder = train_bpe(corpus_path=simple_corpus, vocab_size=100)
        
        text = "hello 世界 🌍"
        tokens = encoder.encode(text)
        decoded = encoder.decode(tokens)
        
        
        assert len(tokens) > 0
        assert "hello" in decoded or "▁hello" in decoded


class TestByteFallback:
    """Test byte-fallback functionality."""

    def test_byte_fallback_unknown_chars(self, simple_corpus):
        """Byte-fallback handles unknown characters."""
        encoder = train_bpe(corpus_path=simple_corpus, vocab_size=100)
        
        
        text = "xyz123 ñ ü"
        tokens = encoder.encode(text)
        decoded = encoder.decode(tokens)
        
        
        assert len(tokens) > 0
        
        normalized = encoder.normalizer.normalize(text)
        assert decoded == normalized
    
    def test_byte_fallback_coverage(self, simple_corpus):
        """Byte-fallback ensures 100% coverage."""
        encoder = train_bpe(corpus_path=simple_corpus, vocab_size=100)
        
        
        test_cases = [
            "completely new text",
            "numbers 123456789",
            "symbols @#$%^&*()",
            "unicode café résumé",
        ]
        
        for text in test_cases:
            tokens = encoder.encode(text)
            decoded = encoder.decode(tokens)
            
            # Should always encode/decode
            assert len(tokens) > 0
            normalized = encoder.normalizer.normalize(text)
            assert decoded == normalized


class TestSpecialTokens:
    """Test special tokens handling."""

    def test_special_tokens_in_vocab(self, simple_corpus):
        """Special tokens are added to vocabulary."""
        encoder = train_bpe(
            corpus_path=simple_corpus,
            vocab_size=100,
            special_tokens=SpecialTokens(
                pad="<PAD>",
                unk="<UNK>",
                bos="<BOS>",
                eos="<EOS>",
            )
        )
        
        vocab_tokens = encoder.vocab.tokens
        assert "<PAD>" in vocab_tokens
        assert "<UNK>" in vocab_tokens
        assert "<BOS>" in vocab_tokens
        assert "<EOS>" in vocab_tokens
    
    def test_special_tokens_not_merged(self, simple_corpus):
       
        encoder = train_bpe(
            corpus_path=simple_corpus,
            vocab_size=100,
            special_tokens=SpecialTokens(bos="<BOS>", eos="<EOS>")
        )
        
       
        tokens = encoder.encode("<BOS> hello <EOS>")
        
        
        assert any("<BOS>" in t for t in tokens)
        assert any("<EOS>" in t for t in tokens)


class TestNormalization:
    

    def test_nfkc_normalization(self, simple_corpus):
      
        encoder = train_bpe(
            corpus_path=simple_corpus,
            vocab_size=100,
            normalizer=Normalizer(normalization_mode=NormalizationMode.NFKC)
        )
        
        
        text = "ﬁ"  # fi ligature
        tokens = encoder.encode(text)
        decoded = encoder.decode(tokens)
        
     
        assert "fi" in decoded or "▁fi" in decoded
    
    def test_none_normalization(self, simple_corpus):
   
        encoder = train_bpe(
            corpus_path=simple_corpus,
            vocab_size=100,
            normalizer=Normalizer(normalization_mode=NormalizationMode.NONE)
        )
        
        text = "Hello World"
        tokens = encoder.encode(text)
        decoded = encoder.decode(tokens)
        
        
        assert decoded in ["Hello World", "Hello▁World", "▁Hello▁World"]


class TestCodeTokenization:

    def test_python_pretokenization(self, code_corpus):
   
        encoder = train_bpe(
            corpus_path=code_corpus,
            vocab_size=200,
            normalizer=Normalizer(pre_tokenization_mode="python")
        )
        
        code = "def hello():\n    pass"
        tokens = encoder.encode(code)
        
        
        assert len(tokens) > 0
        
      
        decoded = encoder.decode(tokens)
        assert "def" in decoded
        assert "hello" in decoded


class TestEdgeCases:
   

    def test_empty_text(self, simple_corpus):

        encoder = train_bpe(corpus_path=simple_corpus, vocab_size=100)
        
        tokens = encoder.encode("")
        assert tokens == []
        
        decoded = encoder.decode([])
        assert decoded == ""
    
    def test_whitespace_only(self, simple_corpus):
  
        encoder = train_bpe(corpus_path=simple_corpus, vocab_size=100)
        
        tokens = encoder.encode("   \n\t  ")

        assert isinstance(tokens, list)
    
    def test_very_long_text(self, simple_corpus):

        encoder = train_bpe(corpus_path=simple_corpus, vocab_size=100)
        
        long_text = "hello world " * 1000
        tokens = encoder.encode(long_text)
        decoded = encoder.decode(tokens)
        

        assert len(tokens) > 0
        assert "hello" in decoded


class TestConsistency:


    def test_same_input_same_output(self, simple_corpus):

        encoder = train_bpe(corpus_path=simple_corpus, vocab_size=100)
        
        text = "hello world python"
        
        tokens1 = encoder.encode(text)
        tokens2 = encoder.encode(text)
        
        assert tokens1 == tokens2
    
    def test_merge_rules_applied_consistently(self, simple_corpus):

        encoder = train_bpe(corpus_path=simple_corpus, vocab_size=100)
        

        tokens1 = encoder.encode("hello")
        tokens2 = encoder.encode("hello world")
        
        assert len(tokens1) > 0
        assert len(tokens2) >= len(tokens1)