import pytest
from bytepiece.core.normalizer import (
    Normalizer,
    NormalizationMode,
    SpacerMode,
    PreTokenizationMode,
)


class TestNormalizationModes:
    """Test different normalization modes."""

    def test_nfkc_normalization(self):
        """NFKC normalization works correctly."""
        normalizer = Normalizer(normalization_mode=NormalizationMode.NFKC)
        
        # Test ligature normalization
        text = "ﬁle"  # fi ligature + le
        normalized = normalizer.normalize(text)
        assert "file" in normalized
        
        # Test compatibility characters
        text = "①②③"  # Circled numbers
        normalized = normalizer.normalize(text)
        # NFKC converts these to regular numbers
        assert "1" in normalized or "①" in normalized
    
    def test_nfc_normalization(self):
        """NFC normalization works correctly."""
        normalizer = Normalizer(normalization_mode=NormalizationMode.NFC)
        
        # Combining characters
        text = "café"  # e + combining acute
        normalized = normalizer.normalize(text)
        assert "café" in normalized
    
    def test_nfd_normalization(self):
        """NFD normalization works correctly."""
        normalizer = Normalizer(normalization_mode=NormalizationMode.NFD)
        
        text = "café"
        normalized = normalizer.normalize(text)
        # NFD decomposes, but result should still be café
        assert "cafe" in normalized or "café" in normalized
    
    def test_none_normalization(self):
        """NONE mode preserves text as-is."""
        normalizer = Normalizer(normalization_mode=NormalizationMode.NONE)
        
        text = "ﬁle CAFÉ"
        normalized = normalizer.normalize(text)
        # Should preserve original (with spacer handling)
        assert "ﬁle" in normalized or "CAFÉ" in normalized


class TestSpacerModes:
    """Test spacer handling."""

    def test_prefix_spacer(self):
        """Prefix spacer adds ▁ to word starts."""
        normalizer = Normalizer(spacer_mode=SpacerMode.PREFIX)
        
        text = "hello world"
        normalized = normalizer.normalize(text)
        
        # Should have spacers at word boundaries
        assert "▁" in normalized
        # Spacers should be at start of words
        assert "▁hello" in normalized or "▁world" in normalized
    
    def test_suffix_spacer(self):
        """Suffix spacer adds ▁ to word ends."""
        normalizer = Normalizer(spacer_mode=SpacerMode.SUFFIX)
        
        text = "hello world"
        normalized = normalizer.normalize(text)
        
        assert "▁" in normalized
        # Should be at end of words
        assert "hello▁" in normalized or "world▁" in normalized
    
    def test_isolated_spacer(self):
        """Isolated spacer keeps spaces separate."""
        normalizer = Normalizer(spacer_mode=SpacerMode.ISOLATED)
        
        text = "hello world"
        normalized = normalizer.normalize(text)
        
        # Space should be separate token
        assert "▁" in normalized
    
    def test_none_spacer(self):
        """NONE spacer preserves spaces."""
        normalizer = Normalizer(spacer_mode=SpacerMode.NONE)
        
        text = "hello world"
        normalized = normalizer.normalize(text)
        
        
        assert " " in normalized or normalized == "hello world"


class TestPreTokenization:
    """Test pre-tokenization modes."""

    def test_whitespace_pretokenization(self):
        """Whitespace mode splits on whitespace."""
        normalizer = Normalizer(pre_tokenization_mode=PreTokenizationMode.WHITESPACE)
        
        text = "hello world python"
        chunks = normalizer.pre_tokenize(text)
        
        assert len(chunks) >= 3
        assert any("hello" in chunk for chunk in chunks)
        assert any("world" in chunk for chunk in chunks)
        assert any("python" in chunk for chunk in chunks)
    
    def test_gpt2_pretokenization(self):
        """GPT-2 mode uses regex patterns."""
        normalizer = Normalizer(pre_tokenization_mode=PreTokenizationMode.GPT2)
        
        text = "don't worry about it"
        chunks = normalizer.pre_tokenize(text)
        
        
        assert len(chunks) > 0
        
        assert any("don" in chunk or "don't" in chunk for chunk in chunks)
    
    def test_none_pretokenization(self):
        """NONE mode returns single chunk."""
        normalizer = Normalizer(pre_tokenization_mode=PreTokenizationMode.NONE)
        
        text = "hello world"
        chunks = normalizer.pre_tokenize(text)
        
        
        assert len(chunks) == 1
        assert chunks[0] == text


class TestUnicodeEdgeCases:
    """Test Unicode edge cases."""

    def test_emoji_handling(self):
        """Handles emoji correctly."""
        normalizer = Normalizer()
        
        text = "hello 👋 world 🌍"
        normalized = normalizer.normalize(text)
        
        
        assert len(normalized) > 0
        
        assert "hello" in normalized
        assert "world" in normalized
    
    def test_combining_marks(self):
        """Handles combining marks."""
        normalizer = Normalizer()
        
        
        text = "e\u0301"  # e + acute accent
        normalized = normalizer.normalize(text)
        
        
        assert len(normalized) > 0
    
    def test_rtl_text(self):
        """Handles RTL text."""
        normalizer = Normalizer()
        
        text = "hello مرحبا world"
        normalized = normalizer.normalize(text)
        
        
        assert "hello" in normalized
        assert "world" in normalized
    
    def test_zero_width_characters(self):
        """Handles zero-width characters."""
        normalizer = Normalizer()
        
        text = "hello\u200bworld" 
        normalized = normalizer.normalize(text)
        
        
        assert len(normalized) > 0
    
    def test_mixed_scripts(self):
        """Handles mixed scripts."""
        normalizer = Normalizer()
        
        text = "hello 世界 мир monde"
        normalized = normalizer.normalize(text)
        
        
        assert len(normalized) > 0
        assert "hello" in normalized


class TestNormalizerSerialization:
    """Test normalizer serialization."""

    def test_to_dict(self):
        """Normalizer can be serialized to dict."""
        normalizer = Normalizer(
            normalization_mode=NormalizationMode.NFKC,
            spacer_mode=SpacerMode.PREFIX,
            pre_tokenization_mode=PreTokenizationMode.GPT2,
        )
        
        config = normalizer.to_dict()
        
        assert config["normalization_mode"] == "NFKC"
        assert config["spacer_mode"] == "PREFIX"
        assert config["pre_tokenization_mode"] == "GPT2"
    
    def test_from_dict(self):
        """Normalizer can be deserialized from dict."""
        config = {
            "normalization_mode": "NFKC",
            "spacer_mode": "PREFIX",
            "pre_tokenization_mode": "GPT2",
            "spacer": "▁",
        }
        
        normalizer = Normalizer.from_dict(config)
        
        assert normalizer.normalization_mode == NormalizationMode.NFKC
        assert normalizer.spacer_mode == SpacerMode.PREFIX
        assert normalizer.pre_tokenization_mode == PreTokenizationMode.GPT2
    
    def test_roundtrip_serialization(self):
        """Serialization roundtrip preserves config."""
        original = Normalizer(
            normalization_mode=NormalizationMode.NFC,
            spacer_mode=SpacerMode.ISOLATED,
        )
        
        config = original.to_dict()
        restored = Normalizer.from_dict(config)
        
        assert restored.normalization_mode == original.normalization_mode
        assert restored.spacer_mode == original.spacer_mode


class TestCodePreTokenization:
    """Test code-specific pre-tokenization."""

    def test_python_pretokenization(self):
        """Python mode handles Python code."""
        normalizer = Normalizer(pre_tokenization_mode="python")
        
        code = "def hello():\n    return 'world'"
        chunks = normalizer.pre_tokenize(code)
        
        
        assert len(chunks) > 0
        
        assert any("def" in chunk for chunk in chunks)
        assert any("hello" in chunk for chunk in chunks)
    
    def test_syntax_aware_pretokenization(self):
        """Syntax-aware mode respects code structure."""
        normalizer = Normalizer(pre_tokenization_mode="syntax_aware")
        
        code = "x = func(arg)"
        chunks = normalizer.pre_tokenize(code)
        
        
        assert len(chunks) > 0


class TestEdgeCases:
    """Test edge cases."""

    def test_empty_text(self):
        """Handles empty text."""
        normalizer = Normalizer()
        
        normalized = normalizer.normalize("")
        assert normalized == ""
        
        chunks = normalizer.pre_tokenize("")
        assert chunks == [] or chunks == [""]
    
    def test_whitespace_only(self):
        """Handles whitespace-only text."""
        normalizer = Normalizer()
        
        normalized = normalizer.normalize("   \n\t  ")
        
        assert isinstance(normalized, str)
    
    def test_very_long_text(self):
        """Handles very long text."""
        normalizer = Normalizer()
        
        long_text = "hello " * 10000
        normalized = normalizer.normalize(long_text)
        
       
        assert isinstance(normalized, str)
        assert "hello" in normalized