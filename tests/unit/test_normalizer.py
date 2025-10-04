"""Unit tests for normalizer."""

import pytest

from bytepiece.core.normalizer import Normalizer, NormalizationMode, SpacerMode


def test_normalizer_spacer_prefix():
    """Test spacer in prefix mode."""
    normalizer = Normalizer(spacer_mode=SpacerMode.PREFIX)
    
    text = "hello world"
    normalized = normalizer.normalize(text)
    
    # Should have spacer prefix
    assert normalized.startswith(Normalizer.SPACER)
    assert Normalizer.SPACER in normalized
    
    # Should denormalize back
    denormalized = normalizer.denormalize(normalized)
    assert denormalized == text


def test_normalizer_round_trip():
    """Test normalize -> denormalize preserves content."""
    for spacer_mode in [SpacerMode.PREFIX, SpacerMode.SEPARATOR, SpacerMode.NONE]:
        normalizer = Normalizer(spacer_mode=spacer_mode)
        
        text = "hello world test"
        normalized = normalizer.normalize(text)
        denormalized = normalizer.denormalize(normalized)
        
        assert denormalized == text
