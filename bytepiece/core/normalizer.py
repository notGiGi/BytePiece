"""Unicode normalization and text preprocessing."""

import unicodedata
from enum import Enum
from typing import Optional


class NormalizationMode(str, Enum):
    """Supported Unicode normalization modes."""
    
    NONE = "none"
    NFC = "nfc"
    NFD = "nfd"
    NFKC = "nfkc"
    NFKD = "nfkd"


class SpacerMode(str, Enum):
    """How to handle word boundaries with spacer character."""
    
    PREFIX = "prefix"  # Add ▁ at the start of each word: "▁hello ▁world"
    SEPARATOR = "separator"  # Keep spaces as separate ▁ tokens: "hello ▁ world"
    NONE = "none"  # Don't use spacer at all


class Normalizer:
    """Handles Unicode normalization and spacer insertion for tokenization."""
    
    SPACER = "▁"  # U+2581 LOWER ONE EIGHTH BLOCK
    
    def __init__(
        self,
        normalization: NormalizationMode = NormalizationMode.NFKC,
        spacer_mode: SpacerMode = SpacerMode.PREFIX,
        lowercase: bool = False,
    ):
        """Initialize normalizer.
        
        Args:
            normalization: Unicode normalization form to apply
            spacer_mode: How to handle word boundaries
            lowercase: Whether to lowercase text
        """
        self.normalization = normalization
        self.spacer_mode = spacer_mode
        self.lowercase = lowercase
    
    def normalize(self, text: str) -> str:
        """Apply full normalization pipeline to text.
        
        Args:
            text: Input text to normalize
            
        Returns:
            Normalized text with spacers applied
        """
        # Apply Unicode normalization
        if self.normalization != NormalizationMode.NONE:
            text = unicodedata.normalize(self.normalization.value.upper(), text)
        
        # Apply lowercasing
        if self.lowercase:
            text = text.lower()
        
        # Apply spacer handling
        text = self._apply_spacer(text)
        
        return text
    
    def _apply_spacer(self, text: str) -> str:
        """Apply spacer character based on mode.
        
        Args:
            text: Text to process
            
        Returns:
            Text with spacers applied
        """
        if self.spacer_mode == SpacerMode.NONE:
            return text
        
        if self.spacer_mode == SpacerMode.PREFIX:
            # Replace spaces with spacer at word boundaries
            # "hello world" → "▁hello▁world"
            parts = text.split(" ")
            return self.SPACER + (self.SPACER.join(parts))
        
        if self.spacer_mode == SpacerMode.SEPARATOR:
            # Keep spaces as separate spacer tokens
            # "hello world" → "hello▁world"
            return text.replace(" ", self.SPACER)
        
        return text
    
    def denormalize(self, text: str) -> str:
        """Remove spacers and restore original spacing.
        
        Args:
            text: Normalized text with spacers
            
        Returns:
            Text with spacers converted back to spaces
        """
        if self.spacer_mode == SpacerMode.NONE:
            return text
        
        if self.spacer_mode == SpacerMode.PREFIX:
            # Remove leading spacer and convert others to spaces
            text = text.lstrip(self.SPACER)
            return text.replace(self.SPACER, " ")
        
        if self.spacer_mode == SpacerMode.SEPARATOR:
            return text.replace(self.SPACER, " ")
        
        return text
    
    def to_dict(self) -> dict:
        """Serialize normalizer configuration to dict.
        
        Returns:
            Dictionary with normalizer settings
        """
        return {
            "normalization": self.normalization.value,
            "spacer_mode": self.spacer_mode.value,
            "lowercase": self.lowercase,
        }
    
    @classmethod
    def from_dict(cls, config: dict) -> "Normalizer":
        """Create normalizer from configuration dict.
        
        Args:
            config: Dictionary with normalizer settings
            
        Returns:
            Configured Normalizer instance
        """
        return cls(
            normalization=NormalizationMode(config["normalization"]),
            spacer_mode=SpacerMode(config["spacer_mode"]),
            lowercase=config.get("lowercase", False),
        )