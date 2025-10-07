import re
import unicodedata
from enum import Enum
from typing import List, Optional


class NormalizationMode(str, Enum):
    
    NONE = "none"
    NFC = "nfc"
    NFD = "nfd"
    NFKC = "nfkc"
    NFKD = "nfkd"


class SpacerMode(str, Enum):
    
    PREFIX = "prefix"  # Add ▁ at the start of each word: "▁hello ▁world"
    SEPARATOR = "separator"  # Keep spaces as separate ▁ tokens: "hello ▁ world"
    NONE = "none"  # Don't use spacer at all


class PreTokenizationMode(str, Enum):
    
    NONE = "none"  # No pre-tokenization (character-level from start)
    WHITESPACE = "whitespace"  # Split on whitespace only
    GPT2 = "gpt2"  # GPT-2 style regex pattern (handles contractions, etc.)


class Normalizer:
    """Handles Unicode normalization and spacer insertion for tokenization."""
    
    SPACER = "▁"  # U+2581 LOWER ONE EIGHTH BLOCK
    
    def __init__(
        self,
        normalization: NormalizationMode = NormalizationMode.NFKC,
        spacer_mode: SpacerMode = SpacerMode.PREFIX,
        lowercase: bool = False,
    ):

        self.normalization = normalization
        self.spacer_mode = spacer_mode
        self.lowercase = lowercase
    
    def normalize(self, text: str) -> str:
        if self.normalization != NormalizationMode.NONE:
            text = unicodedata.normalize(self.normalization.value.upper(), text)
        
        # Apply lowercasing
        if self.lowercase:
            text = text.lower()
        
        # Apply spacer handling
        text = self._apply_spacer(text)
        
        return text
    
    def _apply_spacer(self, text: str) -> str:
        if self.spacer_mode == SpacerMode.NONE:
            return text
        
        if self.spacer_mode == SpacerMode.PREFIX:

            parts = text.split(" ")
            return self.SPACER + (self.SPACER.join(parts))
        
        if self.spacer_mode == SpacerMode.SEPARATOR:

            return text.replace(" ", self.SPACER)
        
        return text
    
    def denormalize(self, text: str) -> str:
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

        return {
            "normalization": self.normalization.value,
            "spacer_mode": self.spacer_mode.value,
            "lowercase": self.lowercase,
        }
    
    @classmethod
    def from_dict(cls, config: dict) -> "Normalizer":
        return cls(
            normalization=NormalizationMode(config["normalization"]),
            spacer_mode=SpacerMode(config["spacer_mode"]),
            lowercase=config.get("lowercase", False),
        )