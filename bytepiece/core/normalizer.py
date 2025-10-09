import regex as re  
import unicodedata
from enum import Enum
from typing import List


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
    CODE = "code"  # Python code-aware (respects strings, comments, operators)
import regex as re  
import unicodedata
from enum import Enum
from typing import List


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


class PreTokenizationMode(str, Enum):
    """Pre-tokenization strategy to use before BPE."""
    
    NONE = "none"  # No pre-tokenization (character-level from start)
    WHITESPACE = "whitespace"  # Split on whitespace only
    GPT2 = "gpt2"  # GPT-2 style regex pattern (handles contractions, etc.)
    CODE = "code"  # Python code-aware (respects strings, comments, operators)


class Normalizer:
    """Handles Unicode normalization and spacer insertion for tokenization."""
    
    SPACER = "▁"  
    
    GPT2_PATTERN = re.compile(
        r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
        re.UNICODE
    )
    
    def __init__(
        self,
        normalization: NormalizationMode = NormalizationMode.NFKC,
        spacer_mode: SpacerMode = SpacerMode.PREFIX,
        lowercase: bool = False,
        pre_tokenization: PreTokenizationMode = PreTokenizationMode.NONE,
    ):
        self.normalization = normalization
        self.spacer_mode = spacer_mode
        self.lowercase = lowercase
        self.pre_tokenization = pre_tokenization
        
        # Lazy import to avoid circular dependency
        self._code_tokenizer = None
    
    def pre_tokenize(self, text: str) -> List[str]:

        if self.pre_tokenization == PreTokenizationMode.NONE:
            return [text]
        
        elif self.pre_tokenization == PreTokenizationMode.WHITESPACE:
            chunks = []
            current = []
            
            for char in text:
                if char.isspace():
                    if current:
                        chunks.append(''.join(current))
                        current = []
                    chunks.append(char)
                else:
                    current.append(char)
            
            if current:
                chunks.append(''.join(current))
            
            return chunks
        
        elif self.pre_tokenization == PreTokenizationMode.GPT2:
            matches = self.GPT2_PATTERN.findall(text)
            return matches if matches else [text]
        
        elif self.pre_tokenization == PreTokenizationMode.CODE:
        
            if self._code_tokenizer is None:
                from bytepiece.core.code_pretokenizer import CodePreTokenizer
                self._code_tokenizer = CodePreTokenizer()
            
            return self._code_tokenizer.tokenize(text)
        
        else:
            return [text]
    
    def normalize(self, text: str) -> str:
      
        if self.normalization != NormalizationMode.NONE:
            text = unicodedata.normalize(self.normalization.value.upper(), text)
        
        if self.lowercase:
            text = text.lower()
        
        text = self._apply_spacer(text)
        
        return text
    
    def _apply_spacer(self, text: str) -> str:
     
        if self.spacer_mode == SpacerMode.NONE:
            return text
        
        if self.spacer_mode == SpacerMode.PREFIX:
            parts = text.split(" ")
            return self.SPACER + f" {self.SPACER}".join(parts)
        
        elif self.spacer_mode == SpacerMode.SEPARATOR:
            return text.replace(" ", f" {self.SPACER} ")
        
        return text
    
    def to_dict(self) -> dict:
     
        return {
            "normalization": self.normalization.value,
            "spacer_mode": self.spacer_mode.value,
            "lowercase": self.lowercase,
            "pre_tokenization": self.pre_tokenization.value,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Normalizer":
    
        return cls(
            normalization=NormalizationMode(data["normalization"]),
            spacer_mode=SpacerMode(data["spacer_mode"]),
            lowercase=data.get("lowercase", False),
            pre_tokenization=PreTokenizationMode(data.get("pre_tokenization", "none")),
        )