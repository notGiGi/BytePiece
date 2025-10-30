__version__ = "0.2.0"  

from bytepiece.algorithms.bpe import BPEEncoder, train_bpe
from bytepiece.algorithms.fair_bpe import FairBPE, train_fair_bpe
from bytepiece.core.io import get_model_info, load_model, save_model
from bytepiece.core.normalizer import (
    Normalizer,
    NormalizationMode,
    PreTokenizationMode,
    SpacerMode,
)
from bytepiece.core.vocab import MergeRules, SpecialTokens, Vocabulary
from bytepiece.core.code_pretokenizer import CodePreTokenizer, code_pretokenize  # Python-specific

__all__ = [
    # Main API
    "train_bpe",
    "train_fair_bpe",
    "FairBPE",
    "BPEEncoder",
    # I/O
    "save_model",
    "load_model",
    "get_model_info",
    # Core components
    "Normalizer",
    "NormalizationMode",
    "SpacerMode",
    "PreTokenizationMode",
    "Vocabulary",
    "MergeRules",
    "SpecialTokens",
    # Python code pre-tokenization
    "CodePreTokenizer",
    "code_pretokenize",
    # Metadata
    "__version__",
]