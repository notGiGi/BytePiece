__version__ = "0.1.0"

from bytepiece.algorithms.bpe import BPEEncoder, train_bpe
from bytepiece.core.io import get_model_info, load_model, save_model
from bytepiece.core.normalizer import (
    Normalizer,
    NormalizationMode,
    PreTokenizationMode,  # ← NUEVO
    SpacerMode,
)
from bytepiece.core.vocab import MergeRules, SpecialTokens, Vocabulary

__all__ = [
    # Main API
    "train_bpe",
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
    # Metadata
    "__version__",
]