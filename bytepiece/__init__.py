"""BytePiece - Educational, production-grade tokenizer implementation."""

__version__ = "0.1.0"

from bytepiece.algorithms.bpe import BPEEncoder, train_bpe
from bytepiece.core.io import get_model_info, load_model, save_model
from bytepiece.core.normalizer import Normalizer, NormalizationMode, SpacerMode
from bytepiece.core.vocab import MergeRules, Vocabulary

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
    "Vocabulary",
    "MergeRules",
    # Metadata
    "__version__",
]