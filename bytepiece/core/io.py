"""Model serialization and loading."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from bytepiece.algorithms.bpe import BPEEncoder
from bytepiece.core.normalizer import Normalizer
from bytepiece.core.vocab import MergeRules, Vocabulary
from bytepiece.utils.hashing import compute_model_hash


def save_model(
    encoder: BPEEncoder,
    path: str,
    metadata: Dict[str, Any] = None,
) -> None:
    """Save trained model to JSON file.
    
    Args:
        encoder: Trained BPE encoder
        path: Output file path
        metadata: Optional metadata to include
    """
    model_dict = {
        "algorithm": "bpe",
        "version": "0.1.0",
        "created_at": datetime.utcnow().isoformat(),
        "normalizer": encoder.normalizer.to_dict(),
        "byte_fallback": encoder.vocab.byte_fallback,
        "vocab": encoder.vocab.to_dict(),
        "merges": encoder.merge_rules.to_dict(),
    }
    
    # Add optional metadata
    if metadata:
        model_dict["metadata"] = metadata
    
    # Compute model hash (excluding non-deterministic fields)
    model_dict["model_hash"] = compute_model_hash(model_dict)
    
    # Write to file
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(model_dict, f, indent=2, ensure_ascii=True)


def load_model(path: str) -> BPEEncoder:
    """Load trained model from JSON file.
    
    Args:
        path: Path to model file
        
    Returns:
        Loaded BPE encoder
        
    Raises:
        ValueError: If model format is invalid
        FileNotFoundError: If file doesn't exist
    """
    with open(path, 'r', encoding='utf-8') as f:
        model_dict = json.load(f)
    
    # Verify algorithm
    if model_dict.get("algorithm") != "bpe":
        raise ValueError(f"Unsupported algorithm: {model_dict.get('algorithm')}")
    
    # Verify hash if present
    if "model_hash" in model_dict:
        expected_hash = model_dict["model_hash"]
        actual_hash = compute_model_hash(model_dict)
        if expected_hash != actual_hash:
            print(f"Warning: Model hash mismatch. Expected {expected_hash}, got {actual_hash}")
    
    # Reconstruct components
    normalizer = Normalizer.from_dict(model_dict["normalizer"])
    vocab = Vocabulary.from_dict(model_dict["vocab"])
    merge_rules = MergeRules.from_dict(model_dict["merges"])
    
    # Create encoder
    encoder = BPEEncoder(
        vocab=vocab,
        merge_rules=merge_rules,
        normalizer=normalizer,
    )
    
    return encoder


def get_model_info(path: str) -> Dict[str, Any]:
    """Get model information without fully loading it.
    
    Args:
        path: Path to model file
        
    Returns:
        Dictionary with model metadata
    """
    with open(path, 'r', encoding='utf-8') as f:
        model_dict = json.load(f)
    
    return {
        "algorithm": model_dict.get("algorithm"),
        "version": model_dict.get("version"),
        "created_at": model_dict.get("created_at"),
        "vocab_size": len(model_dict.get("vocab", {}).get("tokens", [])),
        "num_merges": len(model_dict.get("merges", {}).get("merges", [])),
        "byte_fallback": model_dict.get("byte_fallback"),
        "normalizer": model_dict.get("normalizer"),
        "model_hash": model_dict.get("model_hash"),
        "metadata": model_dict.get("metadata"),
    }