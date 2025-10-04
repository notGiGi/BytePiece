"""Deterministic hashing for model fingerprinting."""

import hashlib
import json
from typing import Any, Dict


def compute_model_hash(model_dict: Dict[str, Any]) -> str:
    """Compute deterministic SHA256 hash of model configuration.
    
    This excludes non-deterministic fields like timestamps and includes
    only the fields that affect model behavior.
    
    Args:
        model_dict: Model configuration dictionary
        
    Returns:
        Hex string of SHA256 hash
    """
    # Extract only deterministic fields
    hashable_fields = {
        "algorithm": model_dict.get("algorithm"),
        "version": model_dict.get("version"),
        "normalizer": model_dict.get("normalizer"),
        "byte_fallback": model_dict.get("byte_fallback"),
        "vocab": model_dict.get("vocab"),
        "merges": model_dict.get("merges"),
        "seed": model_dict.get("seed"),
    }
    
    # Remove None values
    hashable_fields = {k: v for k, v in hashable_fields.items() if v is not None}
    
    # Create deterministic JSON (sorted keys)
    json_str = json.dumps(hashable_fields, sort_keys=True, ensure_ascii=True)
    
    # Compute SHA256
    return hashlib.sha256(json_str.encode('utf-8')).hexdigest()


def compute_corpus_hash(corpus_path: str, sample_size: int = 10000) -> str:
    """Compute hash of training corpus for reproducibility tracking.
    
    For large corpora, samples the first N bytes.
    
    Args:
        corpus_path: Path to corpus file
        sample_size: Number of bytes to sample
        
    Returns:
        Hex string of SHA256 hash
    """
    hasher = hashlib.sha256()
    
    try:
        with open(corpus_path, 'rb') as f:
            # Read in chunks to handle large files
            chunk_size = 8192
            bytes_read = 0
            
            while bytes_read < sample_size:
                chunk = f.read(min(chunk_size, sample_size - bytes_read))
                if not chunk:
                    break
                hasher.update(chunk)
                bytes_read += len(chunk)
        
        return hasher.hexdigest()
    except FileNotFoundError:
        return "unknown"