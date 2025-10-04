"""Vocabulary management with byte-fallback support."""

from typing import Dict, List, Optional, Set, Tuple


class Vocabulary:
    """Manages token vocabulary with byte-fallback for full Unicode coverage."""
    
    def __init__(self, byte_fallback: bool = True):
        """Initialize vocabulary.
        
        Args:
            byte_fallback: If True, add 256 byte tokens for full coverage
        """
        self.byte_fallback = byte_fallback
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        
        # Initialize with byte tokens if enabled
        if byte_fallback:
            self._init_byte_tokens()
    
    def _init_byte_tokens(self) -> None:
        """Add 256 byte tokens <0x00> through <0xFF> to vocabulary."""
        for i in range(256):
            token = f"<0x{i:02X}>"
            self.token_to_id[token] = i
            self.id_to_token[i] = token
    
    def add_token(self, token: str) -> int:
        """Add a token to vocabulary.
        
        Args:
            token: Token string to add
            
        Returns:
            Token ID (new or existing)
        """
        if token in self.token_to_id:
            return self.token_to_id[token]
        
        token_id = len(self.token_to_id)
        self.token_to_id[token] = token_id
        self.id_to_token[token_id] = token
        return token_id
    
    def add_tokens(self, tokens: List[str]) -> List[int]:
        """Add multiple tokens to vocabulary.
        
        Args:
            tokens: List of token strings
            
        Returns:
            List of token IDs
        """
        return [self.add_token(token) for token in tokens]
    
    def get_id(self, token: str) -> Optional[int]:
        """Get ID for a token.
        
        Args:
            token: Token string
            
        Returns:
            Token ID or None if not in vocabulary
        """
        return self.token_to_id.get(token)
    
    def get_token(self, token_id: int) -> Optional[str]:
        """Get token string for an ID.
        
        Args:
            token_id: Token ID
            
        Returns:
            Token string or None if ID not found
        """
        return self.id_to_token.get(token_id)
    
    def encode_with_fallback(self, text: str) -> List[str]:
        """Encode text to tokens using byte-fallback for unknown characters.
        
        This is a character-level fallback that converts any character
        not in the vocabulary to its UTF-8 byte representation.
        
        Args:
            text: Text to encode
            
        Returns:
            List of tokens (mix of vocabulary tokens and byte tokens)
        """
        if not self.byte_fallback:
            # Without byte-fallback, just return characters as-is
            return list(text)
        
        tokens = []
        for char in text:
            # Check if character itself is a token
            if char in self.token_to_id:
                tokens.append(char)
            else:
                # Convert to UTF-8 bytes and use byte tokens
                char_bytes = char.encode('utf-8')
                for byte_val in char_bytes:
                    byte_token = f"<0x{byte_val:02X}>"
                    tokens.append(byte_token)
        
        return tokens
    
    def decode_bytes(self, tokens: List[str]) -> str:
        """Decode tokens back to text, handling byte tokens.
        
        Args:
            tokens: List of token strings (may include byte tokens)
            
        Returns:
            Decoded text
        """
        result = []
        byte_buffer = []
        
        for token in tokens:
            # Check if this is a byte token
            if self._is_byte_token(token):
                # Extract byte value
                byte_val = int(token[3:5], 16)  # <0xAB> → 0xAB
                byte_buffer.append(byte_val)
            else:
                # Flush any pending bytes
                if byte_buffer:
                    try:
                        decoded = bytes(byte_buffer).decode('utf-8')
                        result.append(decoded)
                    except UnicodeDecodeError:
                        # Invalid UTF-8 sequence, keep as replacement char
                        result.append('�' * len(byte_buffer))
                    byte_buffer = []
                
                # Add regular token
                result.append(token)
        
        # Flush remaining bytes
        if byte_buffer:
            try:
                decoded = bytes(byte_buffer).decode('utf-8')
                result.append(decoded)
            except UnicodeDecodeError:
                result.append('�' * len(byte_buffer))
        
        return ''.join(result)
    
    def _is_byte_token(self, token: str) -> bool:
        """Check if token is a byte token like <0xAB>.
        
        Args:
            token: Token string
            
        Returns:
            True if token is a byte token
        """
        return (
            len(token) == 6 and
            token.startswith('<0x') and
            token.endswith('>')
        )
    
    def __len__(self) -> int:
        """Return vocabulary size."""
        return len(self.token_to_id)
    
    def __contains__(self, token: str) -> bool:
        """Check if token is in vocabulary."""
        return token in self.token_to_id
    
    def to_dict(self) -> dict:
        """Serialize vocabulary to dict.
        
        Returns:
            Dictionary with vocab data
        """
        return {
            "byte_fallback": self.byte_fallback,
            "tokens": list(self.id_to_token.values()),
        }
    
    @classmethod
    def from_dict(cls, config: dict) -> "Vocabulary":
        """Create vocabulary from dict.
        
        Args:
            config: Dictionary with vocab data
            
        Returns:
            Vocabulary instance
        """
        vocab = cls(byte_fallback=config["byte_fallback"])
        
        # Add tokens (skipping byte tokens if they're already there)
        start_idx = 256 if vocab.byte_fallback else 0
        for token in config["tokens"][start_idx:]:
            vocab.add_token(token)
        
        return vocab


class MergeRules:
    """Manages BPE merge rules with deterministic ordering."""
    
    def __init__(self):
        """Initialize empty merge rules."""
        self.merges: List[Tuple[str, str]] = []
        self.merge_ranks: Dict[Tuple[str, str], int] = {}
    
    def add_merge(self, pair: Tuple[str, str], rank: Optional[int] = None) -> None:
        """Add a merge rule.
        
        Args:
            pair: Pair of tokens to merge (left, right)
            rank: Optional explicit rank (defaults to current length)
        """
        if pair in self.merge_ranks:
            return  # Already exists
        
        if rank is None:
            rank = len(self.merges)
        
        self.merges.append(pair)
        self.merge_ranks[pair] = rank
    
    def get_rank(self, pair: Tuple[str, str]) -> Optional[int]:
        """Get rank for a merge pair.
        
        Args:
            pair: Token pair
            
        Returns:
            Rank or None if pair not in merges
        """
        return self.merge_ranks.get(pair)
    
    def __len__(self) -> int:
        """Return number of merge rules."""
        return len(self.merges)
    
    def to_dict(self) -> dict:
        """Serialize merge rules to dict.
        
        Returns:
            Dictionary with merge data
        """
        return {
            "merges": [[left, right] for left, right in self.merges],
        }
    
    @classmethod
    def from_dict(cls, config: dict) -> "MergeRules":
        """Create merge rules from dict.
        
        Args:
            config: Dictionary with merge data
            
        Returns:
            MergeRules instance
        """
        rules = cls()
        for rank, (left, right) in enumerate(config["merges"]):
            rules.add_merge((left, right), rank)
        return rules