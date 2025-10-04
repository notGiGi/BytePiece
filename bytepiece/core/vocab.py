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
        """Add 256 byte tokens <0x00> through <0xFF> to vocabulary.
        
        Why exactly 256?
        ----------------
        - 1 byte = 8 bits = 2^8 = 256 possible values (0-255)
        - Every byte can represent values from 0x00 (0) to 0xFF (255)
        - Any digital data (text, images, etc.) is stored as bytes
        - With 256 tokens, we can represent ANY byte that exists
        
        Example: Chinese character '你' in UTF-8
        -----------------------------------------
        '你' (U+4F60) → UTF-8 bytes: [0xE4, 0xBD, 0xA0]
        
        Each byte is in range 0-255:
        - 0xE4 = 228 (decimal) → token "<0xE4>"
        - 0xBD = 189 (decimal) → token "<0xBD>"
        - 0xA0 = 160 (decimal) → token "<0xA0>"
        
        This gives us 100% coverage of any Unicode text by falling back
        to its UTF-8 byte representation when characters aren't in vocab.
        """
        for i in range(256):  # Iterate over all possible byte values (0-255)
            token = f"<0x{i:02X}>"  # Format as hex: <0x00>, <0x41>, <0xFF>, etc.
            self.token_to_id[token] = i  # Map token string to byte value
            self.id_to_token[i] = token  # Map byte value to token string
    
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
        """Add 256 byte tokens <0x00> through <0xFF> to vocabulary.
        
        Why exactly 256?
        ----------------
        - 1 byte = 8 bits = 2^8 = 256 possible values (0-255)
        - Every byte can represent values from 0x00 (0) to 0xFF (255)
        - Any digital data (text, images, etc.) is stored as bytes
        - With 256 tokens, we can represent ANY byte that exists
        
        Example: Chinese character '你' in UTF-8
        -----------------------------------------
        '你' (U+4F60) → UTF-8 bytes: [0xE4, 0xBD, 0xA0]
        
        Each byte is in range 0-255:
        - 0xE4 = 228 (decimal) → token "<0xE4>"
        - 0xBD = 189 (decimal) → token "<0xBD>"
        - 0xA0 = 160 (decimal) → token "<0xA0>"
        
        This gives us 100% coverage of any Unicode text by falling back
        to its UTF-8 byte representation when characters aren't in vocab.
        """
        for i in range(256):  # Iterate over all possible byte values (0-255)
            token = f"<0x{i:02X}>"  # Format as hex: <0x00>, <0x41>, <0xFF>, etc.
            self.token_to_id[token] = i  # Map token string to byte value
            self.id_to_token[i] = token  # Map byte value to token string
    
    def add_token(self, token: str) -> int:
        """Add a token to vocabulary with auto-incremented ID.
        
        How it works:
        -------------
        1. Check if token already exists → return existing ID (idempotent)
        2. If new token → assign next available ID (= current vocab size)
        3. Store bidirectional mapping: token ↔ ID
        
        ID assignment strategy:
        ----------------------
        IDs are assigned sequentially starting from 0 (or 256 with byte_fallback):
        
        Example without byte_fallback:
        - add_token("hello") → ID 0
        - add_token("world") → ID 1
        - add_token("hello") → ID 0 (already exists, returns same ID)
        - add_token("test")  → ID 2
        
        Example WITH byte_fallback (256 IDs already used):
        - Vocab starts with byte tokens: <0x00>=0, ..., <0xFF>=255
        - add_token("hello") → ID 256
        - add_token("world") → ID 257
        
        Why use len(token_to_id) for new IDs:
        -------------------------------------
        - Always gives the next available ID
        - No gaps in ID sequence
        - Efficient: O(1) lookup
        
        Args:
            token: Token string to add (e.g., "hello", "▁world", "ing")
            
        Returns:
            Token ID (integer): existing ID if token already in vocab,
                                new ID if token is new
        
        Example:
        --------
        >>> vocab = Vocabulary(byte_fallback=False)
        >>> vocab.add_token("hello")
        0
        >>> vocab.add_token("world")
        1
        >>> vocab.add_token("hello")  # Idempotent
        0
        >>> len(vocab)
        2
        """
        # Check if token already exists (idempotent operation)
        if token in self.token_to_id:
            return self.token_to_id[token]
        
        # Assign next available ID (= current vocabulary size)
        token_id = len(self.token_to_id)
        
        # Store bidirectional mapping
        self.token_to_id[token] = token_id  # token → ID
        self.id_to_token[token_id] = token  # ID → token
        
        return token_id
    
    def add_tokens(self, tokens: List[str]) -> List[int]:
        """Add multiple tokens to vocabulary in batch.
        
        This is a convenience method that calls add_token() for each token.
        Useful during BPE training when adding many learned tokens at once.
        
        Behavior:
        ---------
        - Preserves order of input tokens
        - Skips duplicates automatically (add_token is idempotent)
        - Returns ID for each token (existing or new)
        
        Example during BPE training:
        ---------------------------
        After initial character-level split, we might have:
        >>> chars = ['h', 'e', 'l', 'l', 'o', ' ', 'w', 'o', 'r', 'l', 'd']
        >>> unique_chars = sorted(set(chars))  # ['', 'd', 'e', 'h', 'l', 'o', 'r', 'w']
        >>> ids = vocab.add_tokens(unique_chars)
        >>> ids
        [256, 257, 258, 259, 260, 261, 262, 263]  # If byte_fallback=True
        
        Then after learning merges:
        >>> learned_merges = ['ll', 'lo', 'he']
        >>> merge_ids = vocab.add_tokens(learned_merges)
        >>> merge_ids
        [264, 265, 266]
        
        Args:
            tokens: List of token strings to add
            
        Returns:
            List of token IDs (one per input token, in same order)
        
        Time complexity: O(n) where n = len(tokens)
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
        
        Process:
        --------
        1. For each character in text:
           - If character is in vocab → use it directly
           - If character NOT in vocab → convert to UTF-8 bytes → use byte tokens
        
        Example with mixed content:
        ---------------------------
        Input: "Hi你好"
        
        'H' → in vocab → "H"
        'i' → in vocab → "i"
        '你' → NOT in vocab → UTF-8: [0xE4, 0xBD, 0xA0]
                            → tokens: ["<0xE4>", "<0xBD>", "<0xA0>"]
        '好' → NOT in vocab → UTF-8: [0xE5, 0xA5, 0xBD]
                            → tokens: ["<0xE5>", "<0xA5>", "<0xBD>"]
        
        Result: ["H", "i", "<0xE4>", "<0xBD>", "<0xA0>", "<0xE5>", "<0xA5>", "<0xBD>"]
        
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
                # Example: '你' → bytes [0xE4, 0xBD, 0xA0]
                char_bytes = char.encode('utf-8')
                for byte_val in char_bytes:
                    # Create byte token: <0xE4>, <0xBD>, etc.
                    byte_token = f"<0x{byte_val:02X}>"
                    tokens.append(byte_token)
        
        return tokens
    
    def decode_bytes(self, tokens: List[str]) -> str:
        """Decode tokens back to text, handling byte tokens.
        
        This reverses the byte-fallback encoding by:
        1. Collecting consecutive byte tokens (e.g., "<0xE4>", "<0xBD>", "<0xA0>")
        2. Converting them back to bytes [0xE4, 0xBD, 0xA0]
        3. Decoding as UTF-8 to get the original character
        
        Process with example:
        ---------------------
        Input tokens: ["H", "i", "<0xE4>", "<0xBD>", "<0xA0>"]
        
        1. "H" → not a byte token → add to result: "H"
        2. "i" → not a byte token → add to result: "Hi"
        3. "<0xE4>" → byte token → store in buffer: [0xE4]
        4. "<0xBD>" → byte token → store in buffer: [0xE4, 0xBD]
        5. "<0xA0>" → byte token → store in buffer: [0xE4, 0xBD, 0xA0]
        6. End of tokens → decode buffer as UTF-8: bytes → '你'
        7. Final result: "Hi你"
        
        Why buffer is needed:
        --------------------
        UTF-8 is a variable-length encoding:
        - ASCII chars: 1 byte  (e.g., 'A' = 0x41)
        - Latin ext:  2 bytes  (e.g., 'é' = 0xC3 0xA9)
        - Chinese:    3 bytes  (e.g., '你' = 0xE4 0xBD 0xA0)
        - Emoji:      4 bytes  (e.g., '🚀' = 0xF0 0x9F 0x9A 0x80)
        
        We need to collect all bytes before decoding.
        
        Args:
            tokens: List of token strings (may include byte tokens)
            
        Returns:
            Decoded text
        """
        result = []
        byte_buffer = []  # Accumulates consecutive byte tokens
        
        for token in tokens:
            # Check if this is a byte token (format: <0xXX>)
            if self._is_byte_token(token):
                # Extract byte value: "<0xE4>" → 0xE4 → 228
                byte_val = int(token[3:5], 16)  # Parse hex digits
                byte_buffer.append(byte_val)
            else:
                # Not a byte token, flush any pending bytes first
                if byte_buffer:
                    try:
                        # Convert byte list to bytes object and decode UTF-8
                        # Example: [0xE4, 0xBD, 0xA0] → bytes → '你'
                        decoded = bytes(byte_buffer).decode('utf-8')
                        result.append(decoded)
                    except UnicodeDecodeError:
                        # Invalid UTF-8 sequence, use replacement character
                        result.append('�' * len(byte_buffer))
                    byte_buffer = []
                
                # Add regular token
                result.append(token)
        
        # Flush remaining bytes at end of token list
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