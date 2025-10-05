from typing import Dict, List, Optional, Set, Tuple


class Vocabulary:

    
    def __init__(self, byte_fallback: bool = True):
        self.byte_fallback = byte_fallback
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        
        if byte_fallback:
            self._init_byte_tokens()
    
    def _init_byte_tokens(self) -> None:
        for i in range(256): 
            token = f"<0x{i:02X}>" 
            self.token_to_id[token] = i  
            self.id_to_token[i] = token 
    
    def add_token(self, token: str) -> int:
        if token in self.token_to_id:
            return self.token_to_id[token]
    
        token_id = len(self.token_to_id)
        
        self.token_to_id[token] = token_id  
        self.id_to_token[token_id] = token  
        
        return token_id
    
    def add_tokens(self, tokens: List[str]) -> List[int]:
        return [self.add_token(token) for token in tokens]
    
    def get_id(self, token: str) -> Optional[int]:
        return self.token_to_id.get(token)
    
    def get_token(self, token_id: int) -> Optional[str]:
        return self.id_to_token.get(token_id)
    
    def encode_with_fallback(self, text: str) -> List[str]:
        """        
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
        """
        if not self.byte_fallback:
            return list(text)
        
        tokens = []
        for char in text:
            if char in self.token_to_id:
                tokens.append(char)
            else:
                char_bytes = char.encode('utf-8')
                for byte_val in char_bytes:
                    byte_token = f"<0x{byte_val:02X}>"
                    tokens.append(byte_token)
        
        return tokens
    
    def decode_bytes(self, tokens: List[str]) -> str:
        """Decode tokens back to text, handling byte tokens.

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
        
        """
        result = []
        byte_buffer = []  # Accumulates consecutive byte tokens
        
        for token in tokens:
            # Check if this is a byte token (format: <0xXX>)
            if self._is_byte_token(token):
                # Extract byte value: "<0xE4>" → 0xE4 → 228
                byte_val = int(token[3:5], 16)  
                byte_buffer.append(byte_val)
            else:
                # Not a byte token, flush any pending bytes first
                if byte_buffer:
                    try:
                        decoded = bytes(byte_buffer).decode('utf-8')
                        result.append(decoded)
                    except UnicodeDecodeError:
                    
                        result.append('�' * len(byte_buffer))
                    byte_buffer = []
                
                # Add regular token
                result.append(token)
        if byte_buffer:
            try:
                decoded = bytes(byte_buffer).decode('utf-8')
                result.append(decoded)
            except UnicodeDecodeError:
                result.append('�' * len(byte_buffer))
        
        return ''.join(result)
    
    def _is_byte_token(self, token: str) -> bool:
        return (
            len(token) == 6 and
            token.startswith('<0x') and
            token.endswith('>')
        )
    
    def __len__(self) -> int:
        return len(self.token_to_id)
    
    def __contains__(self, token: str) -> bool:
        return token in self.token_to_id
    
    def to_dict(self) -> dict:
        return {
            "byte_fallback": self.byte_fallback,
            "tokens": list(self.id_to_token.values()),
        }
    
    @classmethod
    def from_dict(cls, config: dict) -> "Vocabulary":
        vocab = cls(byte_fallback=config["byte_fallback"])
        start_idx = 256 if vocab.byte_fallback else 0
        for token in config["tokens"][start_idx:]:
            vocab.add_token(token)
        return vocab


class MergeRules:
    """Manages BPE merge rules with deterministic ordering."""
    
    def __init__(self):
        self.merges: List[Tuple[str, str]] = []
        self.merge_ranks: Dict[Tuple[str, str], int] = {}
    
    def add_merge(self, pair: Tuple[str, str], rank: Optional[int] = None) -> None:
        if pair in self.merge_ranks:
            return  
        
        if rank is None:
            rank = len(self.merges)
        
        self.merges.append(pair)
        self.merge_ranks[pair] = rank
    
    def get_rank(self, pair: Tuple[str, str]) -> Optional[int]:
        return self.merge_ranks.get(pair)
    
    def __len__(self) -> int:

        return len(self.merges)
    
    def to_dict(self) -> dict:
        return {
            "merges": [[left, right] for left, right in self.merges],
        }
    
    @classmethod
    def from_dict(cls, config: dict) -> "MergeRules":
        rules = cls()
        for rank, (left, right) in enumerate(config["merges"]):
            rules.add_merge((left, right), rank)
        return rules