from typing import Dict, List, Optional, Set, Tuple


class SpecialTokens:
    
    PAD = "<PAD>"
    UNK = "<UNK>"
    BOS = "<BOS>"
    EOS = "<EOS>"
    
    @classmethod
    def get_all(cls) -> List[str]:
        return [cls.PAD, cls.UNK, cls.BOS, cls.EOS]


class Vocabulary:
    
    def __init__(self, byte_fallback: bool = True, use_special_tokens: bool = False):

        self.byte_fallback = byte_fallback
        self.use_special_tokens = use_special_tokens
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        
        if use_special_tokens:
            self._init_special_tokens()
        
        if byte_fallback:
            self._init_byte_tokens()
    
    def _init_special_tokens(self) -> None:
        for token in SpecialTokens.get_all():
            self.add_token(token)
    
    def _init_byte_tokens(self) -> None:

        for i in range(256):
            token = f"<0x{i:02X}>"
            self.add_token(token)
    
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
    
    @property
    def pad_token(self) -> Optional[str]:
        """Get PAD token if special tokens are enabled."""
        return SpecialTokens.PAD if self.use_special_tokens else None
    
    @property
    def unk_token(self) -> Optional[str]:
        """Get UNK token if special tokens are enabled."""
        return SpecialTokens.UNK if self.use_special_tokens else None
    
    @property
    def bos_token(self) -> Optional[str]:
        """Get BOS token if special tokens are enabled."""
        return SpecialTokens.BOS if self.use_special_tokens else None
    
    @property
    def eos_token(self) -> Optional[str]:
        """Get EOS token if special tokens are enabled."""
        return SpecialTokens.EOS if self.use_special_tokens else None
    
    @property
    def pad_token_id(self) -> Optional[int]:
        """Get PAD token ID."""
        return self.get_id(SpecialTokens.PAD) if self.use_special_tokens else None
    
    @property
    def unk_token_id(self) -> Optional[int]:
        """Get UNK token ID."""
        return self.get_id(SpecialTokens.UNK) if self.use_special_tokens else None
    
    @property
    def bos_token_id(self) -> Optional[int]:
        """Get BOS token ID."""
        return self.get_id(SpecialTokens.BOS) if self.use_special_tokens else None
    
    @property
    def eos_token_id(self) -> Optional[int]:
        """Get EOS token ID."""
        return self.get_id(SpecialTokens.EOS) if self.use_special_tokens else None
    
    def encode_with_fallback(self, text: str) -> List[str]:
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
        result = []
        byte_buffer = []
        
        for token in tokens:
        
            if self.use_special_tokens and token in SpecialTokens.get_all():
                if byte_buffer:
                    try:
                        decoded = bytes(byte_buffer).decode('utf-8')
                        result.append(decoded)
                    except UnicodeDecodeError:
                        result.append('�' * len(byte_buffer))
                    byte_buffer = []
              
                continue
            
            if self._is_byte_token(token):
                byte_val = int(token[3:5], 16)
                byte_buffer.append(byte_val)
            else:
                if byte_buffer:
                    try:
                        decoded = bytes(byte_buffer).decode('utf-8')
                        result.append(decoded)
                    except UnicodeDecodeError:
                        result.append('�' * len(byte_buffer))
                    byte_buffer = []
                
                result.append(token)
        
        if byte_buffer:
            try:
                decoded = bytes(byte_buffer).decode('utf-8')
                result.append(decoded)
            except UnicodeDecodeError:
                result.append('�' * len(byte_buffer))
        
        return ''.join(result)
    
    def _is_byte_token(self, token: str) -> bool:
        """Check if token is a byte token."""
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
        """Serialize vocabulary to dict."""
        return {
            "byte_fallback": self.byte_fallback,
            "use_special_tokens": self.use_special_tokens,
            "tokens": list(self.id_to_token.values()),
        }
    
    @classmethod
    def from_dict(cls, config: dict) -> "Vocabulary":
        """Create vocabulary from dict."""
        vocab = cls(
            byte_fallback=config["byte_fallback"],
            use_special_tokens=config.get("use_special_tokens", False)
        )
        

        start_idx = 0
        if vocab.use_special_tokens:
            start_idx += len(SpecialTokens.get_all())
        if vocab.byte_fallback:
            start_idx += 256
        
        # Add remaining tokens
        for token in config["tokens"][start_idx:]:
            vocab.add_token(token)
        
        return vocab


class MergeRules:

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