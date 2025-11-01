from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class SpecialTokens:
    """Configuration for special token strings."""

    pad: str = "<PAD>"
    unk: str = "<UNK>"
    bos: str = "<BOS>"
    eos: str = "<EOS>"

    PAD = "<PAD>"
    UNK = "<UNK>"
    BOS = "<BOS>"
    EOS = "<EOS>"

    @classmethod
    def defaults(cls) -> "SpecialTokens":
        return cls()

    @classmethod
    def get_all(cls, config: Optional["SpecialTokens"] = None) -> List[str]:
        cfg = config or cls.defaults()
        return [cfg.pad, cfg.unk, cfg.bos, cfg.eos]


class Vocabulary:
    """Mutable vocabulary supporting byte fallback and special tokens."""

    def __init__(
        self,
        byte_fallback: bool = True,
        use_special_tokens: bool = False,
        special_tokens: Optional[SpecialTokens] = None,
        init_byte_tokens: bool = True,
    ):
        self.byte_fallback = byte_fallback
        self.use_special_tokens = use_special_tokens
        self.special_tokens = special_tokens or SpecialTokens.defaults()
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self._byte_tokens_added = False

        if use_special_tokens:
            self._init_special_tokens()

        if byte_fallback and init_byte_tokens:
            self._init_byte_tokens()

    def _init_special_tokens(self) -> None:
        for token in SpecialTokens.get_all(self.special_tokens):
            self.add_token(token)

    def _init_byte_tokens(self) -> None:
        if self._byte_tokens_added:
            return
        for i in range(256):
            token = f"<0x{i:02X}>"
            self.add_token(token)
        self._byte_tokens_added = True

    def enable_byte_fallback(self) -> None:
        self.byte_fallback = True
        self._byte_tokens_added = False
        self._init_byte_tokens()

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
        return self.special_tokens.pad if self.use_special_tokens else None

    @property
    def unk_token(self) -> Optional[str]:
        return self.special_tokens.unk if self.use_special_tokens else None

    @property
    def bos_token(self) -> Optional[str]:
        return self.special_tokens.bos if self.use_special_tokens else None

    @property
    def eos_token(self) -> Optional[str]:
        return self.special_tokens.eos if self.use_special_tokens else None

    @property
    def pad_token_id(self) -> Optional[int]:
        return self.get_id(self.pad_token) if self.pad_token else None

    @property
    def unk_token_id(self) -> Optional[int]:
        return self.get_id(self.unk_token) if self.unk_token else None

    @property
    def bos_token_id(self) -> Optional[int]:
        return self.get_id(self.bos_token) if self.bos_token else None

    @property
    def eos_token_id(self) -> Optional[int]:
        return self.get_id(self.eos_token) if self.eos_token else None

    def encode_with_fallback(self, text: str) -> List[str]:
        if not text:
            return []

        if not self.byte_fallback:
            return list(text)

        tokens: List[str] = []
        for char in text:
            if char in self.token_to_id:
                tokens.append(char)
            else:
                for byte_val in char.encode("utf-8"):
                    byte_token = f"<0x{byte_val:02X}>"
                    if byte_token not in self.token_to_id:
                        self.add_token(byte_token)
                    tokens.append(byte_token)
        return tokens

    def decode_bytes(self, tokens: List[str]) -> str:
        result: List[str] = []
        byte_buffer: List[int] = []

        special_set = set(SpecialTokens.get_all(self.special_tokens)) if self.use_special_tokens else set()

        for token in tokens:
            if token in special_set:
                if byte_buffer:
                    result.append(self._flush_bytes(byte_buffer))
                result.append(token)
                continue

            if self._is_byte_token(token):
                byte_buffer.append(int(token[3:5], 16))
            else:
                if byte_buffer:
                    result.append(self._flush_bytes(byte_buffer))
                result.append(token)

        if byte_buffer:
            result.append(self._flush_bytes(byte_buffer))

        return "".join(result)

    def _flush_bytes(self, byte_buffer: List[int]) -> str:
        try:
            return bytes(byte_buffer).decode("utf-8")
        except UnicodeDecodeError:
            return "?" * len(byte_buffer)
        finally:
            byte_buffer.clear()

    def _is_byte_token(self, token: str) -> bool:
        return len(token) == 6 and token.startswith("<0x") and token.endswith(">")

    def __len__(self) -> int:
        return len(self.token_to_id)

    def __contains__(self, token: str) -> bool:
        return token in self.token_to_id

    @property
    def tokens(self) -> List[str]:
        return [self.id_to_token[i] for i in range(len(self.id_to_token))]

    def to_dict(self) -> dict:
        data = {
            "byte_fallback": self.byte_fallback,
            "use_special_tokens": self.use_special_tokens,
            "tokens": [self.id_to_token[i] for i in range(len(self.id_to_token))],
        }
        if self.use_special_tokens:
            data["special_tokens"] = {
                "pad": self.special_tokens.pad,
                "unk": self.special_tokens.unk,
                "bos": self.special_tokens.bos,
                "eos": self.special_tokens.eos,
            }
        return data

    @classmethod
    def from_dict(cls, config: dict) -> "Vocabulary":
        special_tokens = None
        if config.get("use_special_tokens"):
            spec = config.get("special_tokens") or {}
            special_tokens = SpecialTokens(
                pad=spec.get("pad", SpecialTokens.PAD),
                unk=spec.get("unk", SpecialTokens.UNK),
                bos=spec.get("bos", SpecialTokens.BOS),
                eos=spec.get("eos", SpecialTokens.EOS),
            )

        vocab = cls(
            byte_fallback=config.get("byte_fallback", True),
            use_special_tokens=config.get("use_special_tokens", False),
            special_tokens=special_tokens,
            init_byte_tokens=False,
        )

        # Skip the tokens already initialized (special + byte tokens)
        start_idx = len(vocab.token_to_id)
        for token in config.get("tokens", [])[start_idx:]:
            vocab.add_token(token)

        return vocab


class MergeRules:
    """Stores merge operations with lookup by rank."""

    def __init__(self) -> None:
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
        for rank, (left, right) in enumerate(config.get("merges", [])):
            rules.add_merge((left, right), rank)
        return rules
