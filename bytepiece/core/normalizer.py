"""
Normalization and pre-tokenization helpers for BytePiece.
"""

from __future__ import annotations

import unicodedata
from enum import Enum
from typing import List, Optional, Union

import regex as re


class NormalizationMode(str, Enum):
    NONE = "none"
    NFC = "nfc"
    NFD = "nfd"
    NFKC = "nfkc"
    NFKD = "nfkd"


class SpacerMode(str, Enum):
    PREFIX = "prefix"
    SUFFIX = "suffix"
    ISOLATED = "isolated"
    SEPARATOR = "isolated"  # Backwards compatibility alias
    NONE = "none"


class PreTokenizationMode(str, Enum):
    NONE = "none"
    WHITESPACE = "whitespace"
    GPT2 = "gpt2"
    CODE = "code"
    PYTHON = "python"
    SYNTAX_AWARE = "syntax_aware"


class Normalizer:
    """Handles Unicode normalization, spacers and optional pre-tokenization."""

    SPACER = "\u2581"
    WHITESPACE_RE = re.compile(r"\s+")
    WORD_RE = re.compile(r"\S+")

    GPT2_PATTERN = re.compile(
        r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
        re.UNICODE,
    )

    def __init__(
        self,
        normalization: Optional[Union[NormalizationMode, str]] = None,
        spacer_mode: Optional[Union[SpacerMode, str]] = None,
        lowercase: bool = False,
        pre_tokenization: Optional[Union[PreTokenizationMode, str]] = None,
        *,
        normalization_mode: Optional[Union[NormalizationMode, str]] = None,
        pre_tokenization_mode: Optional[Union[PreTokenizationMode, str]] = None,
    ) -> None:
        norm_value = normalization_mode or normalization
        spacer_value = spacer_mode
        pretok_value = pre_tokenization_mode or pre_tokenization

        self.normalization_mode = self._coerce_normalization(norm_value)
        self.spacer_mode = self._coerce_spacer(spacer_value)
        self.pre_tokenization_mode = self._coerce_pretokenization(pretok_value)
        self.lowercase = lowercase

        self._code_tokenizer = None  # Lazy import

    def _coerce_normalization(
        self, value: Optional[Union[NormalizationMode, str]]
    ) -> NormalizationMode:
        if value is None:
            return NormalizationMode.NFKC
        if isinstance(value, NormalizationMode):
            return value
        return NormalizationMode(value.lower())

    def _coerce_spacer(
        self, value: Optional[Union[SpacerMode, str]]
    ) -> SpacerMode:
        if value is None:
            return SpacerMode.PREFIX
        if isinstance(value, SpacerMode):
            return value
        return SpacerMode(value.lower())

    def _coerce_pretokenization(
        self, value: Optional[Union[PreTokenizationMode, str]]
    ) -> PreTokenizationMode:
        if value is None:
            return PreTokenizationMode.NONE
        if isinstance(value, PreTokenizationMode):
            return value
        lookup = value.lower()
        if lookup == "whitespace":
            return PreTokenizationMode.WHITESPACE
        if lookup == "gpt2":
            return PreTokenizationMode.GPT2
        if lookup in {"code", "python"}:
            return PreTokenizationMode.CODE
        if lookup in {"syntax_aware", "syntax"}:
            return PreTokenizationMode.SYNTAX_AWARE
        return PreTokenizationMode.NONE

    def pre_tokenize(self, text: str) -> List[str]:
        mode = self.pre_tokenization_mode

        if mode == PreTokenizationMode.NONE:
            return [text] if text else []

        if mode == PreTokenizationMode.WHITESPACE:
            parts: List[str] = []
            index = 0
            for match in self.WHITESPACE_RE.finditer(text):
                start, end = match.span()
                if start > index:
                    parts.append(text[index:start])
                parts.append(text[start:end])
                index = end
            if index < len(text):
                parts.append(text[index:])
            return parts or [text]

        if mode == PreTokenizationMode.GPT2:
            matches = self.GPT2_PATTERN.findall(text)
            return matches if matches else [text]

        if mode in (PreTokenizationMode.CODE, PreTokenizationMode.PYTHON, PreTokenizationMode.SYNTAX_AWARE):
            if self._code_tokenizer is None:
                from bytepiece.core.code_pretokenizer import CodePreTokenizer

                self._code_tokenizer = CodePreTokenizer()
            return self._code_tokenizer.tokenize(text)

        return [text]

    def normalize(self, text: str) -> str:
        if not text:
            return ""

        normalized = text

        if self.normalization_mode != NormalizationMode.NONE:
            normalized = unicodedata.normalize(self.normalization_mode.value.upper(), normalized)

        if self.lowercase:
            normalized = normalized.lower()

        return self._apply_spacer(normalized)

    def _apply_spacer(self, text: str) -> str:
        mode = self.spacer_mode

        if mode == SpacerMode.NONE:
            return text

        if mode == SpacerMode.PREFIX:
            replaced = self.WHITESPACE_RE.sub(self.SPACER, text)
            return f"{self.SPACER}{replaced}"

        if mode == SpacerMode.SUFFIX:
            return self.WORD_RE.sub(lambda m: f"{m.group(0)}{self.SPACER}", text)

        if mode in (SpacerMode.ISOLATED, SpacerMode.SEPARATOR):
            return self.WHITESPACE_RE.sub(lambda m: f"{self.SPACER}", text)

        return text

    def denormalize(self, text: str) -> str:
        mode = self.spacer_mode

        if mode == SpacerMode.PREFIX:
            result = text
            if result.startswith(self.SPACER):
                result = result[len(self.SPACER):]
            return result.replace(self.SPACER, " ")

        if mode == SpacerMode.SUFFIX:
            return text.replace(self.SPACER, " ").strip()

        if mode in (SpacerMode.ISOLATED, SpacerMode.SEPARATOR):
            return text.replace(self.SPACER, " ")

        return text

    def to_dict(self) -> dict:
        return {
            "normalization_mode": self.normalization_mode.name,
            "spacer_mode": self.spacer_mode.name,
            "pre_tokenization_mode": self.pre_tokenization_mode.name,
            "lowercase": self.lowercase,
            "spacer": self.SPACER,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Normalizer":
        return cls(
            normalization=data.get("normalization_mode", data.get("normalization", "NFKC")),
            spacer_mode=data.get("spacer_mode", "PREFIX"),
            lowercase=data.get("lowercase", False),
            pre_tokenization=data.get("pre_tokenization_mode", data.get("pre_tokenization", "NONE")),
        )

    @property
    def normalization(self) -> NormalizationMode:
        return self.normalization_mode

    @property
    def spacer(self) -> SpacerMode:
        return self.spacer_mode

    @property
    def pre_tokenization(self) -> PreTokenizationMode:
        return self.pre_tokenization_mode
