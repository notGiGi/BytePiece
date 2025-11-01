from __future__ import annotations

from collections import Counter
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

from bytepiece.core.normalizer import Normalizer
from bytepiece.core.vocab import MergeRules, SpecialTokens, Vocabulary


def _resolve_corpus(
    texts: Optional[Union[str, Sequence[str]]],
    corpus_path: Optional[Union[str, Path]],
) -> List[str]:
    if isinstance(texts, str):
        corpus_path = corpus_path or texts
        texts = None

    if texts is not None:
        return list(texts)

    if corpus_path is None:
        raise ValueError("Either texts or corpus_path must be provided.")

    path = Path(corpus_path)
    if not path.exists():
        raise FileNotFoundError(f"Corpus path not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        return [line.rstrip("\n") for line in handle if line.strip()]


def train_bpe(
    texts: Optional[Union[str, Sequence[str]]] = None,
    vocab_size: Optional[int] = None,
    *,
    corpus_path: Optional[Union[str, Path]] = None,
    normalizer: Optional[Normalizer] = None,
    byte_fallback: bool = True,
    use_special_tokens: bool = False,
    special_tokens: Optional[SpecialTokens] = None,
    seed: Optional[int] = None,  # Accepted for API compatibility (unused)
    max_token_length: Optional[int] = 16,
    verbose: bool = False,
    return_encoder: bool = False,
) -> Union[Tuple[Vocabulary, MergeRules, Normalizer], "BPEEncoder"]:
    """
    Train a Byte Pair Encoding tokenizer on the provided corpus.
    """

    if vocab_size is None:
        raise ValueError("vocab_size must be provided.")

    if special_tokens is not None:
        use_special_tokens = True

    if normalizer is None:
        normalizer = Normalizer()

    corpus = _resolve_corpus(texts, corpus_path)

    vocab = Vocabulary(
        byte_fallback=byte_fallback,
        use_special_tokens=use_special_tokens,
        special_tokens=special_tokens,
        init_byte_tokens=False,
    )
    merge_rules = MergeRules()

    word_freqs: Dict[Tuple[str, ...], int] = Counter()
    base_tokens: set[str] = set()

    for line in corpus:
        for chunk in normalizer.pre_tokenize(line):
            normalized = normalizer.normalize(chunk)
            if not normalized:
                continue
            token_sequence = tuple(normalized)
            word_freqs[token_sequence] += 1
            base_tokens.update(token_sequence)

    if base_tokens:
        vocab.add_tokens(sorted(base_tokens))

    if verbose:
        print(f"Initial vocab size: {len(vocab)}")
        print(f"Unique sequences: {len(word_freqs)}")

    spacer_char = getattr(normalizer, "SPACER", None)

    target_merges = max(vocab_size - len(vocab), 0)

    for merge_idx in range(target_merges):
        pair_freqs: Counter[Tuple[str, str]] = Counter()

        for word_tuple, freq in word_freqs.items():
            for i in range(len(word_tuple) - 1):
                pair_freqs[(word_tuple[i], word_tuple[i + 1])] += freq

        if not pair_freqs:
            if verbose:
                print(f"No more pairs to merge at iteration {merge_idx}")
            break

        selected_pair: Optional[Tuple[str, str]] = None
        selected_freq = 0

        for candidate, freq in pair_freqs.most_common():
            combined = candidate[0] + candidate[1]
            if max_token_length is not None and len(combined) > max_token_length:
                continue
            if spacer_char and spacer_char in combined[1:]:
                continue
            selected_pair = candidate
            selected_freq = freq
            break

        if selected_pair is None:
            if verbose:
                print(f"No valid pairs within constraints at iteration {merge_idx}")
            break

        merge_rules.add_merge(selected_pair)
        vocab.add_token(selected_pair[0] + selected_pair[1])

        updated: Dict[Tuple[str, ...], int] = {}
        for word_tuple, freq in word_freqs.items():
            new_tuple = _apply_merge(word_tuple, selected_pair)
            updated[new_tuple] = updated.get(new_tuple, 0) + freq

        word_freqs = updated

        if verbose and merge_idx % 100 == 0:
            print(f"Merge {merge_idx}: {selected_pair} (freq={selected_freq})")

    target_size = vocab_size + (len(SpecialTokens.get_all(special_tokens)) if use_special_tokens else 0)
    while len(vocab.tokens) < target_size:
        vocab.add_token(f"<unused_{len(vocab.tokens)}>")

    if return_encoder:
        return BPEEncoder(vocab, merge_rules, normalizer)

    return vocab, merge_rules, normalizer


def _apply_merge(word: Tuple[str, ...], pair: Tuple[str, str]) -> Tuple[str, ...]:
    if len(word) < 2:
        return word

    merged_word: List[str] = []
    i = 0
    while i < len(word):
        if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
            merged_word.append(word[i] + word[i + 1])
            i += 2
        else:
            merged_word.append(word[i])
            i += 1
    return tuple(merged_word)


class BPEEncoder:
    """Encoder/decoder built from trained vocabulary and merge rules."""

    def __init__(
        self,
        vocab: Vocabulary,
        merge_rules: MergeRules,
        normalizer: Normalizer,
    ) -> None:
        self.vocab = vocab
        self.merge_rules = merge_rules
        self.normalizer = normalizer
        self._special_tokens = (
            SpecialTokens.get_all(self.vocab.special_tokens)
            if self.vocab.use_special_tokens
            else []
        )
        self._special_pattern = (
            re.compile("(" + "|".join(re.escape(tok) for tok in self._special_tokens) + ")")
            if self._special_tokens
            else None
        )

    def encode(self, text: str) -> List[str]:
        all_tokens: List[str] = []

        for chunk in self.normalizer.pre_tokenize(text):
            segments = self._split_on_special_tokens(chunk)
            normalized = self.normalizer.normalize(chunk)
            if not normalized:
                continue
            if len(segments) == 1 and segments[0] == chunk:
                tokens = list(self.vocab.encode_with_fallback(normalized))
                tokens = self._apply_merges(tokens)
                all_tokens.extend(tokens)
                continue

            for segment in segments:
                if segment in self._special_tokens:
                    all_tokens.append(segment)
                    continue
                normalized_segment = self.normalizer.normalize(segment)
                if not normalized_segment:
                    continue
                tokens = list(self.vocab.encode_with_fallback(normalized_segment))
                tokens = self._apply_merges(tokens)
                all_tokens.extend(tokens)

        return all_tokens

    def _apply_merges(self, tokens: List[str]) -> List[str]:
        if len(tokens) < 2:
            return tokens

        while True:
            best_rank = float("inf")
            best_pos = -1

            for idx in range(len(tokens) - 1):
                pair = (tokens[idx], tokens[idx + 1])
                rank = self.merge_rules.get_rank(pair)
                if rank is not None and rank < best_rank:
                    best_rank = rank
                    best_pos = idx

            if best_pos == -1:
                break

            tokens = (
                tokens[:best_pos]
                + [tokens[best_pos] + tokens[best_pos + 1]]
                + tokens[best_pos + 2 :]
            )

        return tokens

    def decode(self, tokens: List[str]) -> str:
        special_set = (
            set(SpecialTokens.get_all(self.vocab.special_tokens))
            if self.vocab.use_special_tokens
            else set()
        )

        filtered = [token for token in tokens if token not in special_set]
        text = self.vocab.decode_bytes(filtered)
        return self.normalizer.denormalize(text)

    def encode_batch(self, texts: Iterable[str]) -> List[List[str]]:
        return [self.encode(text) for text in texts]

    def decode_batch(self, token_lists: Iterable[List[str]]) -> List[str]:
        return [self.decode(tokens) for tokens in token_lists]

    def _split_on_special_tokens(self, text: str) -> List[str]:
        if not self._special_pattern:
            return [text]

        parts: List[str] = []
        last = 0
        for match in self._special_pattern.finditer(text):
            start, end = match.span()
            if start > last:
                parts.append(text[last:start])
            parts.append(match.group(0))
            last = end
        if last < len(text):
            parts.append(text[last:])
        return [part for part in parts if part]
