from collections import Counter, defaultdict
from typing import Dict, List, Optional, Set, Tuple

from bytepiece.core.normalizer import Normalizer
from bytepiece.core.vocab import MergeRules, Vocabulary


def train_bpe(
    texts: List[str],
    vocab_size: int,
    normalizer: Optional[Normalizer] = None,
    byte_fallback: bool = True,
    verbose: bool = False,
) -> Tuple[Vocabulary, MergeRules, Normalizer]:
    """Train a BPE tokenizer on a corpus.
    
    Args:
        texts: List of text strings for training
        vocab_size: Target vocabulary size
        normalizer: Text normalizer (creates default if None)
        byte_fallback: Enable byte-fallback for full coverage
        verbose: Print training progress
        
    Returns:
        Tuple of (vocabulary, merge_rules, normalizer)
    """
    if normalizer is None:
        normalizer = Normalizer()
    
    vocab = Vocabulary(byte_fallback=byte_fallback)
    merge_rules = MergeRules()
    
    normalized_texts = [normalizer.normalize(text) for text in texts]

    word_freqs: Dict[Tuple[str, ...], int] = Counter()
    for text in normalized_texts:
        tokens = tuple(vocab.encode_with_fallback(text))
        word_freqs[tokens] += 1
    

    all_chars = set()
    for word_tuple in word_freqs:
        all_chars.update(word_tuple)
    vocab.add_tokens(sorted(all_chars))
    
    if verbose:
        print(f"Initial vocab size: {len(vocab)}")
        print(f"Unique words: {len(word_freqs)}")
    

    base_size = len(vocab)
    num_merges = vocab_size - base_size
    
    if num_merges <= 0:
        if verbose:
            print(f"Target vocab size {vocab_size} already reached with base tokens")
        return vocab, merge_rules, normalizer
    
    for merge_idx in range(num_merges):
        
        pair_freqs: Counter[Tuple[str, str]] = Counter()
        
        for word_tuple, freq in word_freqs.items():
            for i in range(len(word_tuple) - 1):
                pair = (word_tuple[i], word_tuple[i + 1])
                pair_freqs[pair] += freq
        
        if not pair_freqs:
            if verbose:
                print(f"No more pairs to merge at iteration {merge_idx}")
            break
        
        best_pair, best_freq = pair_freqs.most_common(1)[0]
        #We do not have a tie breaker, and for now, we dont need one
        if verbose and merge_idx % 100 == 0:
            print(f"Merge {merge_idx}/{num_merges}: {best_pair} (freq={best_freq})")
        
        merge_rules.add_merge(best_pair)
        
        # Create new token from the most common pair
        new_token = best_pair[0] + best_pair[1]
        vocab.add_token(new_token)
        
        # Update all words that contain this pair
        new_word_freqs: Dict[Tuple[str, ...], int] = {}
        for word_tuple, freq in word_freqs.items():
            new_word = _apply_merge(word_tuple, best_pair)
            new_word_freqs[new_word] = new_word_freqs.get(new_word, 0) + freq
        
        word_freqs = new_word_freqs
    
    if verbose:
        print(f"Final vocab size: {len(vocab)}")
        print(f"Total merges: {len(merge_rules)}")
    
    return vocab, merge_rules, normalizer


def _apply_merge(
    word: Tuple[str, ...],
    pair: Tuple[str, str],
) -> Tuple[str, ...]:
    """Apply a single merge operation to a word.
    
    Args:
        word: Word as tuple of tokens
        pair: Pair to merge (left, right)
        
    Returns:
        New word tuple with pair merged
    """
    if len(word) < 2:
        return word
    
    new_word = []
    i = 0
    while i < len(word):
    
        if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
            new_word.append(word[i] + word[i + 1])
            i += 2
        else:
            new_word.append(word[i])
            i += 1
    return tuple(new_word)


class BPEEncoder:
    """Encodes text using trained BPE model."""
    
    def __init__(
        self,
        vocab: Vocabulary,
        merge_rules: MergeRules,
        normalizer: Normalizer,
    ):
        """Initialize BPE encoder.
        
        Args:
            vocab: Trained vocabulary
            merge_rules: Trained merge rules
            normalizer: Text normalizer
        """
        self.vocab = vocab
        self.merge_rules = merge_rules
        self.normalizer = normalizer
    
    def encode(self, text: str) -> List[str]:
        """Encode text to tokens using BPE.
        
        Args:
            text: Input text
            
        Returns:
            List of token strings
        """
    
        normalized = self.normalizer.normalize(text)
        
   
        tokens = self.vocab.encode_with_fallback(normalized)
        
    
        while True:
            best_merge = None
            best_rank = float('inf')
            best_pos = -1
            
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                rank = self.merge_rules.get_rank(pair)
                if rank is not None and rank < best_rank:
                    best_merge = pair
                    best_rank = rank
                    best_pos = i
            if best_merge is None:
                break  
            tokens = (
                tokens[:best_pos] +
                [tokens[best_pos] + tokens[best_pos + 1]] +
                tokens[best_pos + 2:]
            )
        
        return tokens
    
    def encode_batch(self, texts: List[str]) -> List[List[str]]:
        """Encode multiple texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of token lists
        """
        return [self.encode(text) for text in texts]
    
    def decode(self, tokens: List[str]) -> str:
        """Decode tokens back to text.
        
        Args:
            tokens: List of token strings
            
        Returns:
            Decoded text
        """
        # Expand all tokens into individual byte tokens or characters
        expanded_tokens = []
        for token in tokens:
            # Check if this token contains byte tokens
            if '<0x' in token:
                # Extract all byte tokens from concatenated token
                i = 0
                while i < len(token):
                    if i + 6 <= len(token) and token[i:i+3] == '<0x' and token[i+5:i+6] == '>':
                        # Found a byte token
                        expanded_tokens.append(token[i:i+6])
                        i += 6
                    else:
                        # Regular character
                        expanded_tokens.append(token[i])
                        i += 1
            else:
                # Regular token without byte tokens
                expanded_tokens.append(token)
        
        # Now decode byte tokens
        text = self.vocab.decode_bytes(expanded_tokens)
        
        # Denormalize (remove spacers, etc.)
        text = self.normalizer.denormalize(text)
        
        return text
    
    def decode_batch(self, token_lists: List[List[str]]) -> List[str]:
        """Decode multiple token sequences.
        
        Args:
            token_lists: List of token lists
            
        Returns:
            List of decoded texts
        """
        return [self.decode(tokens) for tokens in token_lists]