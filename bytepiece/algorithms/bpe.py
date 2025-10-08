from collections import Counter, defaultdict
from typing import Dict, List, Optional, Set, Tuple

from bytepiece.core.normalizer import Normalizer
from bytepiece.core.vocab import MergeRules, Vocabulary


def train_bpe(
    texts: List[str],
    vocab_size: int,
    normalizer: Optional[Normalizer] = None,
    byte_fallback: bool = True,
    use_special_tokens: bool = False,
    verbose: bool = False,
) -> Tuple[Vocabulary, MergeRules, Normalizer]:
    """Train a BPE tokenizer on a corpus.
    
    Args:
        texts: List of text strings for training
        vocab_size: Target vocabulary size
        normalizer: Text normalizer (creates default if None)
        byte_fallback: Enable byte-fallback for full coverage
        use_special_tokens: Add special tokens (PAD, UNK, BOS, EOS)
        verbose: Print training progress
        
    Returns:
        Tuple of (vocabulary, merge_rules, normalizer)
    """
    if normalizer is None:
        normalizer = Normalizer()
    
    vocab = Vocabulary(byte_fallback=byte_fallback, use_special_tokens=use_special_tokens)
    merge_rules = MergeRules()
    
 
    word_freqs: Dict[Tuple[str, ...], int] = Counter()
    
    for text in texts:
      
        chunks = normalizer.pre_tokenize(text)
        
       
        for chunk in chunks:
         
            normalized_chunk = normalizer.normalize(chunk)
            
            
            tokens = tuple(vocab.encode_with_fallback(normalized_chunk))
            

            word_freqs[tokens] += 1
    

    all_chars = set()
    for word_tuple in word_freqs:
        all_chars.update(word_tuple)
    vocab.add_tokens(sorted(all_chars))
    
    if verbose:
        print(f"Initial vocab size: {len(vocab)}")
        print(f"Unique token sequences (chunks): {len(word_freqs)}")
    
   
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
        
        if verbose and merge_idx % 100 == 0:
            print(f"Merge {merge_idx}/{num_merges}: {best_pair} (freq={best_freq})")
        
        
        merge_rules.add_merge(best_pair)
        
       
        new_token = best_pair[0] + best_pair[1]
        vocab.add_token(new_token)
        
        
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
    
    
    def __init__(
        self,
        vocab: Vocabulary,
        merge_rules: MergeRules,
        normalizer: Normalizer,
    ):
        
        self.vocab = vocab
        self.merge_rules = merge_rules
        self.normalizer = normalizer
    
    def encode(self, text: str) -> List[str]:
        
        chunks = self.normalizer.pre_tokenize(text)
        
        all_tokens = []
        
       
        for chunk in chunks:
            
            normalized = self.normalizer.normalize(chunk)
            
            
            tokens = list(self.vocab.encode_with_fallback(normalized))
            
            
            tokens = self._apply_merges(tokens)
            
            all_tokens.extend(tokens)
        
        return all_tokens
    
    def _apply_merges(self, tokens: List[str]) -> List[str]:

        if len(tokens) < 2:
            return tokens
        
    
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
    
    def decode(self, tokens: List[str]) -> str:

        if self.vocab.use_special_tokens:
            from bytepiece.core.vocab import SpecialTokens
            special_token_set = set(SpecialTokens.get_all())
            tokens = [t for t in tokens if t not in special_token_set]

        expanded_tokens = []
        for token in tokens:
        
            if '<0x' in token:
            
                i = 0
                while i < len(token):
                    if i + 6 <= len(token) and token[i:i+3] == '<0x' and token[i+5:i+6] == '>':
                    
                        expanded_tokens.append(token[i:i+6])
                        i += 6
                    else:
                    
                        expanded_tokens.append(token[i])
                        i += 1
            else:
                
                expanded_tokens.append(token)
        
        
        text = self.vocab.decode_bytes(expanded_tokens)
        
    
        text = self.normalizer.denormalize(text)
        
        return text
    
    def encode_batch(self, texts: List[str]) -> List[List[str]]:

        return [self.encode(text) for text in texts]
    
    def decode_batch(self, token_lists: List[List[str]]) -> List[str]:

        return [self.decode(tokens) for tokens in token_lists]