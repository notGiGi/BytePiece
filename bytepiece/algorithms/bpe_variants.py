"""
FIXED BPE Variants with correct pre-tokenization

Replace the content of bytepiece/algorithms/bpe_variants.py with this
"""

from typing import List, Dict
from collections import Counter
import time
import re

from bytepiece.algorithms.entropy import EntropyAnalyzer


# ============================================================================
# SIMPLE PRETOKENIZER (NO syntax awareness)
# ============================================================================

def simple_whitespace_tokenize(text: str) -> List[str]:
    """
    Simple whitespace tokenizer - NO operator/keyword awareness.
    Used for Entropy-Only to avoid syntax bias.
    """
    tokens = []
    for word in text.split():
        # Just split on whitespace, keep everything else together
        tokens.append(word)
    return tokens


# ============================================================================
# SYNTAX-AWARE PRETOKENIZER
# ============================================================================

def syntax_aware_tokenize(text: str) -> List[str]:
    """
    Tokenizer that ONLY separates based on syntax (operators, keywords).
    Used for Syntax-Only variant.
    """
    # Pattern that separates operators and punctuation
    pattern = re.compile(
        r'<<=|>>=|&=|\|=|\^=|'  # Bitwise assignment (RARE) - check FIRST (longer)
        r'>=|<=|==|!=|\+=|-=|->|=>|//=|//|<<|>>|\*\*|\|\||&&|'  # Other operators
        r'[a-zA-Z_][a-zA-Z0-9_]*|'  # Identifiers
        r'\d+\.?\d*|'  # Numbers
        r'"[^"]*"|'  # Strings
        r"'[^']*'|"  # Strings
        r'[^\w\s]|'  # Single punctuation
        r'\s+'  # Whitespace
    )
    
    tokens = pattern.findall(text)
    return [t for t in tokens if t.strip()]  # Remove pure whitespace


# ============================================================================
# VARIANT 1: Entropy-Only BPE
# ============================================================================

class EntropyOnlyBPE:
    """BPE with ONLY Shannon entropy (NO syntax rules)"""
    
    def __init__(self, vocab_size: int = 5000, min_frequency: int = 2):
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.analyzer = EntropyAnalyzer()
        
        self.merges = {}
        self.vocab = set()
        self.stats = {}
    
    def train(self, corpus: List[str], verbose: bool = False) -> Dict:
        """Train with entropy-only fragmentation"""
        start_time = time.time()
        
        if verbose:
            print("Training Entropy-Only BPE...")
        
        # Use SIMPLE whitespace tokenization (no syntax awareness)
        all_words = []
        for text in corpus:
            words = simple_whitespace_tokenize(text)
            all_words.extend(words)
        
        # Decide which words to fragment based ONLY on entropy
        preserved = []
        fragmentable = []
        
        for word in all_words:
            entropy = self.analyzer.shannon_entropy(word)
            
            # ONLY entropy threshold (no syntax rules)
            if entropy > 4.0:
                fragmentable.append(word)
            else:
                preserved.append(word)
        
        # Add preserved to vocab
        for word in set(preserved):
            self.vocab.add(word)
        
        # BPE on fragmentable words
        words_dict = {}
        for word in fragmentable:
            word_tuple = tuple(word)
            words_dict[word_tuple] = words_dict.get(word_tuple, 0) + 1
            for char in word:
                self.vocab.add(char)
        
        # Learn merges
        num_merges = max(0, self.vocab_size - len(self.vocab))
        merge_count = 0
        
        for i in range(num_merges):
            pairs = Counter()
            for word, freq in words_dict.items():
                if len(word) < 2:
                    continue
                for j in range(len(word) - 1):
                    pairs[(word[j], word[j+1])] += freq
            
            if not pairs:
                break
            
            best_pair, pair_freq = pairs.most_common(1)[0]
            
            if pair_freq < self.min_frequency:
                break
            
            self.merges[best_pair] = merge_count
            merge_count += 1
            self.vocab.add(best_pair[0] + best_pair[1])
            
            # Update words
            new_words = {}
            for word, word_freq in words_dict.items():
                new_word = self._merge_pair(list(word), best_pair)
                new_words[tuple(new_word)] = word_freq
            words_dict = new_words
        
        self.stats = {
            'training_time': time.time() - start_time,
            'vocab_size': len(self.vocab),
            'merges': merge_count,
        }
        
        return self.stats
    
    def _merge_pair(self, word: List[str], pair: tuple) -> List[str]:
        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and (word[i], word[i+1]) == pair:
                new_word.append(word[i] + word[i+1])
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        return new_word
    
    def encode(self, text: str) -> List[str]:
        """Encode with entropy-only logic"""
        words = simple_whitespace_tokenize(text)
        tokens = []
        
        for word in words:
            entropy = self.analyzer.shannon_entropy(word)
            
            # If low entropy, keep as is
            if entropy <= 4.0:
                tokens.append(word)
            else:
                # Fragment with BPE
                word_chars = list(word)
                for pair in sorted(self.merges.keys(), key=lambda p: self.merges[p]):
                    word_chars = self._merge_pair(word_chars, pair)
                tokens.extend(word_chars)
        
        return tokens


# ============================================================================
# VARIANT 2: Syntax-Only BPE
# ============================================================================

class SyntaxOnlyBPE:
    """BPE with ONLY syntax rules (NO entropy checks)"""
    
    def __init__(self, vocab_size: int = 5000, min_frequency: int = 2):
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.analyzer = EntropyAnalyzer()
        
        self.merges = {}
        self.vocab = set()
        self.stats = {}
    
    def train(self, corpus: List[str], verbose: bool = False) -> Dict:
        """Train with syntax-only preservation"""
        start_time = time.time()
        
        if verbose:
            print("Training Syntax-Only BPE...")
        
        # Use syntax-aware tokenization
        all_tokens = []
        for text in corpus:
            tokens = syntax_aware_tokenize(text)
            all_tokens.extend(tokens)
        
        # Preserve ONLY based on syntax type (no entropy)
        preserved = []
        fragmentable = []
        
        for token in all_tokens:
            construct_type = self.analyzer.detect_construct_type(token)
            
            # ONLY syntax rules (operators and keywords)
            if construct_type in ['operator', 'keyword']:
                preserved.append(token)
            else:
                fragmentable.append(token)
        
        # Add preserved to vocab
        for token in set(preserved):
            self.vocab.add(token)
        
        # BPE on fragmentable
        words_dict = {}
        for token in fragmentable:
            word_tuple = tuple(token)
            words_dict[word_tuple] = words_dict.get(word_tuple, 0) + 1
            for char in token:
                self.vocab.add(char)
        
        # Learn merges
        num_merges = max(0, self.vocab_size - len(self.vocab))
        merge_count = 0
        
        for i in range(num_merges):
            pairs = Counter()
            for word, freq in words_dict.items():
                if len(word) < 2:
                    continue
                for j in range(len(word) - 1):
                    pairs[(word[j], word[j+1])] += freq
            
            if not pairs:
                break
            
            best_pair, pair_freq = pairs.most_common(1)[0]
            
            if pair_freq < self.min_frequency:
                break
            
            self.merges[best_pair] = merge_count
            merge_count += 1
            self.vocab.add(best_pair[0] + best_pair[1])
            
            new_words = {}
            for word, word_freq in words_dict.items():
                new_word = self._merge_pair(list(word), best_pair)
                new_words[tuple(new_word)] = word_freq
            words_dict = new_words
        
        self.stats = {
            'training_time': time.time() - start_time,
            'vocab_size': len(self.vocab),
            'merges': merge_count,
        }
        
        return self.stats
    
    def _merge_pair(self, word: List[str], pair: tuple) -> List[str]:
        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and (word[i], word[i+1]) == pair:
                new_word.append(word[i] + word[i+1])
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        return new_word
    
    def encode(self, text: str) -> List[str]:
        """Encode with syntax-only logic"""
        tokens = syntax_aware_tokenize(text)
        result = []
        
        for token in tokens:
            construct_type = self.analyzer.detect_construct_type(token)
            
            # If operator/keyword, preserve
            if construct_type in ['operator', 'keyword']:
                result.append(token)
            else:
                # Fragment with BPE
                chars = list(token)
                for pair in sorted(self.merges.keys(), key=lambda p: self.merges[p]):
                    chars = self._merge_pair(chars, pair)
                result.extend(chars)
        
        return result