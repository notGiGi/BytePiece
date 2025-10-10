"""
BPE Variants for Ablation Studies
Path: bytepiece/algorithms/bpe_variants.py

Three variants:
1. EntropyOnlyBPE - Only entropy, no syntax rules
2. SyntaxOnlyBPE - Only syntax rules, no entropy
3. (Hybrid is in entropy_bpe.py)
"""

import re
from collections import Counter
from typing import List, Dict
import time

from bytepiece.algorithms.entropy import EntropyAnalyzer


def simple_whitespace_tokenize(text: str) -> List[str]:
    """Simple whitespace tokenization for Entropy-Only variant"""
    return text.split()


def syntax_aware_tokenize(text: str) -> List[str]:
    """Syntax-aware tokenization for Syntax-Only variant"""
    pattern = re.compile(
        r'<<=|>>=|&=|\|=|\^=|'  # Bitwise assignment
        r'>=|<=|==|!=|\+=|-=|->|=>|//=|//|<<|>>|\*\*|\|\||&&|'  # Operators
        r'[a-zA-Z_][a-zA-Z0-9_]*|'  # Identifiers
        r'\d+\.?\d*|'  # Numbers
        r'"[^"]*"|'  # Strings
        r"'[^']*'|"  # Strings
        r'[^\w\s]|'  # Single punctuation
        r'\s+'  # Whitespace
    )
    
    tokens = pattern.findall(text)
    return [t for t in tokens if t.strip()]


class EntropyOnlyBPE:
    """BPE with ONLY Shannon entropy (NO syntax rules) - WITH VOCAB CONTROL"""
    
    def __init__(self, vocab_size: int = 5000, min_frequency: int = 2):
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.analyzer = EntropyAnalyzer()
        
        self.merges = {}
        self.vocab = set()
        self.stats = {}
    
    def train(self, corpus: List[str], verbose: bool = False) -> Dict:
        """Train with entropy-only fragmentation AND vocabulary control"""
        start_time = time.time()
        
        if verbose:
            print("Training Entropy-Only BPE...")
        
        # Count ALL tokens first
        token_counts = Counter()
        for text in corpus:
            words = simple_whitespace_tokenize(text)
            token_counts.update(words)
        
        # Sort by frequency and entropy
        preserved_candidates = []
        fragmentable = []
        
        for word, freq in token_counts.items():
            if not word:
                continue
                
            entropy = self.analyzer.shannon_entropy(word)
            
            # Score combines frequency and entropy
            # High frequency + low entropy = preserve
            score = freq / (1 + entropy)  # Higher score = more likely to preserve
            
            if entropy > 4.0:  # High entropy = fragment
                fragmentable.append(word)
            else:
                preserved_candidates.append((score, word))
        
        # CRITICAL: Limit preserved vocabulary
        max_preserve = min(
            len(preserved_candidates),
            int(self.vocab_size * 0.3)  # Max 30% for preserved tokens
        )
        
        # Sort by score and take top tokens
        preserved_candidates.sort(reverse=True)
        preserved = [word for score, word in preserved_candidates[:max_preserve]]
        
        # Add preserved to vocab
        for word in preserved:
            self.vocab.add(word)
        
        if verbose:
            print(f"  Preserved: {len(preserved)} tokens (limited to {max_preserve})")
        
        # Build character vocabulary from fragmentable tokens
        for word in fragmentable:
            for char in word:
                self.vocab.add(char)
        
        # BPE on fragmentable words
        words_dict = {}
        for word in fragmentable:
            word_tuple = tuple(word)
            words_dict[word_tuple] = token_counts.get(word, 1)
        
        # Learn BPE merges
        num_merges = max(0, self.vocab_size - len(self.vocab))
        merge_count = 0
        
        for i in range(num_merges):
            if not words_dict:
                break
                
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
            
            # Apply merge
            new_words = {}
            for word, word_freq in words_dict.items():
                new_word = self._merge_pair(list(word), best_pair)
                new_words[tuple(new_word)] = word_freq
            words_dict = new_words
        
        self.stats = {
            'training_time': time.time() - start_time,
            'vocab_size': len(self.vocab),
            'preserved': len(preserved),
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
            if word in self.vocab:
                tokens.append(word)
            else:
                # Apply BPE merges
                word_chars = list(word)
                for pair in sorted(self.merges.keys(), key=lambda p: self.merges[p]):
                    word_chars = self._merge_pair(word_chars, pair)
                tokens.extend(word_chars)
        
        return tokens


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
        
        # Count frequencies
        token_counts = Counter(all_tokens)
        
        # Preserve ONLY based on syntax type
        preserved = []
        fragmentable = []
        
        for token, freq in token_counts.items():
            construct_type = self.analyzer.detect_construct_type(token)
            
            # ONLY syntax rules (operators and keywords)
            if construct_type in ['operator', 'keyword']:
                preserved.append(token)
            else:
                fragmentable.append(token)
        
        # Add preserved to vocab
        for token in set(preserved):
            self.vocab.add(token)
        
        # Add characters from fragmentable
        for token in fragmentable:
            for char in token:
                self.vocab.add(char)
        
        # BPE on fragmentable
        words_dict = {}
        for token in fragmentable:
            word_tuple = tuple(token)
            words_dict[word_tuple] = words_dict.get(word_tuple, 0) + 1
        
        # Learn merges
        num_merges = max(0, self.vocab_size - len(self.vocab))
        merge_count = 0
        
        for i in range(num_merges):
            if not words_dict:
                break
                
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
            
            # Apply merge
            new_words = {}
            for word, word_freq in words_dict.items():
                new_word = self._merge_pair(list(word), best_pair)
                new_words[tuple(new_word)] = word_freq
            words_dict = new_words
        
        self.stats = {
            'training_time': time.time() - start_time,
            'vocab_size': len(self.vocab),
            'preserved': len([t for t in preserved if t in self.vocab]),
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
        raw_tokens = syntax_aware_tokenize(text)
        tokens = []
        
        for token in raw_tokens:
            if token in self.vocab:
                tokens.append(token)
            else:
                # Apply BPE merges
                word_chars = list(token)
                for pair in sorted(self.merges.keys(), key=lambda p: self.merges[p]):
                    word_chars = self._merge_pair(word_chars, pair)
                tokens.extend(word_chars)
        
        return tokens