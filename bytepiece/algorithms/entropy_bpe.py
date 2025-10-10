"""
Entropy-Aware BPE Tokenizer
Path: bytepiece/algorithms/entropy_bpe.py

Integrates entropy analysis with standard BPE training
"""

from typing import List, Dict, Optional
from collections import Counter
import time

from bytepiece.algorithms.entropy import EntropyAnalyzer, EntropyPreTokenizer


class EntropyAwareBPE:
    """
    BPE tokenizer with entropy-aware pre-tokenization.
    
    Key differences from standard BPE:
    1. Preserves operators, keywords (syntax rules)
    2. Preserves low-entropy tokens (common patterns)
    3. Fragments high-entropy tokens (URLs, unique strings)
    """
    
    def __init__(self,
                 vocab_size: int = 5000,
                 entropy_thresholds: Optional[Dict[str, float]] = None,
                 min_frequency: int = 2):
        """
        Args:
            vocab_size: Target vocabulary size
            entropy_thresholds: Custom thresholds for different construct types
            min_frequency: Minimum frequency for a merge to be considered
        """
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        
        # Initialize entropy components
        thresholds = entropy_thresholds or {}
        self.analyzer = EntropyAnalyzer(
            default_threshold=thresholds.get('default', 4.0),
            string_threshold=thresholds.get('string', 4.5),
            identifier_threshold=thresholds.get('identifier', 3.0),
            comment_threshold=thresholds.get('comment', 3.5),
        )
        self.pretokenizer = EntropyPreTokenizer(self.analyzer)
        
        # Learned BPE merges and vocabulary
        self.merges = {}  # (pair) -> rank
        self.vocab = set()
        
        # Training statistics
        self.stats = {
            'preserved_tokens': 0,
            'fragmented_tokens': 0,
            'total_merges': 0,
            'training_time': 0.0,
        }
    
    def _get_pairs(self, word: List[str]) -> Counter:
        """Get all adjacent pairs in a word"""
        pairs = Counter()
        for i in range(len(word) - 1):
            pairs[(word[i], word[i+1])] += 1
        return pairs
    
    def _merge_pair(self, word: List[str], pair: tuple) -> List[str]:
        """Merge all occurrences of a pair in a word"""
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
    
    # Agrega estos métodos a tu clase EntropyAwareBPE existente en entropy_bpe.py

    def _control_vocabulary_size(self, preserved_tokens, fragmentable_tokens):
        """
        Control vocabulary explosion by limiting preserved tokens
        """
        # Calculate budget
        max_preserved = int(self.vocab_size * 0.15)  # Max 15% for preserved
        
        if len(preserved_tokens) > max_preserved:
            # Sort by importance: operators > keywords > frequent tokens
            scored_tokens = []
            
            for token in preserved_tokens:
                score = 0
                
                # Operators get highest priority
                if token.text in ['>=', '<=', '==', '!=', '//', '**', '+=', '-=']:
                    score = 1000
                # Keywords next
                elif token.text in ['def', 'class', 'if', 'else', 'return', 'import']:
                    score = 500
                # Frequency-based for others
                else:
                    score = token.frequency if hasattr(token, 'frequency') else 1
                
                scored_tokens.append((score, token))
            
            # Keep only top tokens
            scored_tokens.sort(key=lambda x: x[0], reverse=True)
            preserved_tokens = [t for _, t in scored_tokens[:max_preserved]]
        
        return preserved_tokens, fragmentable_tokens

    # En entropy_bpe.py, reemplaza el método train() con este:

    def train(self, corpus: List[str], verbose: bool = False) -> Dict:
        """Train entropy-aware BPE on a corpus."""
        start_time = time.time()
        
        # Pre-tokenize with entropy analysis
        all_preserved = []
        all_fragmentable = []
        
        for code in corpus:
            pre_tokens = self.pretokenizer.pretokenize(code)
            for pt in pre_tokens:
                if pt.should_fragment:
                    all_fragmentable.append(pt)
                else:
                    all_preserved.append(pt)
        
        # Control vocabulary size (si agregaste _control_vocabulary_size)
        if hasattr(self, '_control_vocabulary_size'):
            all_preserved, all_fragmentable = self._control_vocabulary_size(
                all_preserved, all_fragmentable
            )
        
        if verbose:
            print(f"  Pre-tokenization complete:")
            print(f"    Preserved: {len(all_preserved)}")
            print(f"    Fragmentable: {len(all_fragmentable)}")
        
        # Add preserved tokens to vocab
        for pt in all_preserved:
            self.vocab.add(pt.text)
        
        # Build character vocabulary from fragmentable
        for pt in all_fragmentable:
            for char in pt.text:
                self.vocab.add(char)
        
        # BPE on fragmentable tokens
        word_freqs = Counter()
        for pt in all_fragmentable:
            word = tuple(pt.text)
            word_freqs[word] += 1
        
        # Initialize working structures
        words = {word: list(word) for word in word_freqs}
        num_merges = 0
        
        # Learn BPE merges
        target_merges = max(0, self.vocab_size - len(self.vocab))
        
        for _ in range(target_merges):
            if not word_freqs:
                break
                
            pairs = Counter()
            for word, freq in word_freqs.items():
                word_list = words[word]
                for i in range(len(word_list) - 1):
                    pairs[(word_list[i], word_list[i+1])] += freq
            
            if not pairs:
                break
            
            # Find best pair
            best_pair, best_freq = pairs.most_common(1)[0]
            
            if best_freq < self.min_frequency:
                break
            
            # Apply merge
            for word in words:
                words[word] = self._merge_pair(words[word], best_pair)
            
            # Record merge
            self.merges[best_pair] = num_merges
            self.vocab.add(best_pair[0] + best_pair[1])
            num_merges += 1
        
        # CRITICAL FIX: Ensure vocab is never empty
        if len(self.vocab) == 0:
            # Emergency fallback: add all characters from corpus
            for text in corpus:
                for char in text:
                    if char.strip():  # Skip pure whitespace
                        self.vocab.add(char)
            
            # Add some basic tokens
            basic_tokens = ['def', 'return', 'if', 'for', 'class', '(', ')', ':', '=']
            for token in basic_tokens:
                self.vocab.add(token)
        
        # Store statistics
        self.stats = {
            'preserved_tokens': len(all_preserved),
            'fragmented_tokens': len(all_fragmentable),
            'total_merges': num_merges,
            'training_time': time.time() - start_time,
            'final_vocab_size': len(self.vocab)
        }
        
        if verbose:
            print(f"  Training complete:")
            print(f"    Final vocab size: {len(self.vocab)}")
            print(f"    Merges learned: {num_merges}")
        
        return self.stats
        

    
    def encode(self, text: str) -> List[str]:
        """
        Encode text using entropy-aware BPE.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        # Pre-tokenize with entropy analysis
        pre_tokens = self.pretokenizer.pretokenize(text)
        
        tokens = []
        
        for pt in pre_tokens:
            if not pt.should_fragment:
                # Preserved token - keep as is
                tokens.append(pt.text)
            else:
                # Fragmentable token - apply BPE merges
                word = list(pt.text)
                
                # Apply all learned merges in order
                for pair in sorted(self.merges.keys(), key=lambda p: self.merges[p]):
                    word = self._merge_pair(word, pair)
                
                # Add resulting tokens
                tokens.extend(word)
        
        return tokens
    
    def decode(self, tokens: List[str]) -> str:
        """
        Decode tokens back to text.
        
        Args:
            tokens: List of tokens
            
        Returns:
            Decoded text
        """
        # Simple concatenation (spaces should be handled by pre-tokenizer)
        return ''.join(tokens)
    
    def get_stats(self) -> Dict:
        """Get training statistics"""
        return {
            **self.stats,
            'vocab_size': len(self.vocab),
            'num_merges': len(self.merges),
        }
    
    def save(self, path: str):
        """
        Save model to file.
        
        Args:
            path: Output file path
        """
        import json
        
        model_data = {
            'vocab': list(self.vocab),
            'merges': [(pair[0], pair[1], rank) for pair, rank in self.merges.items()],
            'config': {
                'vocab_size': self.vocab_size,
                'min_frequency': self.min_frequency,
                'thresholds': self.analyzer.thresholds,
            },
            'stats': self.stats,
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(model_data, f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, path: str) -> 'EntropyAwareBPE':
        """
        Load model from file.
        
        Args:
            path: Input file path
            
        Returns:
            Loaded EntropyAwareBPE instance
        """
        import json
        
        with open(path, 'r', encoding='utf-8') as f:
            model_data = json.load(f)
        
        # Create instance with saved config
        config = model_data['config']
        instance = cls(
            vocab_size=config['vocab_size'],
            entropy_thresholds=config.get('thresholds'),
            min_frequency=config.get('min_frequency', 2),
        )
        
        # Restore vocab and merges
        instance.vocab = set(model_data['vocab'])
        instance.merges = {
            (pair[0], pair[1]): rank 
            for pair in model_data['merges'] 
            for rank in [pair[2]]
        }
        instance.stats = model_data.get('stats', {})
        
        return instance


# Quick test when run directly
if __name__ == '__main__':
    print("Testing Entropy-Aware BPE...")
    
    # Sample corpus
    corpus = [
        'def calculate(): return x >= 10',
        'if status >= 200: return True',
        'url = "https://api.example.com/users"',
        'for i in range(100): print(i)',
    ] * 5
    
    # Train
    bpe = EntropyAwareBPE(vocab_size=100, min_frequency=2)
    stats = bpe.train(corpus, verbose=True)
    
    # Test encoding
    test_code = 'def fetch(): url = "https://api.test.com/data"; return url if status >= 200 else None'
    tokens = bpe.encode(test_code)
    
    print(f"\n{'='*80}")
    print("ENCODING TEST")
    print(f"{'='*80}")
    print(f"\n📝 Code: {test_code}")
    print(f"🎯 Tokens ({len(tokens)}): {tokens}")
    print(f"📊 Compression: {len(tokens)/len(test_code):.3f} tokens/char")
    
    # Check operator preservation
    operators = ['>=', '<=', '==', '!=', '+=', '-=', '->', '||', '&&']
    preserved = [op for op in operators if op in tokens]
    print(f"\n✅ Preserved operators: {preserved}")