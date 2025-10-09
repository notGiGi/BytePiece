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
    
    def train(self, corpus: List[str], verbose: bool = False) -> Dict:
        """
        Train entropy-aware BPE on a corpus.
        
        Args:
            corpus: List of text strings to train on
            verbose: Print training progress
            
        Returns:
            Dictionary with training statistics
        """
        start_time = time.time()
        
        if verbose:
            print("=" * 80)
            print("ENTROPY-AWARE BPE TRAINING")
            print("=" * 80)
        
        # Step 1: Pre-tokenize entire corpus with entropy analysis
        if verbose:
            print("\n[1/4] Pre-tokenizing corpus with entropy analysis...")
        
        all_pre_tokens = []
        for text in corpus:
            all_pre_tokens.extend(self.pretokenizer.pretokenize(text))
        
        # Separate preserved vs fragmentable tokens
        preserved = [pt for pt in all_pre_tokens if not pt.should_fragment]
        fragmentable = [pt for pt in all_pre_tokens if pt.should_fragment]
        
        self.stats['preserved_tokens'] = len(preserved)
        self.stats['fragmented_tokens'] = len(fragmentable)
        
        if verbose:
            print(f"  Preserved tokens: {len(preserved)}")
            print(f"  Fragmentable tokens: {len(fragmentable)}")
        
        # Step 2: Add all preserved tokens directly to vocabulary
        if verbose:
            print("\n[2/4] Adding preserved tokens to vocabulary...")
        
        for pt in preserved:
            self.vocab.add(pt.text)
        
        if verbose:
            print(f"  Vocab size after preserved: {len(self.vocab)}")
        
        # Step 3: Prepare fragmentable tokens for BPE
        if verbose:
            print("\n[3/4] Preparing fragmentable tokens for BPE...")
        
        # Convert fragmentable tokens to character sequences
        words = {}
        for pt in fragmentable:
            # Split into characters
            word = tuple(pt.text)
            words[word] = words.get(word, 0) + 1
            
            # Add individual characters to vocab
            for char in word:
                self.vocab.add(char)
        
        if verbose:
            print(f"  Unique fragmentable words: {len(words)}")
            print(f"  Vocab size with chars: {len(self.vocab)}")
        
        # Step 4: Learn BPE merges on fragmentable tokens only
        if verbose:
            print(f"\n[4/4] Learning BPE merges (target vocab: {self.vocab_size})...")
        
        num_merges = self.vocab_size - len(self.vocab)
        merge_count = 0
        
        for i in range(num_merges):
            # Count all pairs in all words
            pairs = Counter()
            for word, freq in words.items():
                if len(word) < 2:
                    continue
                for j in range(len(word) - 1):
                    pairs[(word[j], word[j+1])] += freq
            
            # Stop if no pairs left
            if not pairs:
                if verbose:
                    print(f"  No more pairs to merge at iteration {i}")
                break
            
            # Get most frequent pair
            most_common_pair, pair_freq = pairs.most_common(1)[0]
            
            # ✅ FIX: Check min_frequency before merging
            if pair_freq < self.min_frequency:
                if verbose:
                    print(f"  Stopping: pair frequency {pair_freq} < min_frequency {self.min_frequency}")
                break
            
            # Record the merge
            self.merges[most_common_pair] = merge_count
            merge_count += 1
            
            # Add merged token to vocab
            merged_token = most_common_pair[0] + most_common_pair[1]
            self.vocab.add(merged_token)
            
            # Update all words by applying this merge
            new_words = {}
            for word, word_freq in words.items():
                new_word = tuple(self._merge_pair(list(word), most_common_pair))
                new_words[new_word] = word_freq
            words = new_words
            
            # Progress update
            if verbose and (i + 1) % 100 == 0:
                print(f"  Merges: {merge_count}, Vocab: {len(self.vocab)}")
        
        # Final statistics
        self.stats['total_merges'] = merge_count
        self.stats['training_time'] = time.time() - start_time
        
        if verbose:
            print(f"\n✅ Training complete!")
            print(f"  Final vocab size: {len(self.vocab)}")
            print(f"  Total merges: {merge_count}")
            print(f"  Training time: {self.stats['training_time']:.2f}s")
        
        return {
            'vocab_size': len(self.vocab),
            'preserved': len(preserved),
            'fragmented': len(fragmentable),
            'merges': merge_count,
            'time': self.stats['training_time'],
        }
    
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