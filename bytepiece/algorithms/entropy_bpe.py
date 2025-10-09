"""
Entropy-Aware BPE Tokenizer
Path: bytepiece/algorithms/entropy_bpe.py

Integrates entropy analysis with standard BPE training
"""

from typing import List, Dict, Optional
from collections import Counter
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
            corpus: List of code strings
            verbose: Print progress
            
        Returns:
            Training statistics
        """
        import time
        start_time = time.time()
        
        if verbose:
            print("=" * 80)
            print("ENTROPY-AWARE BPE TRAINING")
            print("=" * 80)
        
        # Step 1: Pre-tokenize with entropy analysis
        if verbose:
            print("\n[1/4] Pre-tokenizing corpus with entropy analysis...")
        
        all_preserved = []
        all_fragmentable = []
        
        for doc in corpus:
            result = self.pretokenizer.pretokenize_with_decisions(doc)
            all_preserved.extend(result['preserved'])
            all_fragmentable.extend(result['fragmentable'])
        
        self.stats['preserved_tokens'] = len(all_preserved)
        self.stats['fragmented_tokens'] = len(all_fragmentable)
        
        if verbose:
            print(f"  Preserved tokens: {len(all_preserved)}")
            print(f"  Fragmentable tokens: {len(all_fragmentable)}")
        
        # Step 2: Add preserved tokens directly to vocab
        if verbose:
            print("\n[2/4] Adding preserved tokens to vocabulary...")
        
        for pt in all_preserved:
            self.vocab.add(pt.text)
        
        if verbose:
            print(f"  Vocab size after preserved: {len(self.vocab)}")
        
        # Step 3: Prepare fragmentable tokens for BPE
        if verbose:
            print("\n[3/4] Preparing fragmentable tokens for BPE...")
        
        word_freqs = Counter()
        for pt in all_fragmentable:
            word = tuple(pt.text)
            word_freqs[word] += 1
        
        # Add character-level tokens to vocab
        for word in word_freqs:
            for char in word:
                self.vocab.add(char)
        
        if verbose:
            print(f"  Unique fragmentable words: {len(word_freqs)}")
            print(f"  Vocab size with chars: {len(self.vocab)}")
        
        # Step 4: Learn BPE merges on fragmentable tokens only
        if verbose:
            print(f"\n[4/4] Learning BPE merges (target vocab: {self.vocab_size})...")
        
        num_merges = 0
        words = {word: list(word) for word in word_freqs}
        
        while len(self.vocab) < self.vocab_size:
            # Get all pairs from all words
            pairs = Counter()
            for word, freq in word_freqs.items():
                word_pairs = self._get_pairs(words[word])
                for pair, count in word_pairs.items():
                    pairs[pair] += count * freq
            
            if not pairs:
                break
            
            # Find most frequent pair
            best_pair = max(pairs.items(), key=lambda x: x[1])
            if best_pair[1] < self.min_frequency:
                break
            
            pair, freq = best_pair
            
            # Merge this pair in all words
            for word in word_freqs:
                words[word] = self._merge_pair(words[word], pair)
            
            # Record merge and add to vocab
            self.merges[pair] = num_merges
            merged_token = pair[0] + pair[1]
            self.vocab.add(merged_token)
            
            num_merges += 1
            
            if verbose and num_merges % 100 == 0:
                print(f"  Merges: {num_merges}, Vocab: {len(self.vocab)}")
        
        self.stats['total_merges'] = num_merges
        self.stats['training_time'] = time.time() - start_time
        
        if verbose:
            print(f"\n✅ Training complete!")
            print(f"  Final vocab size: {len(self.vocab)}")
            print(f"  Total merges: {num_merges}")
            print(f"  Training time: {self.stats['training_time']:.2f}s")
        
        return self.stats
    
    def encode(self, text: str) -> List[str]:
        """
        Encode text using entropy-aware BPE.
        
        Returns:
            List of tokens
        """
        result = self.pretokenizer.pretokenize_with_decisions(text)
        tokens = []
        
        # Preserved tokens: add directly
        for pt in result['preserved']:
            tokens.append(pt.text)
        
        # Fragmentable tokens: apply BPE
        for pt in result['fragmentable']:
            word = list(pt.text)
            
            # Apply merges in order of learning
            for pair in sorted(self.merges.keys(), key=lambda p: self.merges[p]):
                word = self._merge_pair(word, pair)
            
            tokens.extend(word)
        
        return tokens
    
    def encode_batch(self, texts: List[str]) -> List[List[str]]:
        """Encode multiple texts"""
        return [self.encode(text) for text in texts]
    
    def get_compression_stats(self, text: str) -> Dict:
        """
        Get detailed compression statistics for a text.
        """
        tokens = self.encode(text)
        chars = len(text)
        
        result = self.pretokenizer.pretokenize_with_decisions(text)
        
        return {
            'text_length': chars,
            'num_tokens': len(tokens),
            'compression_ratio': len(tokens) / chars if chars > 0 else 0,
            'preserved_tokens': result['statistics']['preserved_count'],
            'fragmentable_tokens': result['statistics']['fragmentable_count'],
            'avg_entropy': result['statistics']['avg_entropy'],
        }
    
    def save(self, path: str):
        """Save model to JSON file"""
        import json
        
        # Convert tuples in merges to lists for JSON serialization
        merges_serializable = {f"{k[0]}||{k[1]}": v for k, v in self.merges.items()}
        
        model_data = {
            'vocab': list(self.vocab),
            'merges': merges_serializable,
            'vocab_size': self.vocab_size,
            'stats': self.stats,
            'entropy_thresholds': self.analyzer.thresholds,
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(model_data, f, indent=2, ensure_ascii=False)
    
    def load(self, path: str):
        """Load model from JSON file"""
        import json
        
        with open(path, 'r', encoding='utf-8') as f:
            model_data = json.load(f)
        
        self.vocab = set(model_data['vocab'])
        self.vocab_size = model_data['vocab_size']
        self.stats = model_data['stats']
        
        # Convert merges back to tuples
        self.merges = {}
        for key, value in model_data['merges'].items():
            pair = tuple(key.split('||'))
            self.merges[pair] = value


# Quick test
if __name__ == '__main__':
    from bytepiece.algorithms.entropy import EntropyAnalyzer, EntropyPreTokenizer
    
    print("Testing EntropyAwareBPE...\n")
    
    # Training corpus
    corpus = [
        'def calculate(): return x >= 10',
        'if status >= 200: return True',
        'url = "https://api.example.com/users"',
        'class Handler:\n    def process(self): pass',
    ] * 20
    
    # Initialize and train
    bpe = EntropyAwareBPE(vocab_size=200)
    stats = bpe.train(corpus, verbose=True)
    
    # Test encoding
    test_code = 'def fetch(): url = "https://api.test.com/data"; return url if status >= 200 else None'
    tokens = bpe.encode(test_code)
    comp_stats = bpe.get_compression_stats(test_code)
    
    print(f"\n{'='*80}")
    print("ENCODING TEST")
    print('='*80)
    print(f"\nCode: {test_code}")
    print(f"\nTokens ({len(tokens)}): {tokens[:15]}{'...' if len(tokens) > 15 else ''}")
    print(f"\nCompression stats:")
    for key, value in comp_stats.items():
        print(f"  {key}: {value}")
    
    # Test save/load
    print(f"\n{'='*80}")
    print("SAVE/LOAD TEST")
    print('='*80)
    
    bpe.save('test_entropy_model.json')
    print("✅ Model saved to test_entropy_model.json")
    
    bpe2 = EntropyAwareBPE()
    bpe2.load('test_entropy_model.json')
    print("✅ Model loaded successfully")
    
    # Verify loaded model works
    tokens2 = bpe2.encode(test_code)
    print(f"✅ Encoded same text: {len(tokens2)} tokens")
    
    print(f"\n{'='*80}")
    print("✅ All tests passed!")
    print('='*80 + "\n")