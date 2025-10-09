"""
Entropy-Aware Tokenization for Code - Complete Implementation
ALL IN ONE FILE - No import issues

Save this as: entropy_tokenizer_complete.py
Run: python entropy_tokenizer_complete.py
"""

from collections import Counter
from math import log2
from typing import List, Tuple, Dict
import re
from dataclasses import dataclass


# ============================================================================
# PART 1: ENTROPY ANALYZER
# ============================================================================

class EntropyAnalyzer:
    """Calculates Shannon entropy and makes fragmentation decisions"""
    
    def __init__(self, 
                 default_threshold: float = 4.0,
                 string_threshold: float = 4.5,
                 identifier_threshold: float = 3.0,
                 comment_threshold: float = 3.5):
        self.thresholds = {
            'default': default_threshold,
            'string': string_threshold,
            'identifier': identifier_threshold,
            'comment': comment_threshold,
        }
        
        self.patterns = {
            'url': re.compile(r'https?://|www\.|\.(com|org|net|io|dev)'),
            'path': re.compile(r'[/\\][\w/\\.-]+|[A-Z]:\\'),
            'error_msg': re.compile(r'Error|Exception|Warning|Invalid|Failed'),
            'operator': re.compile(r'^(>=|<=|==|!=|->|=>|\+=|-=|\*=|/=|//|<<|>>|\|\||&&)$'),
            'keyword': re.compile(r'^(def|class|if|else|for|while|return|import|from|try|except|with|as|lambda|yield)$'),
        }
    
    def shannon_entropy(self, text: str) -> float:
        """Calculate Shannon entropy: H(X) = -Σ p(x) * log₂(p(x))"""
        if len(text) == 0:
            return 0.0
        
        counts = Counter(text)
        total = len(text)
        
        entropy = 0.0
        for count in counts.values():
            prob = count / total
            entropy -= prob * log2(prob)
        
        return entropy
    
    def detect_construct_type(self, token: str) -> str:
        """Detect code construct type"""
        if self.patterns['operator'].match(token):
            return 'operator'
        if self.patterns['keyword'].match(token):
            return 'keyword'
        if self.patterns['url'].search(token):
            return 'url'
        if self.patterns['path'].search(token):
            return 'path'
        if self.patterns['error_msg'].search(token):
            return 'error_msg'
        if token.startswith('#') or token.startswith('//'):
            return 'comment'
        if token.startswith('"') or token.startswith("'"):
            return 'string'
        if token.replace('_', '').isalnum():
            return 'identifier'
        return 'default'
    
    def should_fragment(self, token: str, entropy: float = None) -> Tuple[bool, str]:
        """Decide whether to fragment based on entropy + type"""
        construct_type = self.detect_construct_type(token)
        
        # Syntax rules
        if construct_type in ['operator', 'keyword']:
            return False, f'syntax_rule:{construct_type}'
        
        if entropy is None:
            entropy = self.shannon_entropy(token)
        
        threshold = self.thresholds.get(construct_type, self.thresholds['default'])
        
        # Special cases
        if construct_type == 'url':
            return entropy > 4.5, f'url:H={entropy:.2f}'
        if construct_type == 'error_msg':
            return entropy < 2.5, f'error_msg:H={entropy:.2f}'
        
        should_frag = entropy > threshold
        return should_frag, f'{construct_type}:H={entropy:.2f},T={threshold}'
    
    def analyze_token(self, token: str) -> dict:
        """Full analysis of a token"""
        entropy = self.shannon_entropy(token)
        construct_type = self.detect_construct_type(token)
        should_frag, reason = self.should_fragment(token, entropy)
        
        return {
            'token': token,
            'entropy': entropy,
            'type': construct_type,
            'should_fragment': should_frag,
            'reason': reason,
        }


# ============================================================================
# PART 2: PRE-TOKENIZER
# ============================================================================

@dataclass
class PreToken:
    """A pre-tokenized unit with metadata"""
    text: str
    entropy: float
    construct_type: str
    should_fragment: bool


class EntropyPreTokenizer:
    """Pre-tokenizes code using entropy-aware heuristics"""
    
    def __init__(self, entropy_analyzer):
        self.analyzer = entropy_analyzer
        self.token_pattern = re.compile(
            r'>=|<=|==|!=|->|=>|\+=|-=|\*=|/=|//|<<|>>|\|\||&&|'
            r'["\'][^"\']*["\']|'
            r'\#[^\n]*|'
            r'\d+\.?\d*|'
            r'[a-zA-Z_][a-zA-Z0-9_]*|'
            r'\s+|'
            r'[^\w\s]',
            re.VERBOSE
        )
    
    def pretokenize(self, code: str) -> List[PreToken]:
        """Pre-tokenize with entropy analysis"""
        tokens = []
        for match in self.token_pattern.finditer(code):
            token_text = match.group(0)
            if token_text.isspace():
                continue
            
            analysis = self.analyzer.analyze_token(token_text)
            tokens.append(PreToken(
                text=token_text,
                entropy=analysis['entropy'],
                construct_type=analysis['type'],
                should_fragment=analysis['should_fragment']
            ))
        return tokens


# ============================================================================
# PART 3: ENTROPY-AWARE BPE
# ============================================================================

class EntropyAwareBPE:
    """BPE with entropy-aware pre-tokenization"""
    
    def __init__(self, pretokenizer, vocab_size: int = 5000):
        self.pretokenizer = pretokenizer
        self.vocab_size = vocab_size
        self.merges = {}
        self.vocab = set()
    
    def _get_pairs(self, word: List[str]) -> Counter:
        pairs = Counter()
        for i in range(len(word) - 1):
            pairs[(word[i], word[i+1])] += 1
        return pairs
    
    def _merge_pair(self, word: List[str], pair: Tuple[str, str]) -> List[str]:
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
        """Train entropy-aware BPE"""
        # Pre-tokenize
        all_preserved = []
        all_fragmentable = []
        
        for doc in corpus:
            tokens = self.pretokenizer.pretokenize(doc)
            for pt in tokens:
                if pt.should_fragment:
                    all_fragmentable.append(pt)
                else:
                    all_preserved.append(pt)
        
        # Add preserved to vocab
        for pt in all_preserved:
            self.vocab.add(pt.text)
        
        # Learn BPE on fragmentable
        word_freqs = Counter()
        for pt in all_fragmentable:
            word = tuple(pt.text)
            word_freqs[word] += 1
        
        for word in word_freqs:
            for char in word:
                self.vocab.add(char)
        
        words = {word: list(word) for word in word_freqs}
        num_merges = 0
        
        while len(self.vocab) < self.vocab_size:
            pairs = Counter()
            for word, freq in word_freqs.items():
                word_pairs = self._get_pairs(words[word])
                for pair, count in word_pairs.items():
                    pairs[pair] += count * freq
            
            if not pairs:
                break
            
            best_pair = max(pairs.items(), key=lambda x: x[1])[0]
            
            for word in word_freqs:
                words[word] = self._merge_pair(words[word], best_pair)
            
            self.merges[best_pair] = num_merges
            self.vocab.add(best_pair[0] + best_pair[1])
            num_merges += 1
        
        return {
            'vocab_size': len(self.vocab),
            'preserved': len(all_preserved),
            'fragmented': len(all_fragmentable),
            'merges': num_merges,
        }
    
    def encode(self, text: str) -> List[str]:
        """Encode with entropy awareness"""
        pre_tokens = self.pretokenizer.pretokenize(text)
        tokens = []
        
        for pt in pre_tokens:
            if not pt.should_fragment:
                tokens.append(pt.text)
            else:
                word = list(pt.text)
                for pair in sorted(self.merges.keys(), key=lambda p: self.merges[p]):
                    word = self._merge_pair(word, pair)
                tokens.extend(word)
        
        return tokens


# ============================================================================
# PART 4: DEMO & TESTS
# ============================================================================

def demo_entropy_analysis():
    """Demo: Show entropy analysis"""
    print("="*70)
    print("DEMO: Entropy Analysis")
    print("="*70)
    
    analyzer = EntropyAnalyzer()
    
    test_cases = [
        ("Operator", ">="),
        ("Keyword", "def"),
        ("URL", "https://api.example.com/v1/users"),
        ("Variable", "x"),
        ("Function", "calculate_total_price"),
    ]
    
    for label, token in test_cases:
        result = analyzer.analyze_token(token)
        decision = "🔀 FRAGMENT" if result['should_fragment'] else "✅ PRESERVE"
        print(f"\n{label}: '{token}'")
        print(f"  Entropy: {result['entropy']:.3f} bits")
        print(f"  Type: {result['type']}")
        print(f"  Decision: {decision}")


def demo_tokenization():
    """Demo: Full tokenization"""
    print("\n" + "="*70)
    print("DEMO: Entropy-Aware Tokenization")
    print("="*70)
    
    # Initialize
    analyzer = EntropyAnalyzer()
    pretokenizer = EntropyPreTokenizer(analyzer)
    bpe = EntropyAwareBPE(pretokenizer, vocab_size=500)
    
    # Training corpus
    corpus = [
        'def calculate(): return x >= 10',
        'if status >= 200: return True',
        'url = "https://api.example.com/users"',
    ] * 10
    
    # Train
    print("\nTraining...")
    stats = bpe.train(corpus)
    print(f"✅ Trained! Vocab: {stats['vocab_size']}, "
          f"Preserved: {stats['preserved']}, Fragmented: {stats['fragmented']}")
    
    # Test
    test_code = 'def fetch(): url = "https://api.test.com/data"; return url if status >= 200 else None'
    tokens = bpe.encode(test_code)
    
    print(f"\n📝 Code: {test_code}")
    print(f"🎯 Tokens ({len(tokens)}): {tokens[:20]}")
    print(f"📊 Compression: {len(tokens)/len(test_code):.3f} tokens/char")


def main():
    """Run all demos"""
    print("\n" + "="*70)
    print(" "*10 + "🚀 ENTROPY-AWARE TOKENIZATION")
    print("="*70)
    
    demo_entropy_analysis()
    demo_tokenization()
    
    print("\n" + "="*70)
    print("✅ All demos complete!")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()