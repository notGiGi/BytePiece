"""
Advanced Entropy Analyzer for Code Tokenization
Combines: Shannon entropy + PMI + Left/Right entropy + Syntax rules

This EXTENDS the June 2025 paper by adding:
1. Code-specific syntax rules
2. Context-aware thresholds
3. Multi-signal decision making
"""

from collections import Counter, defaultdict
from math import log2
from typing import List, Tuple, Dict, Optional
import re


class AdvancedEntropyAnalyzer:
    """
    Multi-signal entropy analyzer combining:
    - Shannon entropy (baseline)
    - PMI (pointwise mutual information) 
    - Left/Right entropy (boundary detection)
    - Syntax rules (code-specific)
    """
    
    def __init__(self, 
                 default_threshold: float = 4.0,
                 string_threshold: float = 4.5,
                 identifier_threshold: float = 3.0,
                 pmi_threshold: float = 2.0):
        """
        Args:
            default_threshold: Shannon entropy threshold
            pmi_threshold: PMI threshold for segmentation
        """
        self.thresholds = {
            'default': default_threshold,
            'string': string_threshold,
            'identifier': identifier_threshold,
        }
        self.pmi_threshold = pmi_threshold
        
        # Code-specific patterns
        self.patterns = {
            'operator': re.compile(r'^(>=|<=|==|!=|->|=>|\+=|-=|\*=|/=|//|<<|>>|\|\||&&)$'),
            'keyword': re.compile(r'^(def|class|if|else|for|while|return|import|from|try|except|with|as|lambda|yield|async|await)$'),
            'url': re.compile(r'https?://|www\.|\.(com|org|net|io|dev)'),
        }
        
        # Statistics for PMI calculation (populated during training)
        self.bigram_counts = Counter()  # (token1, token2) -> count
        self.unigram_counts = Counter()  # token -> count
        self.total_bigrams = 0
        
        # Left/Right entropy cache
        self.left_entropy_cache = {}
        self.right_entropy_cache = {}
    
    # ========================================================================
    # METRIC 1: Shannon Entropy (baseline)
    # ========================================================================
    
    def shannon_entropy(self, text: str) -> float:
        """H(X) = -Σ p(x) * log₂(p(x))"""
        if len(text) == 0:
            return 0.0
        
        counts = Counter(text)
        total = len(text)
        
        entropy = 0.0
        for count in counts.values():
            prob = count / total
            entropy -= prob * log2(prob)
        
        return entropy
    
    # ========================================================================
    # METRIC 2: Pointwise Mutual Information (PMI)
    # From the June 2025 paper
    # ========================================================================
    
    def update_statistics(self, tokens: List[str]):
        """
        Update bigram/unigram statistics for PMI calculation.
        Call this during training on your corpus.
        """
        # Update unigram counts
        self.unigram_counts.update(tokens)
        
        # Update bigram counts
        for i in range(len(tokens) - 1):
            bigram = (tokens[i], tokens[i+1])
            self.bigram_counts[bigram] += 1
            self.total_bigrams += 1
    
    def pmi(self, token1: str, token2: str) -> float:
        """
        Calculate PMI between two tokens.
        
        PMI(x,y) = log₂(P(x,y) / (P(x) * P(y)))
        
        High PMI → tokens strongly associated → keep together
        Low/negative PMI → independent → can split
        """
        if self.total_bigrams == 0:
            return 0.0
        
        # P(x,y)
        bigram_count = self.bigram_counts.get((token1, token2), 0)
        if bigram_count == 0:
            return -float('inf')  # Never seen together
        p_xy = bigram_count / self.total_bigrams
        
        # P(x) and P(y)
        total_unigrams = sum(self.unigram_counts.values())
        p_x = self.unigram_counts.get(token1, 0) / total_unigrams
        p_y = self.unigram_counts.get(token2, 0) / total_unigrams
        
        if p_x == 0 or p_y == 0:
            return -float('inf')
        
        # PMI
        pmi_value = log2(p_xy / (p_x * p_y))
        return pmi_value
    
    # ========================================================================
    # METRIC 3: Left/Right Entropy (boundary detection)
    # From the June 2025 paper
    # ========================================================================
    
    def left_entropy(self, token: str, context_tokens: List[str]) -> float:
        """
        Entropy of tokens that appear to the LEFT of this token.
        
        High left entropy → many different predecessors → good boundary
        Low left entropy → predictable predecessor → maybe merge
        """
        if token not in self.left_entropy_cache:
            # Count predecessors
            predecessors = Counter()
            for i, t in enumerate(context_tokens):
                if t == token and i > 0:
                    predecessors[context_tokens[i-1]] += 1
            
            # Calculate entropy
            total = sum(predecessors.values())
            if total == 0:
                self.left_entropy_cache[token] = 0.0
            else:
                entropy = 0.0
                for count in predecessors.values():
                    prob = count / total
                    entropy -= prob * log2(prob)
                self.left_entropy_cache[token] = entropy
        
        return self.left_entropy_cache[token]
    
    def right_entropy(self, token: str, context_tokens: List[str]) -> float:
        """
        Entropy of tokens that appear to the RIGHT of this token.
        
        High right entropy → many different successors → good boundary
        """
        if token not in self.right_entropy_cache:
            # Count successors
            successors = Counter()
            for i, t in enumerate(context_tokens):
                if t == token and i < len(context_tokens) - 1:
                    successors[context_tokens[i+1]] += 1
            
            # Calculate entropy
            total = sum(successors.values())
            if total == 0:
                self.right_entropy_cache[token] = 0.0
            else:
                entropy = 0.0
                for count in successors.values():
                    prob = count / total
                    entropy -= prob * log2(prob)
                self.right_entropy_cache[token] = entropy
        
        return self.right_entropy_cache[token]
    
    # ========================================================================
    # YOUR CONTRIBUTION: Syntax-aware rules (NOT in the paper)
    # ========================================================================
    
    def detect_construct_type(self, token: str) -> str:
        """Detect code construct type"""
        if self.patterns['operator'].match(token):
            return 'operator'
        if self.patterns['keyword'].match(token):
            return 'keyword'
        if self.patterns['url'].search(token):
            return 'url'
        if token.isdigit():
            return 'number'
        if token[0].isalpha() and token.isalnum():
            return 'identifier'
        return 'default'
    
    # ========================================================================
    # COMBINED DECISION: Multi-signal approach
    # ========================================================================
    
    def should_fragment(self, 
                       token: str, 
                       prev_token: Optional[str] = None,
                       next_token: Optional[str] = None,
                       context_tokens: Optional[List[str]] = None) -> Tuple[bool, str, Dict]:
        """
        Multi-signal decision combining ALL metrics.
        
        Returns:
            (should_fragment, reason, all_scores)
        """
        scores = {}
        
        # SIGNAL 1: Syntax rules (highest priority for code)
        construct_type = self.detect_construct_type(token)
        scores['construct_type'] = construct_type
        
        if construct_type in ['operator', 'keyword']:
            return False, f"Preserve {construct_type} for syntax integrity", scores
        
        # SIGNAL 2: Shannon entropy
        shannon_ent = self.shannon_entropy(token)
        scores['shannon_entropy'] = shannon_ent
        
        # SIGNAL 3: PMI with neighbors
        if prev_token and next_token:
            pmi_left = self.pmi(prev_token, token)
            pmi_right = self.pmi(token, next_token)
            scores['pmi_left'] = pmi_left
            scores['pmi_right'] = pmi_right
            
            # High PMI = strong association = don't fragment
            if pmi_left > self.pmi_threshold or pmi_right > self.pmi_threshold:
                return False, f"High PMI (left={pmi_left:.2f}, right={pmi_right:.2f})", scores
        
        # SIGNAL 4: Left/Right entropy (boundary detection)
        if context_tokens:
            left_ent = self.left_entropy(token, context_tokens)
            right_ent = self.right_entropy(token, context_tokens)
            scores['left_entropy'] = left_ent
            scores['right_entropy'] = right_ent
            
            # High boundary entropy = good segmentation point
            avg_boundary_entropy = (left_ent + right_ent) / 2
            if avg_boundary_entropy > 3.0:
                return True, f"High boundary entropy ({avg_boundary_entropy:.2f})", scores
        
        # SIGNAL 5: Shannon entropy threshold (context-specific)
        threshold = self.thresholds.get(construct_type, self.thresholds['default'])
        should_frag = shannon_ent > threshold
        
        reason = f"Shannon entropy {shannon_ent:.2f} {'>' if should_frag else '<='} threshold {threshold}"
        
        return should_frag, reason, scores
    
    # ========================================================================
    # TRAINING: Collect statistics
    # ========================================================================
    
    def train_on_corpus(self, corpus: List[str], tokenizer_fn):
        """
        Train the analyzer on a corpus to collect statistics.
        
        Args:
            corpus: List of code strings
            tokenizer_fn: Function that tokenizes a string into tokens
        """
        print("Training entropy analyzer on corpus...")
        
        all_tokens = []
        for text in corpus:
            tokens = tokenizer_fn(text)
            all_tokens.extend(tokens)
            self.update_statistics(tokens)
        
        print(f"  Collected statistics on {len(all_tokens)} tokens")
        print(f"  Unique bigrams: {len(self.bigram_counts)}")
        print(f"  Unique unigrams: {len(self.unigram_counts)}")
        
        return {
            'total_tokens': len(all_tokens),
            'unique_bigrams': len(self.bigram_counts),
            'unique_unigrams': len(self.unigram_counts),
        }


# ============================================================================
# DEMO: Compare single vs multi-signal
# ============================================================================

def demo_comparison():
    """Show difference between single-signal and multi-signal"""
    
    print("="*80)
    print("COMPARISON: Single-signal vs Multi-signal Entropy")
    print("="*80)
    
    analyzer = AdvancedEntropyAnalyzer()
    
    # Simulate some training data for PMI
    corpus = [
        'def calculate total price',
        'def calculate final result',
        'def process data',
        'if x >= 10',
        'if y >= 20',
        'url https://api.example.com',
    ]
    
    # Simple tokenizer
    def simple_tokenize(text):
        return text.split()
    
    # Train analyzer
    analyzer.train_on_corpus(corpus, simple_tokenize)
    
    # Test cases
    test_cases = [
        ('def', 'calculate', 'total'),
        ('>=', '10', None),
        ('https://api.example.com', None, None),
    ]
    
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    
    for i, (token, prev, next_tok) in enumerate(test_cases, 1):
        print(f"\n[{i}] Token: '{token}'")
        
        # Get all context
        all_tokens = []
        for text in corpus:
            all_tokens.extend(simple_tokenize(text))
        
        should_frag, reason, scores = analyzer.should_fragment(
            token, prev, next_tok, all_tokens
        )
        
        decision = "🔀 FRAGMENT" if should_frag else "✅ PRESERVE"
        print(f"  Decision: {decision}")
        print(f"  Reason: {reason}")
        print(f"  All scores: {scores}")


if __name__ == '__main__':
    demo_comparison()