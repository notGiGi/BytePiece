"""
Entropy analyzer for code tokenization
Path: bytepiece/algorithms/entropy/analyzer.py
"""

from collections import Counter
from math import log2
from typing import Tuple
import re


class EntropyAnalyzer:
    """Calculates Shannon entropy and makes fragmentation decisions"""
    
    def __init__(self, 
                 default_threshold: float = 4.0,
                 string_threshold: float = 4.5,
                 identifier_threshold: float = 3.0,
                 comment_threshold: float = 3.5):
        """
        Args:
            default_threshold: Generic threshold for unknown types
            string_threshold: Threshold for string literals (URLs, paths)
            identifier_threshold: Threshold for identifiers (variables, functions)
            comment_threshold: Threshold for comments
        """
        self.thresholds = {
            'default': default_threshold,
            'string': string_threshold,
            'identifier': identifier_threshold,
            'comment': comment_threshold,
        }
        
        # Patterns for detecting construct types
        self.patterns = {
            'url': re.compile(r'https?://|www\.|\.(com|org|net|io|dev)'),
            'path': re.compile(r'[/\\][\w/\\.-]+|[A-Z]:\\'),
            'error_msg': re.compile(r'Error|Exception|Warning|Invalid|Failed'),
            'operator': re.compile(r'^(>=|<=|==|!=|->|=>|\+=|-=|\*=|/=|//|<<|>>|\|\||&&)$'),
            'keyword': re.compile(r'^(def|class|if|else|for|while|return|import|from|try|except|with|as|lambda|yield)$'),
        }
    
    def shannon_entropy(self, text: str) -> float:
        """
        Calculate Shannon entropy of a string.
        
        H(X) = -Σ p(x) * log₂(p(x))
        
        Args:
            text: Input string
            
        Returns:
            Entropy in bits (0 = perfectly predictable, ~5 = highly random)
        """
        if len(text) == 0:
            return 0.0
        
        # Count character frequencies
        counts = Counter(text)
        total = len(text)
        
        # Calculate entropy
        entropy = 0.0
        for count in counts.values():
            prob = count / total
            entropy -= prob * log2(prob)
        
        return entropy
    
    def detect_construct_type(self, token: str) -> str:
        """
        Detect what type of code construct this token represents.
        
        Args:
            token: String to analyze
            
        Returns:
            Type string: 'url', 'path', 'operator', 'keyword', 'error_msg',
                        'string', 'identifier', 'comment', 'default'
        """
        # Check patterns in order of specificity
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
        
        # Heuristics for other types
        if token.startswith('#') or token.startswith('//') or token.startswith('"""'):
            return 'comment'
        
        if token.startswith('"') or token.startswith("'"):
            return 'string'
        
        if token.replace('_', '').isalnum():
            return 'identifier'
        
        return 'default'
    
    def should_fragment(self, token: str, entropy: float = None) -> Tuple[bool, str]:
        """
        Decide whether to fragment a token based on entropy and type.
        
        Args:
            token: Token to analyze
            entropy: Pre-calculated entropy (optional, will compute if None)
            
        Returns:
            Tuple of (should_fragment: bool, reason: str)
        """
        # Always preserve operators and keywords (syntax rule)
        construct_type = self.detect_construct_type(token)
        
        if construct_type in ['operator', 'keyword']:
            return False, f'syntax_rule:{construct_type}'
        
        # Calculate entropy if not provided
        if entropy is None:
            entropy = self.shannon_entropy(token)
        
        # Get threshold for this construct type
        threshold = self.thresholds.get(construct_type, self.thresholds['default'])
        
        # Special cases with fixed rules
        if construct_type == 'url':
            return entropy > 4.5, f'url:H={entropy:.2f}'
        
        if construct_type == 'error_msg':
            return entropy < 2.5, f'error_msg:H={entropy:.2f}'
        
        # General entropy-based decision
        should_frag = entropy > threshold
        return should_frag, f'{construct_type}:H={entropy:.2f},T={threshold}'
    
    def analyze_token(self, token: str) -> dict:
        """
        Full analysis of a token with all metrics.
        
        Returns:
            Dictionary with entropy, type, decision, and reasoning
        """
        entropy = self.shannon_entropy(token)
        construct_type = self.detect_construct_type(token)
        should_frag, reason = self.should_fragment(token, entropy)
        
        return {
            'token': token,
            'entropy': entropy,
            'type': construct_type,
            'should_fragment': should_frag,
            'reason': reason,
            'length': len(token),
        }


# Quick test when run directly
if __name__ == '__main__':
    analyzer = EntropyAnalyzer()
    
    test_tokens = [
        ">=",
        "def",
        "https://api.example.com/v1/users/123",
        "calculate_total_price",
        "x",
    ]
    
    print("ENTROPY ANALYSIS TEST")
    print("=" * 70)
    
    for token in test_tokens:
        result = analyzer.analyze_token(token)
        print(f"\nToken: {token}")
        print(f"  Type: {result['type']}")
        print(f"  Entropy: {result['entropy']:.3f} bits")
        print(f"  Fragment: {result['should_fragment']}")
        print(f"  Reason: {result['reason']}")