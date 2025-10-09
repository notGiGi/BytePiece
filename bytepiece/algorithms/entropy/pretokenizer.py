"""
Entropy-aware pre-tokenizer for code
Path: bytepiece/algorithms/entropy/pretokenizer.py
"""

import re
from typing import List, Dict
from dataclasses import dataclass

# Import for when used as module
from .analyzer import EntropyAnalyzer


@dataclass
class PreToken:
    """A pre-tokenized unit with metadata"""
    text: str
    entropy: float
    construct_type: str
    should_fragment: bool
    start_pos: int
    end_pos: int


class EntropyPreTokenizer:
    """
    Pre-tokenizes code using entropy-aware heuristics.
    
    This runs BEFORE BPE to create better initial units.
    """
    
    def __init__(self, entropy_analyzer):
        """
        Args:
            entropy_analyzer: Instance of EntropyAnalyzer
        """
        self.analyzer = entropy_analyzer
        
        # Basic tokenization patterns (similar to regex in BPE)
        self.token_pattern = re.compile(
            r"""
            # Python operators (preserve as units)
            >=|<=|==|!=|->|=>|\+=|-=|\*=|/=|//|<<|>>|\|\||&&|
            # String literals (capture quotes)
            "[^"]*"|'[^']*'|
            # Comments
            \#[^\n]*|
            # Numbers
            \d+\.?\d*|
            # Identifiers
            [a-zA-Z_][a-zA-Z0-9_]*|
            # Whitespace
            \s+|
            # Single special chars
            [^\w\s]
            """,
            re.VERBOSE
        )
    
    def basic_tokenize(self, code: str) -> List[tuple]:
        """
        Basic regex-based tokenization.
        
        Returns:
            List of (token, start_pos, end_pos)
        """
        tokens = []
        for match in self.token_pattern.finditer(code):
            token = match.group(0)
            start = match.start()
            end = match.end()
            tokens.append((token, start, end))
        return tokens
    
    def pretokenize(self, code: str) -> List[PreToken]:
        """
        Pre-tokenize code with entropy analysis.
        
        Args:
            code: Source code string
            
        Returns:
            List of PreToken objects with analysis metadata
        """
        # Step 1: Basic tokenization
        basic_tokens = self.basic_tokenize(code)
        
        # Step 2: Analyze each token with entropy
        pre_tokens = []
        for token_text, start, end in basic_tokens:
            # Skip pure whitespace
            if token_text.isspace():
                continue
            
            # Analyze token
            analysis = self.analyzer.analyze_token(token_text)
            
            pre_token = PreToken(
                text=token_text,
                entropy=analysis['entropy'],
                construct_type=analysis['type'],
                should_fragment=analysis['should_fragment'],
                start_pos=start,
                end_pos=end
            )
            pre_tokens.append(pre_token)
        
        return pre_tokens
    
    def pretokenize_with_decisions(self, code: str) -> Dict:
        """
        Pre-tokenize and return detailed decisions.
        
        Returns:
            Dictionary with:
            - preserved: tokens to keep as units
            - fragmentable: tokens that can be split by BPE
            - statistics: entropy distribution stats
        """
        pre_tokens = self.pretokenize(code)
        
        preserved = []
        fragmentable = []
        
        for pt in pre_tokens:
            if pt.should_fragment:
                fragmentable.append(pt)
            else:
                preserved.append(pt)
        
        # Calculate statistics
        all_entropies = [pt.entropy for pt in pre_tokens]
        stats = {
            'total_tokens': len(pre_tokens),
            'preserved_count': len(preserved),
            'fragmentable_count': len(fragmentable),
            'avg_entropy': sum(all_entropies) / len(all_entropies) if all_entropies else 0,
            'max_entropy': max(all_entropies) if all_entropies else 0,
            'min_entropy': min(all_entropies) if all_entropies else 0,
        }
        
        return {
            'preserved': preserved,
            'fragmentable': fragmentable,
            'statistics': stats,
            'all_tokens': pre_tokens,
        }
    
    def explain(self, code: str, max_tokens: int = 20) -> str:
        """
        Human-readable explanation of tokenization decisions.
        """
        result = self.pretokenize_with_decisions(code)
        
        lines = []
        lines.append("=" * 80)
        lines.append("ENTROPY-AWARE PRE-TOKENIZATION ANALYSIS")
        lines.append("=" * 80)
        lines.append(f"\nCode snippet:\n{code[:200]}")
        if len(code) > 200:
            lines.append("...")
        
        lines.append(f"\n\n📊 Statistics:")
        stats = result['statistics']
        lines.append(f"  Total tokens: {stats['total_tokens']}")
        lines.append(f"  Preserved: {stats['preserved_count']} ({stats['preserved_count']/stats['total_tokens']*100:.1f}%)")
        lines.append(f"  Fragmentable: {stats['fragmentable_count']} ({stats['fragmentable_count']/stats['total_tokens']*100:.1f}%)")
        lines.append(f"  Avg entropy: {stats['avg_entropy']:.3f} bits")
        lines.append(f"  Entropy range: [{stats['min_entropy']:.3f}, {stats['max_entropy']:.3f}]")
        
        lines.append(f"\n\n✅ Preserved tokens (will NOT be split by BPE):")
        for i, pt in enumerate(result['preserved'][:max_tokens]):
            lines.append(f"  {i+1}. '{pt.text}' - {pt.construct_type} (H={pt.entropy:.3f})")
        if len(result['preserved']) > max_tokens:
            lines.append(f"  ... and {len(result['preserved']) - max_tokens} more")
        
        lines.append(f"\n\n🔀 Fragmentable tokens (BPE can split these):")
        for i, pt in enumerate(result['fragmentable'][:max_tokens]):
            lines.append(f"  {i+1}. '{pt.text[:30]}...' - {pt.construct_type} (H={pt.entropy:.3f})")
        if len(result['fragmentable']) > max_tokens:
            lines.append(f"  ... and {len(result['fragmentable']) - max_tokens} more")
        
        return "\n".join(lines)


# Test when run directly
if __name__ == '__main__':
    # FIXED: Import using absolute path from bytepiece package
    from bytepiece.algorithms.entropy.analyzer import EntropyAnalyzer
    
    analyzer = EntropyAnalyzer()
    pretokenizer = EntropyPreTokenizer(analyzer)
    
    test_code = """
def calculate_price(items, discount=0.1):
    # TODO: add tax calculation
    total = sum(item.price for item in items)
    url = "https://api.example.com/pricing/v2/calculate"
    
    if total >= 100:
        total *= (1 - discount)
    
    return total
"""
    
    print(pretokenizer.explain(test_code))