"""
Debug script to verify the 3 variants are actually different

Run: python benchmarks/debug_variants.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bytepiece.algorithms.bpe_variants import EntropyOnlyBPE, SyntaxOnlyBPE
from bytepiece.algorithms.entropy_bpe import EntropyAwareBPE


def test_variants():
    print("="*80)
    print("DEBUG: Testing that variants behave differently")
    print("="*80)
    
    # Simple corpus
    corpus = [
        'def calculate(): return x >= 10',
        'if status >= 200: return True',
    ] * 5
    
    test_code = 'def test(): return x >= 10'
    
    # Train all 3
    print("\n[1/3] Training Entropy-Only...")
    entropy_only = EntropyOnlyBPE(vocab_size=100, min_frequency=1)
    entropy_only.train(corpus, verbose=False)
    tokens_entropy = entropy_only.encode(test_code)
    
    print(f"  Vocab: {len(entropy_only.vocab)}")
    print(f"  Tokens: {tokens_entropy}")
    print(f"  Total: {len(tokens_entropy)}")
    
    print("\n[2/3] Training Syntax-Only...")
    syntax_only = SyntaxOnlyBPE(vocab_size=100, min_frequency=1)
    syntax_only.train(corpus, verbose=False)
    tokens_syntax = syntax_only.encode(test_code)
    
    print(f"  Vocab: {len(syntax_only.vocab)}")
    print(f"  Tokens: {tokens_syntax}")
    print(f"  Total: {len(tokens_syntax)}")
    
    print("\n[3/3] Training Hybrid...")
    hybrid = EntropyAwareBPE(vocab_size=100, min_frequency=1)
    hybrid.train(corpus, verbose=False)
    tokens_hybrid = hybrid.encode(test_code)
    
    print(f"  Vocab: {len(hybrid.vocab)}")
    print(f"  Tokens: {tokens_hybrid}")
    print(f"  Total: {len(tokens_hybrid)}")
    
    # Compare
    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)
    
    print(f"\nAre they different?")
    print(f"  Entropy vs Syntax: {tokens_entropy != tokens_syntax}")
    print(f"  Entropy vs Hybrid:  {tokens_entropy != tokens_hybrid}")
    print(f"  Syntax vs Hybrid:   {tokens_syntax != tokens_hybrid}")
    
    if tokens_entropy == tokens_syntax == tokens_hybrid:
        print("\n❌ ERROR: All 3 variants produce IDENTICAL results!")
        print("   This means they're using the same logic.")
        print("   Check bpe_variants.py implementation.")
    else:
        print("\n✅ GOOD: Variants produce different results!")
        
        # Show which tokens differ
        print("\nKey differences:")
        
        # Check operator preservation
        has_op_entropy = '>=' in tokens_entropy
        has_op_syntax = '>=' in tokens_syntax
        has_op_hybrid = '>=' in tokens_hybrid
        
        print(f"  Operator '>=' preserved:")
        print(f"    Entropy-Only: {has_op_entropy}")
        print(f"    Syntax-Only:  {has_op_syntax}")
        print(f"    Hybrid:       {has_op_hybrid}")
        
        # Check keyword preservation
        has_def_entropy = 'def' in tokens_entropy
        has_def_syntax = 'def' in tokens_syntax
        has_def_hybrid = 'def' in tokens_hybrid
        
        print(f"  Keyword 'def' preserved:")
        print(f"    Entropy-Only: {has_def_entropy}")
        print(f"    Syntax-Only:  {has_def_syntax}")
        print(f"    Hybrid:       {has_def_hybrid}")


if __name__ == '__main__':
    test_variants()