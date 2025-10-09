"""
Diagnose why operators are being fragmented
Path: bytepiece/benchmarks/diagnose_operators.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bytepiece.algorithms.entropy_bpe import EntropyAwareBPE


def diagnose():
    """Diagnose operator tokenization"""
    
    print("=" * 80)
    print("OPERATOR TOKENIZATION DIAGNOSIS")
    print("=" * 80)
    
    # Initialize
    bpe = EntropyAwareBPE(vocab_size=1000)
    
    # Small training corpus with operators
    corpus = [
        'if x >= 10:',
        'if y <= 20:',
        'if a == b:',
        'if c != d:',
        'result += 5',
        'value -= 3',
    ] * 20
    
    print("\n📚 Training on small corpus with operators...")
    stats = bpe.train(corpus, verbose=False)
    
    print(f"✅ Trained!")
    print(f"  Vocab: {len(bpe.vocab)}")
    print(f"  Preserved: {stats['preserved_tokens']}")
    print(f"  Fragmented: {stats['fragmented_tokens']}")
    
    # Test each operator
    print("\n" + "=" * 80)
    print("OPERATOR TESTS")
    print("=" * 80)
    
    test_cases = [
        'if x >= 10:',
        'if y <= 20:',
        'if a == b:',
        'if c != d:',
        'result += 5',
        'value -= 3',
        'x -> y',
        'a || b',
        'c && d',
    ]
    
    for test in test_cases:
        tokens = bpe.encode(test)
        print(f"\n📝 Code: {test}")
        print(f"🎯 Tokens: {tokens}")
        
        # Check if operator is preserved
        operators = ['>=', '<=', '==', '!=', '+=', '-=', '->', '||', '&&']
        for op in operators:
            if op in test:
                if op in tokens:
                    print(f"   ✅ '{op}' PRESERVED as single token")
                else:
                    # Check if it's in a larger token
                    found_in = [t for t in tokens if op in t]
                    if found_in:
                        print(f"   ⚠️  '{op}' found INSIDE token(s): {found_in}")
                    else:
                        print(f"   ❌ '{op}' FRAGMENTED (not found in any single token)")
    
    # Detailed pre-tokenization analysis
    print("\n" + "=" * 80)
    print("PRE-TOKENIZATION ANALYSIS")
    print("=" * 80)
    
    test_code = 'if x >= 10 and y <= 20:'
    result = bpe.pretokenizer.pretokenize_with_decisions(test_code)
    
    print(f"\n📝 Code: {test_code}")
    print(f"\n🔍 Pre-tokens:")
    
    for pt in result['all_tokens']:
        decision = "PRESERVE" if not pt.should_fragment else "FRAGMENT"
        print(f"  '{pt.text}' → {decision} (type: {pt.construct_type}, H: {pt.entropy:.2f})")
    
    # Check analyzer decisions directly
    print("\n" + "=" * 80)
    print("ANALYZER DECISIONS")
    print("=" * 80)
    
    operators_to_test = ['>=', '<=', '==', '!=', '+=', '-=', '->', '=>']
    
    print(f"\nDirect analyzer tests:")
    for op in operators_to_test:
        result = bpe.analyzer.analyze_token(op)
        decision = "PRESERVE" if not result['should_fragment'] else "FRAGMENT"
        print(f"  '{op}' → {decision} (type: {result['type']}, H: {result['entropy']:.2f})")


if __name__ == '__main__':
    diagnose()