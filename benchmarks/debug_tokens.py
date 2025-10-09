"""
Debug: See actual tokens produced by both tokenizers
Path: bytepiece/benchmarks/debug_tokens.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bytepiece.algorithms.bpe import BPEEncoder, train_bpe
from bytepiece.algorithms.entropy_bpe import EntropyAwareBPE


def debug_tokens():
    """Compare tokens from both tokenizers"""
    
    print("=" * 80)
    print("TOKEN COMPARISON DEBUG")
    print("=" * 80)
    
    # Training corpus
    corpus = [
        'if x >= 10 and y <= 20:',
        'result += value',
        'items == []',
        'status != 200',
    ] * 10
    
    # Train standard BPE
    print("\n[1/2] Training Standard BPE...")
    vocab, merge_rules, normalizer = train_bpe(corpus, vocab_size=500, verbose=False)
    standard_bpe = BPEEncoder(vocab, merge_rules, normalizer)
    print(f"✅ Vocab size: {len(vocab)}")
    
    # Train entropy BPE
    print("\n[2/2] Training Entropy-Aware BPE...")
    entropy_bpe = EntropyAwareBPE(vocab_size=500)
    entropy_bpe.train(corpus, verbose=False)
    print(f"✅ Vocab size: {len(entropy_bpe.vocab)}")
    
    # Compare on test cases
    print("\n" + "=" * 80)
    print("TOKEN COMPARISON")
    print("=" * 80)
    
    test_cases = [
        'if x >= 10:',
        'result += value',
        'a == b',
        'status != 200',
        'x <= y',
    ]
    
    for test in test_cases:
        print(f"\n📝 Code: {test}")
        
        # Standard BPE
        std_tokens = standard_bpe.encode(test)
        print(f"\n  Standard BPE ({len(std_tokens)} tokens):")
        print(f"    {std_tokens}")
        
        # Check for operators
        operators = ['>=', '<=', '==', '!=', '+=', '-=']
        for op in operators:
            if op in test:
                # Check exact match
                if op in std_tokens:
                    print(f"    ✅ '{op}' found as token")
                else:
                    # Check with spacer
                    spacer_variants = [f'▁{op}', f'{op}▁', f'▁{op}▁']
                    found = [v for v in spacer_variants if v in std_tokens]
                    if found:
                        print(f"    ⚠️  '{op}' found as: {found}")
                    else:
                        # Check substring
                        containing = [t for t in std_tokens if op in t]
                        if containing:
                            print(f"    ⚠️  '{op}' found inside: {containing}")
                        else:
                            print(f"    ❌ '{op}' NOT found")
        
        # Entropy BPE
        ent_tokens = entropy_bpe.encode(test)
        print(f"\n  Entropy BPE ({len(ent_tokens)} tokens):")
        print(f"    {ent_tokens}")
        
        # Check for operators
        for op in operators:
            if op in test:
                if op in ent_tokens:
                    print(f"    ✅ '{op}' found as token")
                else:
                    spacer_variants = [f'▁{op}', f'{op}▁', f'▁{op}▁']
                    found = [v for v in spacer_variants if v in ent_tokens]
                    if found:
                        print(f"    ⚠️  '{op}' found as: {found}")
                    else:
                        containing = [t for t in ent_tokens if op in t]
                        if containing:
                            print(f"    ⚠️  '{op}' found inside: {containing}")
                        else:
                            print(f"    ❌ '{op}' NOT found")
    
    # Analyze token formats
    print("\n" + "=" * 80)
    print("TOKEN FORMAT ANALYSIS")
    print("=" * 80)
    
    test = 'if x >= 10:'
    std_tokens = standard_bpe.encode(test)
    ent_tokens = entropy_bpe.encode(test)
    
    print(f"\nTest: {test}")
    print(f"\nStandard BPE tokens:")
    for i, token in enumerate(std_tokens):
        print(f"  [{i}] '{token}' (len={len(token)}, repr={repr(token)})")
    
    print(f"\nEntropy BPE tokens:")
    for i, token in enumerate(ent_tokens):
        print(f"  [{i}] '{token}' (len={len(token)}, repr={repr(token)})")
    
    # Check if spacer is being used
    print("\n" + "=" * 80)
    print("SPACER DETECTION")
    print("=" * 80)
    
    has_spacer_std = any('▁' in t for t in std_tokens)
    has_spacer_ent = any('▁' in t for t in ent_tokens)
    
    print(f"\nStandard BPE uses spacer ▁: {has_spacer_std}")
    print(f"Entropy BPE uses spacer ▁: {has_spacer_ent}")
    
    if has_spacer_std:
        spacer_tokens_std = [t for t in std_tokens if '▁' in t]
        print(f"  Spacer tokens: {spacer_tokens_std}")
    
    if has_spacer_ent:
        spacer_tokens_ent = [t for t in ent_tokens if '▁' in t]
        print(f"  Spacer tokens: {spacer_tokens_ent}")


if __name__ == '__main__':
    debug_tokens()