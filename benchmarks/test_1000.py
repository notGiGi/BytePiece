"""
Test rápido con vocab_size=1000
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.downstream_eval_github import *

def test_vocab_1000():
    # Load corpus
    corpus = load_github_files_if_exist() or load_large_corpus()
    print(f"\n{'='*60}")
    print(f"TESTING WITH VOCAB_SIZE=1000")
    print(f"{'='*60}")
    print(f"Corpus size: {len(corpus)} files\n")
    
    # Create test data
    rare_op_tests = create_rare_operator_test(corpus)
    long_id_tests = create_long_identifier_test(corpus)
    
    # Train 3 tokenizers
    print("[1/3] Training Standard BPE...")
    norm = Normalizer(spacer_mode=SpacerMode.NONE, pre_tokenization=PreTokenizationMode.WHITESPACE)
    v, m, _ = train_bpe(corpus, vocab_size=1000, normalizer=norm, byte_fallback=False, verbose=False)
    tok_standard = BPEEncoder(v, m, norm)
    print(f"      Vocab: {len(v)}")
    
    print("[2/3] Training Entropy-Only...")
    tok_entropy = EntropyOnlyBPE(vocab_size=1000, min_frequency=1)
    tok_entropy.train(corpus, verbose=False)
    print(f"      Vocab: {len(tok_entropy.vocab)}")
    
    print("[3/3] Training Hybrid (Entropy+Syntax)...")
    tok_hybrid = EntropyAwareBPE(vocab_size=1000, min_frequency=1)
    stats = tok_hybrid.train(corpus, verbose=True)
    
    print(f"\n{'='*60}")
    print("EVALUATION RESULTS")
    print(f"{'='*60}")
    
    # Test rare operators
    print("\n🎯 Rare Operators (e.g., <<=, >>=, |=)")
    for name, tok in [("Standard", tok_standard), ("Entropy", tok_entropy), ("Hybrid", tok_hybrid)]:
        score = evaluate_rare_operators(tok, rare_op_tests)
        print(f"  {name:10s}: {score:.1%} preserved")
    
    # Test long identifiers  
    print("\n📏 Long Identifiers (>15 chars)")
    for name, tok in [("Standard", tok_standard), ("Entropy", tok_entropy), ("Hybrid", tok_hybrid)]:
        score = evaluate_long_identifiers(tok, long_id_tests)
        print(f"  {name:10s}: {score:.1%} preserved")
    
    # Sample encoding
    print(f"\n{'='*60}")
    print("SAMPLE ENCODING")
    print(f"{'='*60}")
    
    test_code = 'def calculate_total_price_with_discount(): flags |= PERMISSION_READ'
    
    for name, tok in [("Standard", tok_standard), ("Entropy", tok_entropy), ("Hybrid", tok_hybrid)]:
        tokens = tok.encode(test_code)
        print(f"\n{name}:")
        print(f"  Tokens ({len(tokens)}): {tokens[:15]}{'...' if len(tokens) > 15 else ''}")
        
        # Check specific preservations
        preserved_ops = [op for op in ['|=', '>=', '<<='] if op in tokens]
        preserved_ids = [id for id in ['calculate_total_price_with_discount', 'PERMISSION_READ'] if id in tokens]
        
        if preserved_ops:
            print(f"  ✓ Operators: {preserved_ops}")
        if preserved_ids:
            print(f"  ✓ Identifiers: {preserved_ids}")

if __name__ == "__main__":
    test_vocab_1000()