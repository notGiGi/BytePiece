"""
Downstream Evaluation - PROPERLY CHALLENGING
Uses LARGE corpus + SMALL vocab to force real differences

Run: python benchmarks/downstream_eval_github.py
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple
import sys
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

from bytepiece.algorithms.bpe import BPEEncoder, train_bpe
from bytepiece.algorithms.entropy_bpe import EntropyAwareBPE
from bytepiece.algorithms.bpe_variants import EntropyOnlyBPE, SyntaxOnlyBPE
from bytepiece.core.normalizer import Normalizer, SpacerMode, PreTokenizationMode


# ============================================================================
# LOAD EXISTING GITHUB FILES (already downloaded)
# ============================================================================

def load_github_files_if_exist():
    """Load from already downloaded repos"""
    dataset_dir = Path("benchmarks/datasets/temp")
    
    if not dataset_dir.exists():
        return []
    
    all_contents = []
    
    for repo_dir in dataset_dir.iterdir():
        if not repo_dir.is_dir():
            continue
        
        print(f"  Reading {repo_dir.name}...")
        count = 0
        
        for py_file in repo_dir.rglob("*.py"):
            # Skip tests/docs
            if any(x in str(py_file) for x in ["test", "doc", "example", "__pycache__", "venv"]):
                continue
            
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if 50 < len(content) < 50000:  # Reasonable size
                        all_contents.append(content)
                        count += 1
                        
                        if count >= 100:  # Limit per repo
                            break
            except:
                continue
        
        print(f"    ✓ {count} files")
    
    return all_contents


def load_large_corpus():
    """Load LARGE corpus (500+ files)"""
    print("\n[Corpus] Loading...")
    
    # Try GitHub files first
    github_files = load_github_files_if_exist()
    
    if len(github_files) >= 50:
        print(f"  ✓ Loaded {len(github_files)} files from GitHub")
        return github_files
    
    # Fallback: Large synthetic corpus with MANY variations
    print("  ⚠️  Creating large synthetic corpus...")
    
    # Base patterns
    common = [
        'def calc(): return x > 10',
        'if status == 200: return True',
        'for i in range(100): print(i)',
        'while x > 0: x -= 1',
    ]
    
    # Rare operators (THESE are the challenge)
    rare = [
        'result <<= 2',
        'mask >>= shift',
        'flags |= PERMISSION',
        'bits &= MASK',
        'value ^= toggle',
        'power = base ** exponent',
    ]
    
    # Long identifiers
    long_funcs = [
        'def calculate_total_price_with_discount(items): pass',
        'def validate_user_authentication_token(token): pass',
        'def process_payment_transaction_securely(data): pass',
    ]
    
    # Build large corpus
    corpus = []
    corpus.extend(common * 100)  # 400 common samples
    corpus.extend(rare * 5)      # 30 rare samples (challenging!)
    corpus.extend(long_funcs * 10)  # 30 long func samples
    
    print(f"  ✓ Created {len(corpus)} synthetic samples")
    return corpus


# ============================================================================
# EVALUATION TASKS
# ============================================================================

def create_rare_operator_test(corpus: List[str]) -> List[Tuple[str, str]]:
    """
    Extract lines with rare operators.
    Returns (full_line, operator)
    """
    rare_ops = ['<<=', '>>=', '|=', '&=', '^=', '**']
    
    test_cases = []
    for line in corpus:
        for op in rare_ops:
            if op in line:
                test_cases.append((line, op))
    
    return test_cases


def evaluate_rare_operators(tokenizer, test_cases: List[Tuple[str, str]]) -> float:
    """
    Check if rare operators are preserved as single tokens.
    
    Returns: % of rare operators that are single tokens
    """
    if not test_cases:
        return 0.0
    
    preserved = 0
    
    for line, operator in test_cases:
        tokens = tokenizer.encode(line)
        
        # Check if operator is a single token
        if operator in tokens:
            preserved += 1
    
    return preserved / len(test_cases)


def create_long_identifier_test(corpus: List[str]) -> List[Tuple[str, str]]:
    """Extract lines with long function names"""
    import re
    
    test_cases = []
    
    for line in corpus:
        # Find function definitions with long names (20+ chars)
        pattern = r'def\s+([a-z_]{20,})\s*\('
        matches = re.findall(pattern, line)
        
        for func_name in matches:
            test_cases.append((line, func_name))
    
    return test_cases


def evaluate_long_identifiers(tokenizer, test_cases: List[Tuple[str, str]]) -> float:
    """Check if long identifiers are single tokens"""
    if not test_cases:
        return 0.0
    
    preserved = 0
    
    for line, identifier in test_cases:
        tokens = tokenizer.encode(line)
        
        if identifier in tokens:
            preserved += 1
    
    return preserved / len(test_cases)


def cosine_sim(tokens1, tokens2):
    """Cosine similarity"""
    f1, f2 = Counter(tokens1), Counter(tokens2)
    all_t = set(f1.keys()) | set(f2.keys())
    
    dot = sum(f1[t] * f2[t] for t in all_t)
    m1 = sum(v**2 for v in f1.values()) ** 0.5
    m2 = sum(v**2 for v in f2.values()) ** 0.5
    
    return dot / (m1 * m2) if m1 > 0 and m2 > 0 else 0.0


def create_similarity_test(corpus: List[str]) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """Create similar/dissimilar pairs"""
    
    # Similar: same code with variable rename
    similar = []
    for line in corpus[:50]:
        if ' x' in line or ' a' in line:
            similar_line = line.replace(' x', ' y').replace(' a', ' b')
            if similar_line != line:
                similar.append((line, similar_line))
    
    # Dissimilar: completely different code
    dissimilar = []
    functions = [l for l in corpus if 'def ' in l][:20]
    loops = [l for l in corpus if 'for ' in l or 'while ' in l][:20]
    
    for i in range(min(len(functions), len(loops))):
        dissimilar.append((functions[i], loops[i]))
    
    return similar, dissimilar


def evaluate_similarity(tokenizer, similar, dissimilar) -> Dict:
    """Evaluate semantic similarity preservation"""
    
    sim_scores = [cosine_sim(tokenizer.encode(c1), tokenizer.encode(c2)) for c1, c2 in similar]
    dis_scores = [cosine_sim(tokenizer.encode(c1), tokenizer.encode(c2)) for c1, c2 in dissimilar]
    
    avg_sim = sum(sim_scores) / len(sim_scores) if sim_scores else 0
    avg_dis = sum(dis_scores) / len(dis_scores) if dis_scores else 0
    
    return {
        'similar': avg_sim,
        'dissimilar': avg_dis,
        'separation': avg_sim - avg_dis,
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*80)
    print(" "*15 + "DOWNSTREAM EVAL - CHALLENGING")
    print("="*80)
    
    # Load large corpus
    corpus = load_large_corpus()
    print(f"\n[Corpus] Size: {len(corpus)} samples")
    
    # Create test datasets
    print("\n[Test Datasets]")
    rare_op_tests = create_rare_operator_test(corpus)
    long_id_tests = create_long_identifier_test(corpus)
    similar, dissimilar = create_similarity_test(corpus)
    
    print(f"  Rare operators: {len(rare_op_tests)} test cases")
    print(f"  Long identifiers: {len(long_id_tests)} test cases")
    print(f"  Similarity: {len(similar)} similar, {len(dissimilar)} dissimilar")
    
    if len(rare_op_tests) == 0:
        print("\n  ⚠️  WARNING: No rare operators found!")
        print("  Using fallback corpus...")
        corpus = load_large_corpus()  # Force fallback
        rare_op_tests = create_rare_operator_test(corpus)
    
    # Train tokenizers with SMALL vocab (forces tough choices)
    print("\n[Training] Using MEDIUM vocab_size=1000 (balanced!)")
    vocab_size = 1000  # <-- Agregar esta línea
    
    tokenizers = {}
    
    print("  [1/4] Standard BPE...")
    norm = Normalizer(spacer_mode=SpacerMode.NONE, pre_tokenization=PreTokenizationMode.WHITESPACE)
    v, m, _ = train_bpe(corpus, vocab_size=vocab_size, normalizer=norm, byte_fallback=False, verbose=False)
    tokenizers['Standard BPE'] = BPEEncoder(v, m, norm)
    print(f"        Vocab: {len(v)}")
    
    print("  [2/4] Entropy-Only...")
    tok_e = EntropyOnlyBPE(vocab_size=vocab_size, min_frequency=1)
    tok_e.train(corpus, verbose=False)
    tokenizers['Entropy-Only'] = tok_e
    print(f"        Vocab: {len(tok_e.vocab)}")
    
    print("  [3/4] Syntax-Only...")
    tok_s = SyntaxOnlyBPE(vocab_size=vocab_size, min_frequency=1)
    tok_s.train(corpus, verbose=False)
    tokenizers['Syntax-Only'] = tok_s
    print(f"        Vocab: {len(tok_s.vocab)}")
    
    print("  [4/4] Hybrid...")
    tok_h = EntropyAwareBPE(vocab_size=vocab_size, min_frequency=1)
    tok_h.train(corpus, verbose=False)
    tokenizers['Hybrid'] = tok_h
    print(f"        Vocab: {len(tok_h.vocab)}")
    
    # Evaluate
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    
    results = {}
    
    for name, tok in tokenizers.items():
        print(f"\n[{name}]")
        
        # Task 1: Rare operators
        rare_score = evaluate_rare_operators(tok, rare_op_tests)
        print(f"  Rare Operators: {rare_score:.3f} ({int(rare_score*len(rare_op_tests))}/{len(rare_op_tests)})")
        
        # Task 2: Long identifiers
        long_score = evaluate_long_identifiers(tok, long_id_tests)
        print(f"  Long Identifiers: {long_score:.3f} ({int(long_score*len(long_id_tests))}/{len(long_id_tests)})")
        
        # Task 3: Similarity
        sim_res = evaluate_similarity(tok, similar, dissimilar)
        print(f"  Semantic Similarity:")
        print(f"    Similar:    {sim_res['similar']:.3f}")
        print(f"    Dissimilar: {sim_res['dissimilar']:.3f}")
        print(f"    Separation: {sim_res['separation']:.3f} {'✅' if sim_res['separation'] > 0.15 else '⚠️'}")
        
        results[name] = {
            'rare_operators': rare_score,
            'long_identifiers': long_score,
            'semantic_separation': sim_res['separation'],
        }
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    print(f"\n{'Tokenizer':<20} {'Rare Op':>10} {'Long ID':>10} {'Sem Sep':>10} {'Avg':>10}")
    print("-"*80)
    
    for name, res in results.items():
        avg = (res['rare_operators'] + res['long_identifiers'] + res['semantic_separation']) / 3
        print(f"{name:<20} {res['rare_operators']:>10.3f} {res['long_identifiers']:>10.3f} {res['semantic_separation']:>10.3f} {avg:>10.3f}")
    
    # Find winners
    best_rare = max(results.keys(), key=lambda k: results[k]['rare_operators'])
    best_long = max(results.keys(), key=lambda k: results[k]['long_identifiers'])
    
    print(f"\n🏆 Best at rare operators: {best_rare} ({results[best_rare]['rare_operators']:.3f})")
    print(f"🏆 Best at long identifiers: {best_long} ({results[best_long]['long_identifiers']:.3f})")
    
    # Save
    out = Path("benchmarks/results")
    out.mkdir(parents=True, exist_ok=True)
    
    with open(out / "downstream_final.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Saved to benchmarks/results/downstream_final.json")
    print("="*80)


if __name__ == '__main__':
    main()