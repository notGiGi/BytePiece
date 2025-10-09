"""
Downstream Evaluation V2 - More Challenging Dataset
Fixes:
1. Larger, more diverse corpus
2. Rare operators and identifiers
3. More similarity pairs
4. Harder evaluation cases

Run: python benchmarks/downstream_eval_v2.py
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple
import sys
from collections import Counter
import random

sys.path.insert(0, str(Path(__file__).parent.parent))

from bytepiece.algorithms.bpe import BPEEncoder, train_bpe
from bytepiece.algorithms.entropy_bpe import EntropyAwareBPE
from bytepiece.algorithms.bpe_variants import EntropyOnlyBPE, SyntaxOnlyBPE
from bytepiece.core.normalizer import Normalizer, SpacerMode, PreTokenizationMode


# ============================================================================
# LARGER, MORE DIVERSE CORPUS
# ============================================================================

def load_challenging_corpus():
    """
    Create a larger corpus with:
    - Common operators (>=, <=, ==) - easy
    - Rare operators (<<, >>, |=) - hard
    - Short identifiers (x, y) - easy
    - Long identifiers (calculate_total_price) - hard
    """
    
    # Common patterns (will be in vocab for all tokenizers)
    common = [
        'def calculate(): return x >= 10',
        'if status >= 200: return True',
        'for i in range(100): print(i)',
        'while count > 0: count -= 1',
        'result += value',
        'data = {"key": "value"}',
        'class Calculator: pass',
        'return x == y',
        'if a != b: raise ValueError',
    ]
    
    # Rare operators (challenging - may not be in all vocabs)
    rare_operators = [
        'result <<= shift_amount',  # Left shift assignment
        'mask >>= 2',  # Right shift assignment
        'flags |= permission',  # Bitwise OR assignment
        'bits &= filter_mask',  # Bitwise AND assignment
        'value ^= toggle',  # XOR assignment
        'a ** b',  # Power operator
        'matrix @ vector',  # Matrix multiplication
    ]
    
    # Long/complex identifiers (challenging)
    long_identifiers = [
        'calculate_total_price_with_discount()',
        'process_user_authentication_request()',
        'validate_input_parameters_and_return_errors()',
        'fetch_database_records_by_query_string()',
        'transform_json_response_to_dataframe()',
    ]
    
    # Similar functions (for similarity test)
    similar_pairs_source = [
        ('def process_data(x): return x * 2', 'def process_data(y): return y * 2'),
        ('def calculate(a, b): return a + b', 'def calculate(x, y): return x + y'),
        ('if status == 200: return True', 'if code == 200: return True'),
        ('for i in range(10): print(i)', 'for j in range(10): print(j)'),
        ('while x > 0: x -= 1', 'while y > 0: y -= 1'),
    ]
    
    # Dissimilar code (for similarity test)
    dissimilar_pairs_source = [
        ('def add(a, b): return a + b', 'class DataProcessor: pass'),
        ('if x > 0: return True', 'async def fetch(): await api.get()'),
        ('for i in range(10): print(i)', 'try: process() except: pass'),
    ]
    
    # Build full corpus
    corpus = []
    corpus.extend(common * 20)  # Common patterns repeated
    corpus.extend(rare_operators * 3)  # Rare operators less frequent
    corpus.extend(long_identifiers * 5)  # Long identifiers
    
    # Add from similarity pairs
    for p1, p2 in similar_pairs_source:
        corpus.extend([p1, p2] * 3)
    
    for p1, p2 in dissimilar_pairs_source:
        corpus.extend([p1, p2] * 2)
    
    return corpus, similar_pairs_source, dissimilar_pairs_source


# ============================================================================
# TASK 1: Operator Prediction (Challenging version)
# ============================================================================

def create_challenging_operator_dataset(corpus: List[str]) -> List[Tuple[str, str, str]]:
    """
    Focus on RARE operators that may not be in all vocabs.
    """
    # These are less likely to be learned by frequency-only BPE
    rare_operators = ['<<=', '>>=', '|=', '&=', '^=', '**', '@']
    
    dataset = []
    for code in corpus:
        for op in rare_operators:
            if op in code:
                idx = code.find(op)
                before = code[:idx]
                after = code[idx + len(op):]
                dataset.append((before, op, after))
    
    return dataset


# ============================================================================
# TASK 2: Semantic Similarity (Fixed version)
# ============================================================================

def evaluate_semantic_similarity_v2(tokenizer, 
                                     similar_pairs: List[Tuple[str, str]],
                                     dissimilar_pairs: List[Tuple[str, str]]) -> Dict:
    """
    Evaluate with properly separated similar and dissimilar pairs.
    """
    def cosine_sim(tokens1, tokens2):
        freq1 = Counter(tokens1)
        freq2 = Counter(tokens2)
        all_tokens = set(freq1.keys()) | set(freq2.keys())
        
        dot = sum(freq1[t] * freq2[t] for t in all_tokens)
        mag1 = sum(f**2 for f in freq1.values()) ** 0.5
        mag2 = sum(f**2 for f in freq2.values()) ** 0.5
        
        if mag1 == 0 or mag2 == 0:
            return 0.0
        return dot / (mag1 * mag2)
    
    # Evaluate similar pairs
    similar_scores = []
    for code1, code2 in similar_pairs:
        tokens1 = tokenizer.encode(code1)
        tokens2 = tokenizer.encode(code2)
        sim = cosine_sim(tokens1, tokens2)
        similar_scores.append(sim)
    
    # Evaluate dissimilar pairs
    dissimilar_scores = []
    for code1, code2 in dissimilar_pairs:
        tokens1 = tokenizer.encode(code1)
        tokens2 = tokenizer.encode(code2)
        sim = cosine_sim(tokens1, tokens2)
        dissimilar_scores.append(sim)
    
    avg_similar = sum(similar_scores) / len(similar_scores) if similar_scores else 0
    avg_dissimilar = sum(dissimilar_scores) / len(dissimilar_scores) if dissimilar_scores else 0
    
    separation = avg_similar - avg_dissimilar
    
    return {
        'avg_similar': avg_similar,
        'avg_dissimilar': avg_dissimilar,
        'separation': separation,
        'num_similar_pairs': len(similar_scores),
        'num_dissimilar_pairs': len(dissimilar_scores),
    }


# ============================================================================
# TASK 3: Long Identifier Preservation
# ============================================================================

def create_long_identifier_dataset(corpus: List[str]) -> List[Tuple[str, str]]:
    """
    Test if long, meaningful identifiers are preserved.
    These are HARD for frequency-only BPE.
    """
    dataset = []
    
    for code in corpus:
        words = code.split()
        for word in words:
            # Long identifiers (>10 chars, alphanumeric)
            if len(word) > 10 and word.replace('_', '').isalnum():
                dataset.append((code, word))
    
    return dataset


def evaluate_long_identifier_preservation(tokenizer, dataset: List[Tuple[str, str]]) -> Dict:
    """
    Check if long identifiers are preserved as single tokens.
    """
    preserved_count = 0
    fragmented_count = 0
    
    for code, identifier in dataset:
        tokens = tokenizer.encode(code)
        
        if identifier in tokens:
            preserved_count += 1
        else:
            # Check if it's fragmented
            if any(identifier in t for t in tokens):
                fragmented_count += 1
    
    total = len(dataset)
    preservation_rate = preserved_count / total if total > 0 else 0
    fragmentation_rate = fragmented_count / total if total > 0 else 0
    
    return {
        'preserved': preserved_count,
        'fragmented': fragmented_count,
        'total': total,
        'preservation_rate': preservation_rate,
        'fragmentation_rate': fragmentation_rate,
    }


# ============================================================================
# MAIN EVALUATION
# ============================================================================

def train_tokenizers(corpus, vocab_size=300):
    """Train all tokenizer variants"""
    tokenizers = {}
    
    print("\n[Training] Standard BPE...")
    normalizer = Normalizer(
        spacer_mode=SpacerMode.NONE,
        pre_tokenization=PreTokenizationMode.WHITESPACE
    )
    vocab, merges, _ = train_bpe(corpus, vocab_size, normalizer, byte_fallback=False, verbose=False)
    tokenizers['Standard BPE'] = BPEEncoder(vocab, merges, normalizer)
    
    print("[Training] Entropy-Only...")
    tok_entropy = EntropyOnlyBPE(vocab_size, min_frequency=1)
    tok_entropy.train(corpus, verbose=False)
    tokenizers['Entropy-Only'] = tok_entropy
    
    print("[Training] Syntax-Only...")
    tok_syntax = SyntaxOnlyBPE(vocab_size, min_frequency=1)
    tok_syntax.train(corpus, verbose=False)
    tokenizers['Syntax-Only'] = tok_syntax
    
    print("[Training] Hybrid...")
    tok_hybrid = EntropyAwareBPE(vocab_size, min_frequency=1)
    tok_hybrid.train(corpus, verbose=False)
    tokenizers['Hybrid'] = tok_hybrid
    
    return tokenizers


def main():
    print("="*80)
    print(" "*10 + "DOWNSTREAM EVALUATION V2 - CHALLENGING DATASET")
    print(" "*20 + "(GAP #3 - Novel Contribution)")
    print("="*80)
    
    # Load challenging corpus
    print("\n[Setup] Loading challenging corpus...")
    corpus, similar_pairs, dissimilar_pairs = load_challenging_corpus()
    print(f"  Corpus size: {len(corpus)} samples")
    print(f"  Similar pairs: {len(similar_pairs)}")
    print(f"  Dissimilar pairs: {len(dissimilar_pairs)}")
    
    # Train tokenizers
    print("\n[Setup] Training tokenizers...")
    tokenizers = train_tokenizers(corpus)
    
    # Create evaluation datasets
    print("\n[Setup] Creating evaluation datasets...")
    rare_op_dataset = create_challenging_operator_dataset(corpus)
    long_id_dataset = create_long_identifier_dataset(corpus)
    
    print(f"  Rare operator prediction: {len(rare_op_dataset)} samples")
    print(f"  Long identifier preservation: {len(long_id_dataset)} samples")
    
    # Evaluate
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    
    results = {}
    
    for name, tokenizer in tokenizers.items():
        print(f"\n[{name}]")
        
        # Task 1: Rare Operator Prediction
        op_correct = sum(1 for _, op, _ in rare_op_dataset if op in tokenizer.encode(_+op+__))
        op_acc = op_correct / len(rare_op_dataset) if rare_op_dataset else 0
        print(f"  Task 1 - Rare Operator Prediction: {op_acc:.3f} ({op_correct}/{len(rare_op_dataset)})")
        
        # Task 2: Semantic Similarity
        sim_results = evaluate_semantic_similarity_v2(tokenizer, similar_pairs, dissimilar_pairs)
        print(f"  Task 2 - Semantic Similarity:")
        print(f"    Similar pairs:    {sim_results['avg_similar']:.3f}")
        print(f"    Dissimilar pairs: {sim_results['avg_dissimilar']:.3f}")
        print(f"    Separation:       {sim_results['separation']:.3f} {'✅' if sim_results['separation'] > 0.15 else '⚠️'}")
        
        # Task 3: Long Identifier Preservation
        id_results = evaluate_long_identifier_preservation(tokenizer, long_id_dataset)
        print(f"  Task 3 - Long Identifier Preservation: {id_results['preservation_rate']:.3f}")
        print(f"    Preserved: {id_results['preserved']}/{id_results['total']}")
        print(f"    Fragmented: {id_results['fragmented']}/{id_results['total']}")
        
        results[name] = {
            'rare_operator_prediction': op_acc,
            'semantic_similarity': sim_results,
            'long_identifier_preservation': id_results,
        }
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY COMPARISON")
    print("="*80)
    
    print(f"\n{'Tokenizer':<20} {'Rare Op':>10} {'Sem Sep':>10} {'Long ID':>10} {'Avg':>10}")
    print("-"*80)
    
    for name in tokenizers.keys():
        rare_op = results[name]['rare_operator_prediction']
        sem_sep = results[name]['semantic_similarity']['separation']
        long_id = results[name]['long_identifier_preservation']['preservation_rate']
        avg = (rare_op + sem_sep + long_id) / 3
        
        print(f"{name:<20} {rare_op:>10.3f} {sem_sep:>10.3f} {long_id:>10.3f} {avg:>10.3f}")
    
    # Save
    output_dir = Path("benchmarks/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "downstream_eval_v2.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n✅ Results saved to benchmarks/results/downstream_eval_v2.json")
    
    # Analysis
    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)
    
    # Find best performers
    best_rare_op = max(tokenizers.keys(), key=lambda k: results[k]['rare_operator_prediction'])
    best_long_id = max(tokenizers.keys(), key=lambda k: results[k]['long_identifier_preservation']['preservation_rate'])
    
    print(f"\n🏆 Best at rare operator prediction: {best_rare_op}")
    print(f"🏆 Best at long identifier preservation: {best_long_id}")
    
    print("\n📝 For paper:")
    syntax_score = results['Syntax-Only']['rare_operator_prediction']
    standard_score = results['Standard BPE']['rare_operator_prediction']
    if syntax_score > standard_score:
        improvement = ((syntax_score - standard_score) / standard_score) * 100
        print(f"  'Syntax-aware tokenization improves rare operator")
        print(f"   prediction by {improvement:.1f}% ({syntax_score:.3f} vs {standard_score:.3f})'")
    
    print("\n" + "="*80)


if __name__ == '__main__':
    main()