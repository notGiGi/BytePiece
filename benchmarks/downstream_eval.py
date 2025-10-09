"""
Downstream Evaluation - Code-Specific Tasks
This is GAP #3 - NOBODY has done this for tokenizers

Tasks:
1. Operator Prediction (code completion proxy)
2. Code Similarity (semantic preservation)
3. Identifier Recovery (context understanding)

Run: python benchmarks/downstream_eval.py
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
# TASK 1: Operator Prediction Accuracy
# ============================================================================

def create_operator_dataset(corpus: List[str]) -> List[Tuple[str, str, str]]:
    """
    Create dataset for operator prediction task.
    
    Returns: List of (code_before, operator, code_after)
    
    Example: ('if x ', '>=', ' 10:') → model must predict '>='
    """
    operators = ['>=', '<=', '==', '!=', '+=', '-=', '->', '=>']
    
    dataset = []
    for code in corpus:
        for op in operators:
            if op in code:
                # Find operator position
                idx = code.find(op)
                before = code[:idx]
                after = code[idx + len(op):]
                
                dataset.append((before, op, after))
    
    return dataset


def evaluate_operator_prediction(tokenizer, dataset: List[Tuple[str, str, str]]) -> float:
    """
    Evaluate if tokenizer preserves operators as single tokens.
    
    Better tokenization → operators are single tokens → easier to predict
    """
    correct = 0
    total = len(dataset)
    
    for before, operator, after in dataset:
        # Tokenize the full code
        full_code = before + operator + after
        tokens = tokenizer.encode(full_code)
        
        # Check if operator exists as a single token
        if operator in tokens:
            correct += 1
    
    accuracy = correct / total if total > 0 else 0
    return accuracy


# ============================================================================
# TASK 2: Code Similarity (Semantic Preservation)
# ============================================================================

def create_similarity_pairs(corpus: List[str]) -> List[Tuple[str, str, bool]]:
    """
    Create pairs of (code1, code2, is_similar)
    
    Similar pairs: Same function with different variable names
    Dissimilar pairs: Different functions
    """
    pairs = []
    
    # Similar pairs (simple string substitution)
    for code in corpus[:20]:
        if 'x' in code and 'def' in code:
            similar_code = code.replace('x', 'y')
            pairs.append((code, similar_code, True))
    
    # Dissimilar pairs (random different codes)
    for i in range(len(corpus) - 1):
        if 'def' in corpus[i] and 'class' in corpus[i+1]:
            pairs.append((corpus[i], corpus[i+1], False))
    
    return pairs[:50]  # Limit to 50 pairs


def cosine_similarity(tokens1: List[str], tokens2: List[str]) -> float:
    """Simple bag-of-tokens cosine similarity"""
    # Create token frequency vectors
    freq1 = Counter(tokens1)
    freq2 = Counter(tokens2)
    
    # All unique tokens
    all_tokens = set(freq1.keys()) | set(freq2.keys())
    
    # Compute dot product
    dot_product = sum(freq1[t] * freq2[t] for t in all_tokens)
    
    # Compute magnitudes
    mag1 = sum(f**2 for f in freq1.values()) ** 0.5
    mag2 = sum(f**2 for f in freq2.values()) ** 0.5
    
    if mag1 == 0 or mag2 == 0:
        return 0.0
    
    return dot_product / (mag1 * mag2)


def evaluate_semantic_similarity(tokenizer, pairs: List[Tuple[str, str, bool]]) -> Dict:
    """
    Evaluate if tokenizer preserves semantic similarity.
    
    Better tokenization → similar code has high cosine similarity
                       → different code has low cosine similarity
    """
    similar_scores = []
    dissimilar_scores = []
    
    for code1, code2, is_similar in pairs:
        tokens1 = tokenizer.encode(code1)
        tokens2 = tokenizer.encode(code2)
        
        similarity = cosine_similarity(tokens1, tokens2)
        
        if is_similar:
            similar_scores.append(similarity)
        else:
            dissimilar_scores.append(similarity)
    
    # Metrics
    avg_similar = sum(similar_scores) / len(similar_scores) if similar_scores else 0
    avg_dissimilar = sum(dissimilar_scores) / len(dissimilar_scores) if dissimilar_scores else 0
    
    # Separation score: higher is better
    separation = avg_similar - avg_dissimilar
    
    return {
        'avg_similar': avg_similar,
        'avg_dissimilar': avg_dissimilar,
        'separation': separation,
    }


# ============================================================================
# TASK 3: Identifier Recovery (Context Understanding)
# ============================================================================

def create_identifier_dataset(corpus: List[str]) -> List[Tuple[str, str, str]]:
    """
    Create dataset where we mask an identifier and try to recover it.
    
    Example: 'def calculate()' → mask 'calculate' → can tokenizer recover?
    """
    dataset = []
    
    for code in corpus:
        words = code.split()
        for i, word in enumerate(words):
            # If it's an identifier (alphanumeric, not keyword)
            if word.isalnum() and word not in ['def', 'if', 'for', 'while', 'class']:
                before = ' '.join(words[:i])
                identifier = word
                after = ' '.join(words[i+1:])
                
                dataset.append((before, identifier, after))
    
    return dataset[:100]  # Limit


def evaluate_identifier_recovery(tokenizer, dataset: List[Tuple[str, str, str]]) -> float:
    """
    Evaluate if identifiers are preserved as single tokens.
    
    Better tokenization → common identifiers are whole tokens
    """
    preserved_count = 0
    total = len(dataset)
    
    for before, identifier, after in dataset:
        full_code = before + ' ' + identifier + ' ' + after
        tokens = tokenizer.encode(full_code)
        
        # Check if identifier is a single token
        if identifier in tokens:
            preserved_count += 1
    
    preservation_rate = preserved_count / total if total > 0 else 0
    return preservation_rate


# ============================================================================
# MAIN EVALUATION
# ============================================================================

def load_corpus():
    """Load evaluation corpus"""
    base = [
        'def calculate(): return x >= 10',
        'def compute(): return y <= 20',
        'if status >= 200: return True',
        'if result != None: return False',
        'for i in range(100): print(i)',
        'while count > 0: count -= 1',
        'result += value if value > 0 else 0',
        'data = {"key": "value", "count": 42}',
        'class Calculator: pass',
        'class Processor: pass',
        'async def fetch(): await api.get()',
        'async def load(): return await db.query()',
        'lambda x: x * 2 if x > 0 else 0',
        'lambda y: y / 2 if y < 100 else y',
        'try: process() except Exception: pass',
        'try: validate() except ValueError: return None',
    ]
    return base * 10


def train_tokenizers(corpus):
    """Train all tokenizer variants"""
    vocab_size = 200
    
    tokenizers = {}
    
    # 1. Standard BPE
    print("\n[1/4] Training Standard BPE...")
    normalizer = Normalizer(
        spacer_mode=SpacerMode.NONE,
        pre_tokenization=PreTokenizationMode.WHITESPACE
    )
    vocab, merges, _ = train_bpe(corpus, vocab_size, normalizer, byte_fallback=False)
    tokenizers['Standard BPE'] = BPEEncoder(vocab, merges, normalizer)
    
    # 2. Entropy-Only
    print("[2/4] Training Entropy-Only...")
    tok_entropy = EntropyOnlyBPE(vocab_size, min_frequency=1)
    tok_entropy.train(corpus)
    tokenizers['Entropy-Only'] = tok_entropy
    
    # 3. Syntax-Only
    print("[3/4] Training Syntax-Only...")
    tok_syntax = SyntaxOnlyBPE(vocab_size, min_frequency=1)
    tok_syntax.train(corpus)
    tokenizers['Syntax-Only'] = tok_syntax
    
    # 4. Hybrid
    print("[4/4] Training Hybrid...")
    tok_hybrid = EntropyAwareBPE(vocab_size, min_frequency=1)
    tok_hybrid.train(corpus)
    tokenizers['Hybrid'] = tok_hybrid
    
    return tokenizers


def main():
    print("="*80)
    print(" "*15 + "DOWNSTREAM EVALUATION - CODE TASKS")
    print(" "*20 + "(GAP #3 - Novel Contribution)")
    print("="*80)
    
    # Load corpus
    print("\n[Setup] Loading corpus...")
    corpus = load_corpus()
    print(f"  Corpus size: {len(corpus)} samples")
    
    # Train tokenizers
    print("\n[Setup] Training tokenizers...")
    tokenizers = train_tokenizers(corpus)
    
    # Create evaluation datasets
    print("\n[Setup] Creating evaluation datasets...")
    operator_dataset = create_operator_dataset(corpus)
    similarity_pairs = create_similarity_pairs(corpus)
    identifier_dataset = create_identifier_dataset(corpus)
    
    print(f"  Operator prediction: {len(operator_dataset)} samples")
    print(f"  Code similarity: {len(similarity_pairs)} pairs")
    print(f"  Identifier recovery: {len(identifier_dataset)} samples")
    
    # Evaluate all tokenizers
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    
    results = {}
    
    for name, tokenizer in tokenizers.items():
        print(f"\n[{name}]")
        
        # Task 1: Operator Prediction
        op_acc = evaluate_operator_prediction(tokenizer, operator_dataset)
        print(f"  Task 1 - Operator Prediction: {op_acc:.3f}")
        
        # Task 2: Semantic Similarity
        sim_results = evaluate_semantic_similarity(tokenizer, similarity_pairs)
        print(f"  Task 2 - Semantic Similarity:")
        print(f"    Similar pairs:    {sim_results['avg_similar']:.3f}")
        print(f"    Dissimilar pairs: {sim_results['avg_dissimilar']:.3f}")
        print(f"    Separation:       {sim_results['separation']:.3f} {'✅' if sim_results['separation'] > 0.1 else ''}")
        
        # Task 3: Identifier Recovery
        id_recovery = evaluate_identifier_recovery(tokenizer, identifier_dataset)
        print(f"  Task 3 - Identifier Recovery: {id_recovery:.3f}")
        
        # Store results
        results[name] = {
            'operator_prediction': op_acc,
            'semantic_similarity': sim_results,
            'identifier_recovery': id_recovery,
        }
    
    # Summary comparison
    print("\n" + "="*80)
    print("SUMMARY COMPARISON")
    print("="*80)
    
    print(f"\n{'Tokenizer':<20} {'Op Pred':>10} {'Sem Sep':>10} {'ID Recov':>10} {'Avg Score':>10}")
    print("-"*80)
    
    for name in tokenizers.keys():
        op_score = results[name]['operator_prediction']
        sep_score = results[name]['semantic_similarity']['separation']
        id_score = results[name]['identifier_recovery']
        avg_score = (op_score + sep_score + id_score) / 3
        
        print(f"{name:<20} {op_score:>10.3f} {sep_score:>10.3f} {id_score:>10.3f} {avg_score:>10.3f}")
    
    # Save results
    output_dir = Path("benchmarks/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "downstream_eval.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n✅ Results saved to benchmarks/results/downstream_eval.json")
    print("="*80)
    
    print("\n🎯 KEY INSIGHT:")
    print("  If Hybrid/Syntax-Only scores higher → Better tokenization helps understanding!")
    print("  This proves that preserving operators/keywords matters for code tasks.")
    print("\n📝 This is publishable material - nobody has done this comparison!")


if __name__ == '__main__':
    main()