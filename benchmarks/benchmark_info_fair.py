"""
REALISTIC & CHALLENGING BENCHMARK: Information-Theoretic Fair BPE

Tests scalability, robustness, and real-world performance across:
- Large corpora (10k+ samples per language)
- Complex morphology (real agglutination, compounds)
- Diverse scripts (Latin, Cyrillic, ideographic simulation)
- Multiple vocab sizes
- Imbalanced corpora
- Performance metrics (time, memory)

Run from project root:
    python benchmarks/benchmark_realistic.py
"""

import sys
import time
import gc
import psutil
import numpy as np
from collections import Counter
from typing import Dict, List, Tuple
from pathlib import Path
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from bytepiece.algorithms.bpe import train_bpe, BPEEncoder
from bytepiece.algorithms.info_fair_bpe import train_info_fair_bpe


class RealisticCorpusGenerator:
    """
    Generate realistic multilingual corpus with authentic morphological complexity.
    
    Languages:
    - English: Analytic (minimal morphology)
    - German: Synthetic (productive compounding)
    - Turkish: Agglutinative (extensive suffix chains)
    - Finnish: Agglutinative (complex case system)
    - Russian: Fusional (case/gender/aspect)
    - Pseudo-Chinese: Logographic simulation (dense characters)
    """
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        self.seed = seed
        
        # Base vocabularies (expanded)
        self.bases = {
            'English': [
                'cat', 'dog', 'house', 'tree', 'book', 'water', 'fire', 'moon',
                'computer', 'phone', 'window', 'door', 'table', 'chair', 'pen',
                'run', 'walk', 'talk', 'think', 'write', 'read', 'sleep', 'eat',
                'big', 'small', 'red', 'blue', 'fast', 'slow', 'good', 'bad'
            ],
            'German': [
                'Katze', 'Hund', 'Haus', 'Baum', 'Buch', 'Wasser', 'Feuer', 'Mond',
                'Computer', 'Telefon', 'Fenster', 'Tür', 'Tisch', 'Stuhl', 'Stift',
                'laufen', 'gehen', 'sprechen', 'denken', 'schreiben', 'lesen', 'schlafen',
                'groß', 'klein', 'rot', 'blau', 'schnell', 'langsam', 'gut', 'schlecht'
            ],
            'Turkish': [
                'kedi', 'köpek', 'ev', 'ağaç', 'kitap', 'su', 'ateş', 'ay',
                'bilgisayar', 'telefon', 'pencere', 'kapı', 'masa', 'sandalye', 'kalem',
                'koş', 'yürü', 'konuş', 'düşün', 'yaz', 'oku', 'uyu', 'ye',
                'büyük', 'küçük', 'kırmızı', 'mavi', 'hızlı', 'yavaş', 'iyi', 'kötü'
            ],
            'Finnish': [
                'kissa', 'koira', 'talo', 'puu', 'kirja', 'vesi', 'tuli', 'kuu',
                'tietokone', 'puhelin', 'ikkuna', 'ovi', 'pöytä', 'tuoli', 'kynä',
                'juosta', 'kävellä', 'puhua', 'ajatella', 'kirjoittaa', 'lukea',
                'iso', 'pieni', 'punainen', 'sininen', 'nopea', 'hidas', 'hyvä', 'huono'
            ],
            'Russian': [
                'кот', 'собака', 'дом', 'дерево', 'книга', 'вода', 'огонь', 'луна',
                'компьютер', 'телефон', 'окно', 'дверь', 'стол', 'стул', 'ручка',
                'бежать', 'идти', 'говорить', 'думать', 'писать', 'читать', 'спать',
                'большой', 'маленький', 'красный', 'синий', 'быстрый', 'медленный'
            ],
            'Chinese': [  # Using Latin simulation for testing
                'mao1', 'gou3', 'fang2', 'shu4', 'shu1', 'shui3', 'huo3', 'yue4',
                'dian4nao3', 'dian4hua4', 'chuang1hu', 'men2', 'zhuo1zi', 'yi3zi',
                'pao3', 'zou3', 'shuo1', 'xiang3', 'xie3', 'du2', 'shui4', 'chi1',
                'da4', 'xiao3', 'hong2', 'lan2', 'kuai4', 'man4', 'hao3', 'huai4'
            ]
        }
        
        # Realistic morphology
        self.morphology = {
            'English': {
                'plural': ['s', 'es'],
                'past': ['ed'],
                'present': ['ing'],
                'adverb': ['ly'],
                'comparative': ['er'],
                'superlative': ['est']
            },
            'German': {
                'plural': ['en', 'e', 'er', 's'],
                'diminutive': ['chen', 'lein'],
                'adjective': ['lich', 'ig', 'isch'],
                # Compounds: concatenate base words
            },
            'Turkish': {
                # Realistic Turkish agglutination chains
                'plural': ['lar', 'ler'],
                'possessive_1sg': ['im', 'ım', 'um', 'üm'],
                'possessive_1pl': ['imiz', 'ımız', 'umuz', 'ümüz'],
                'possessive_2sg': ['in', 'ın', 'un', 'ün'],
                'possessive_3sg': ['i', 'ı', 'u', 'ü'],
                'case_abl': ['dan', 'den', 'tan', 'ten'],
                'case_dat': ['a', 'e', 'ya', 'ye'],
                'case_loc': ['da', 'de', 'ta', 'te'],
                'case_acc': ['i', 'ı', 'u', 'ü', 'yi', 'yı'],
            },
            'Finnish': {
                'plural': ['t'],
                'case_gen': ['n'],
                'case_part': ['a', 'ä', 'ta', 'tä'],
                'case_iness': ['ssa', 'ssä'],
                'case_elat': ['sta', 'stä'],
                'case_illat': ['an', 'än', 'han', 'hän'],
                'case_adess': ['lla', 'llä'],
                'possessive_1sg': ['ni'],
                'possessive_2sg': ['si'],
            },
            'Russian': {
                # Simplified: just suffixes (real Russian has stem changes)
                'plural': ['ы', 'и', 'а'],
                'genitive': ['а', 'я', 'ов', 'ев'],
                'dative': ['у', 'ю', 'ам', 'ям'],
                'accusative': ['а', 'я', 'ы', 'и'],
                'instrumental': ['ом', 'ем', 'ами', 'ями'],
                'prepositional': ['е', 'и', 'ах', 'ях'],
            },
            'Chinese': {
                # Measure words and particles (simulated)
                'measure': ['ge', 'zhi', 'ben', 'tiao'],
                'aspect': ['le', 'guo', 'zhe'],
                'particle': ['de', 'ma', 'ba', 'ne'],
            }
        }
        
        # Function words
        self.function_words = {
            'English': ['the', 'a', 'an', 'is', 'are', 'was', 'were', 'in', 'on', 'at', 'to', 'from'],
            'German': ['der', 'die', 'das', 'ein', 'eine', 'ist', 'sind', 'war', 'waren', 'in', 'an', 'zu'],
            'Turkish': ['bir', 've', 'bu', 'şu', 'o', 'için', 'ile', 'gibi', 'kadar'],
            'Finnish': ['on', 'ovat', 'oli', 'olivat', 'ja', 'tai', 'mutta', 'että', 'jos'],
            'Russian': ['это', 'в', 'на', 'с', 'по', 'о', 'и', 'но', 'что', 'как'],
            'Chinese': ['shi4', 'zai4', 'he2', 'de', 'ma', 'ne', 'le', 'guo']
        }
    
    def generate_turkish_word(self) -> str:
        """Generate realistic Turkish word with suffix chain."""
        base = np.random.choice(self.bases['Turkish'])
        
        # Turkish agglutination: can chain multiple suffixes
        # Order: plural → possessive → case
        word = base
        
        # Plural (30% chance)
        if np.random.random() < 0.3:
            word += np.random.choice(self.morphology['Turkish']['plural'])
        
        # Possessive (40% chance)
        if np.random.random() < 0.4:
            poss_keys = [k for k in self.morphology['Turkish'].keys() if 'possessive' in k]
            poss_key = np.random.choice(poss_keys)
            word += np.random.choice(self.morphology['Turkish'][poss_key])
        
        # Case (50% chance)
        if np.random.random() < 0.5:
            case_keys = [k for k in self.morphology['Turkish'].keys() if 'case' in k]
            case_key = np.random.choice(case_keys)
            word += np.random.choice(self.morphology['Turkish'][case_key])
        
        return word
    
    def generate_finnish_word(self) -> str:
        """Generate realistic Finnish word with case marking."""
        base = np.random.choice(self.bases['Finnish'])
        word = base
        
        # Finnish: extensive case system
        # Can have: plural + case + possessive
        
        # Plural (25% chance)
        if np.random.random() < 0.25:
            word += np.random.choice(self.morphology['Finnish']['plural'])
        
        # Case (60% chance - very common in Finnish)
        if np.random.random() < 0.6:
            case_keys = [k for k in self.morphology['Finnish'].keys() if 'case' in k]
            case_key = np.random.choice(case_keys)
            word += np.random.choice(self.morphology['Finnish'][case_key])
        
        # Possessive (20% chance)
        if np.random.random() < 0.2:
            poss_keys = [k for k in self.morphology['Finnish'].keys() if 'possessive' in k]
            poss_key = np.random.choice(poss_keys)
            word += np.random.choice(self.morphology['Finnish'][poss_key])
        
        return word
    
    def generate_german_word(self) -> str:
        """Generate German word with possible compound."""
        base = np.random.choice(self.bases['German'])
        
        # German compounds (30% chance)
        if np.random.random() < 0.3:
            base2 = np.random.choice(self.bases['German'])
            word = base + base2  # Compound
        else:
            word = base
        
        # Add suffix (20% chance)
        if np.random.random() < 0.2:
            morph_keys = list(self.morphology['German'].keys())
            morph_key = np.random.choice(morph_keys)
            word += np.random.choice(self.morphology['German'][morph_key])
        
        return word
    
    def generate_russian_word(self) -> str:
        """Generate Russian word with case inflection."""
        base = np.random.choice(self.bases['Russian'])
        
        # Russian: case inflection (50% chance)
        if np.random.random() < 0.5:
            case_keys = list(self.morphology['Russian'].keys())
            case_key = np.random.choice(case_keys)
            word = base + np.random.choice(self.morphology['Russian'][case_key])
        else:
            word = base
        
        return word
    
    def generate_chinese_word(self) -> str:
        """Generate pseudo-Chinese word (simulated as Latin)."""
        base = np.random.choice(self.bases['Chinese'])
        
        # Chinese: mostly isolating, add particles (30% chance)
        if np.random.random() < 0.3:
            morph_keys = list(self.morphology['Chinese'].keys())
            morph_key = np.random.choice(morph_keys)
            particle = np.random.choice(self.morphology['Chinese'][morph_key])
            word = base + particle
        else:
            word = base
        
        return word
    
    def generate_english_word(self) -> str:
        """Generate English word with simple morphology."""
        base = np.random.choice(self.bases['English'])
        
        # English: simple suffixes (30% chance)
        if np.random.random() < 0.3:
            morph_keys = list(self.morphology['English'].keys())
            morph_key = np.random.choice(morph_keys)
            word = base + np.random.choice(self.morphology['English'][morph_key])
        else:
            word = base
        
        return word
    
    def generate_sentence(self, lang: str, length: int = None) -> str:
        """Generate a sentence for given language."""
        if length is None:
            length = np.random.randint(3, 12)
        
        words = []
        
        for i in range(length):
            # Function word at start (50% chance)
            if i == 0 and np.random.random() < 0.5:
                words.append(np.random.choice(self.function_words[lang]))
            
            # Content word
            if lang == 'Turkish':
                words.append(self.generate_turkish_word())
            elif lang == 'Finnish':
                words.append(self.generate_finnish_word())
            elif lang == 'German':
                words.append(self.generate_german_word())
            elif lang == 'Russian':
                words.append(self.generate_russian_word())
            elif lang == 'Chinese':
                words.append(self.generate_chinese_word())
            else:  # English
                words.append(self.generate_english_word())
            
            # Function word between (20% chance)
            if i < length - 1 and np.random.random() < 0.2:
                words.append(np.random.choice(self.function_words[lang]))
        
        return ' '.join(words)
    
    def generate_corpus(
        self,
        num_samples: int = 10000,
        languages: List[str] = None
    ) -> Dict[str, List[str]]:
        """Generate realistic multilingual corpus."""
        if languages is None:
            languages = ['English', 'German', 'Turkish', 'Finnish', 'Russian', 'Chinese']
        
        corpora = {}
        
        for lang in languages:
            sentences = []
            for _ in range(num_samples):
                sentence = self.generate_sentence(lang)
                sentences.append(sentence)
            corpora[lang] = sentences
        
        return corpora


def compute_info_metrics(
    encoder: BPEEncoder,
    texts: List[str],
    lang: str
) -> Dict[str, float]:
    """Compute comprehensive metrics for a language."""
    total_tokens = 0
    total_words = 0
    total_chars = 0
    
    for text in texts:
        words = text.split()
        tokens = encoder.encode(text)
        
        total_words += len(words)
        total_tokens += len(tokens)
        total_chars += len(text)
    
    # Compute character-level entropy
    char_counts = Counter()
    for text in texts:
        for char in text:
            char_counts[char] += 1
    
    total_char_count = sum(char_counts.values())
    char_entropy = 0.0
    for count in char_counts.values():
        p = count / total_char_count
        if p > 0:
            char_entropy -= p * np.log2(p)
    
    # Metrics
    fertility = total_tokens / total_words if total_words > 0 else 0
    compression = total_tokens / total_chars if total_chars > 0 else 0
    info_per_token = char_entropy / fertility if fertility > 0 else 0
    
    return {
        'fertility': fertility,
        'compression': compression,
        'char_entropy': char_entropy,
        'info_per_token': info_per_token,
        'total_tokens': total_tokens,
        'total_words': total_words,
        'total_chars': total_chars,
    }


def compute_fairness_metrics(metrics_by_lang: Dict[str, Dict]) -> Dict[str, float]:
    """Compute fairness metrics across languages."""
    fertilities = [m['fertility'] for m in metrics_by_lang.values()]
    infos = [m['info_per_token'] for m in metrics_by_lang.values()]
    
    # Fertility metrics
    fertility_gap = max(fertilities) - min(fertilities)
    fertility_gini = compute_gini(fertilities)
    fertility_var = np.var(fertilities)
    fertility_cv = np.std(fertilities) / np.mean(fertilities) if np.mean(fertilities) > 0 else 0
    
    # Info/token metrics (KEY)
    info_gap = max(infos) - min(infos)
    info_gini = compute_gini(infos)
    info_var = np.var(infos)
    info_cv = np.std(infos) / np.mean(infos) if np.mean(infos) > 0 else 0
    
    # Training cost index
    cost_index = sum(m['total_tokens'] for m in metrics_by_lang.values())
    
    # Cost per language (relative to English)
    english_tokens = metrics_by_lang.get('English', {}).get('total_tokens', 1)
    relative_costs = {
        lang: m['total_tokens'] / english_tokens 
        for lang, m in metrics_by_lang.items()
    }
    
    return {
        'fertility_gap': fertility_gap,
        'fertility_gini': fertility_gini,
        'fertility_variance': fertility_var,
        'fertility_cv': fertility_cv,
        'info_gap': info_gap,
        'info_gini': info_gini,
        'info_variance': info_var,
        'info_cv': info_cv,
        'cost_index': cost_index,
        'relative_costs': relative_costs,
    }


def compute_gini(values: List[float]) -> float:
    """Compute Gini coefficient."""
    if not values or all(v == 0 for v in values):
        return 0.0
    
    sorted_values = sorted(values)
    n = len(sorted_values)
    cumsum = np.cumsum(sorted_values)
    
    return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n


def measure_memory_usage() -> float:
    """Get current memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def run_experiment(
    corpora: Dict[str, List[str]],
    vocab_size: int,
    fairness_weight: float,
    algorithm: str = "standard",
    verbose: bool = True
) -> Dict:
    """Run single experiment with performance tracking."""
    if verbose:
        print(f"\n{'='*80}")
        print(f"Experiment: {algorithm.upper()} | λ={fairness_weight:.1f} | vocab={vocab_size}")
        print(f"{'='*80}")
    
    # Measure memory before
    gc.collect()
    mem_before = measure_memory_usage()
    
    start_time = time.perf_counter()
    
    if algorithm == "standard":
        all_texts = []
        for texts in corpora.values():
            all_texts.extend(texts)
        
        vocab, merges, normalizer = train_bpe(
            texts=all_texts,
            vocab_size=vocab_size,
            byte_fallback=False,
            verbose=False,
        )
        
        encoder = BPEEncoder(vocab, merges, normalizer)
        
    else:  # info_theoretic
        vocab, merges, normalizer, language_stats = train_info_fair_bpe(
            corpora=corpora,
            vocab_size=vocab_size,
            fairness_weight=fairness_weight,
            byte_fallback=False,
            verbose=False,
        )
        
        encoder = BPEEncoder(vocab, merges, normalizer)
    
    training_time = time.perf_counter() - start_time
    
    # Measure memory after
    mem_after = measure_memory_usage()
    mem_used = mem_after - mem_before
    
    # Evaluate
    metrics_by_lang = {}
    for lang, texts in corpora.items():
        metrics = compute_info_metrics(encoder, texts, lang)
        metrics_by_lang[lang] = metrics
    
    fairness = compute_fairness_metrics(metrics_by_lang)
    
    if verbose:
        print(f"Training time: {training_time:.2f}s")
        print(f"Memory used: {mem_used:.1f} MB")
        print(f"\nFairness metrics:")
        print(f"  Fertility gap: {fairness['fertility_gap']:.4f}")
        print(f"  Info/token gap: {fairness['info_gap']:.2f} bits ← KEY")
        print(f"  Cost index: {fairness['cost_index']:.0f} tokens")
    
    return {
        'algorithm': algorithm,
        'lambda': fairness_weight,
        'vocab_size': vocab_size,
        'training_time': training_time,
        'memory_mb': mem_used,
        'metrics_by_lang': metrics_by_lang,
        'fairness': fairness,
    }


def print_detailed_results(result: Dict):
    """Print detailed per-language results."""
    print(f"\nPer-language metrics:")
    print(f"{'Language':<12} | {'Fertility':<10} | {'Info/tok':<10} | {'Rel Cost':<10}")
    print(f"{'-'*12}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
    
    for lang, metrics in result['metrics_by_lang'].items():
        rel_cost = result['fairness']['relative_costs'].get(lang, 1.0)
        print(f"{lang:<12} | {metrics['fertility']:<10.4f} | {metrics['info_per_token']:<10.2f} | {rel_cost:<10.2f}x")


def run_scalability_test(
    generator: RealisticCorpusGenerator,
    vocab_sizes: List[int] = [1000, 5000, 10000],
    corpus_size: int = 5000
):
    """Test scalability across different vocab sizes."""
    print(f"\n{'='*80}")
    print("SCALABILITY TEST: Different Vocabulary Sizes")
    print(f"{'='*80}")
    
    results = []
    
    # Generate corpus once
    print(f"\nGenerating corpus ({corpus_size} samples per language)...")
    corpora = generator.generate_corpus(num_samples=corpus_size)
    
    for vocab_size in vocab_sizes:
        print(f"\n--- Testing vocab_size = {vocab_size} ---")
        
        # Standard BPE
        result_std = run_experiment(
            corpora, vocab_size, 0.0, "standard", verbose=False
        )
        
        # Info-Theoretic BPE
        result_info = run_experiment(
            corpora, vocab_size, 0.3, "info_theoretic", verbose=False
        )
        
        print(f"\nStandard BPE:")
        print(f"  Time: {result_std['training_time']:.2f}s")
        print(f"  Info gap: {result_std['fairness']['info_gap']:.2f} bits")
        
        print(f"\nInfo-Theoretic BPE (λ=0.3):")
        print(f"  Time: {result_info['training_time']:.2f}s")
        print(f"  Info gap: {result_info['fairness']['info_gap']:.2f} bits")
        print(f"  Improvement: {(1 - result_info['fairness']['info_gap']/result_std['fairness']['info_gap'])*100:.1f}%")
        
        results.append({
            'vocab_size': vocab_size,
            'standard': result_std,
            'info_theoretic': result_info
        })
    
    return results


def run_comprehensive_benchmark():
    """Run full benchmark suite."""
    print("="*80)
    print("COMPREHENSIVE REALISTIC BENCHMARK")
    print("Information-Theoretic Fair BPE")
    print("="*80)
    
    generator = RealisticCorpusGenerator(seed=42)
    
    # Test 1: Main comparison with realistic corpus
    print(f"\n{'='*80}")
    print("TEST 1: REALISTIC CORPUS (10k samples, complex morphology)")
    print(f"{'='*80}")
    
    print("\nGenerating realistic multilingual corpus...")
    corpora = generator.generate_corpus(num_samples=10000)
    
    print("\nCorpus statistics:")
    for lang, texts in corpora.items():
        total_words = sum(len(t.split()) for t in texts)
        total_chars = sum(len(t) for t in texts)
        avg_word_len = total_chars / total_words if total_words > 0 else 0
        print(f"  {lang}: {len(texts)} sentences, {total_words} words, avg word len: {avg_word_len:.2f} chars")
    
    vocab_size = 5000
    fairness_weights = [0.0, 0.1, 0.3, 0.5]
    
    results = []
    
    # Baseline
    print(f"\n{'='*40}")
    print("BASELINE: Standard BPE")
    print(f"{'='*40}")
    result = run_experiment(corpora, vocab_size, 0.0, "standard", verbose=True)
    print_detailed_results(result)
    results.append(result)
    
    # Fair BPE variants
    for lam in fairness_weights[1:]:
        print(f"\n{'='*40}")
        print(f"INFO-THEORETIC FAIR BPE: λ={lam:.1f}")
        print(f"{'='*40}")
        result = run_experiment(corpora, vocab_size, lam, "info_theoretic", verbose=True)
        print_detailed_results(result)
        results.append(result)
    
    # Analysis
    print(f"\n{'='*80}")
    print("COMPARATIVE ANALYSIS")
    print(f"{'='*80}")
    
    baseline = results[0]
    best = min(results[1:], key=lambda r: r['fairness']['info_gap'])
    
    print(f"\nBaseline (Standard BPE):")
    print(f"  Info/token gap: {baseline['fairness']['info_gap']:.2f} bits")
    print(f"  Fertility gap: {baseline['fairness']['fertility_gap']:.4f}")
    print(f"  Cost index: {baseline['fairness']['cost_index']:.0f}")
    print(f"  Training time: {baseline['training_time']:.2f}s")
    
    print(f"\nBest (Info-Theoretic λ={best['lambda']:.1f}):")
    print(f"  Info/token gap: {best['fairness']['info_gap']:.2f} bits")
    print(f"  Fertility gap: {best['fairness']['fertility_gap']:.4f}")
    print(f"  Cost index: {best['fairness']['cost_index']:.0f}")
    print(f"  Training time: {best['training_time']:.2f}s")
    
    info_improvement = (1 - best['fairness']['info_gap']/baseline['fairness']['info_gap']) * 100
    cost_change = (best['fairness']['cost_index']/baseline['fairness']['cost_index'] - 1) * 100
    time_overhead = (best['training_time']/baseline['training_time'] - 1) * 100
    
    print(f"\nImprovements:")
    print(f"  Info gap reduction: {info_improvement:.1f}%")
    print(f"  Cost change: {cost_change:+.1f}%")
    print(f"  Time overhead: {time_overhead:+.1f}%")
    
    # Test 2: Scalability
    print(f"\n{'='*80}")
    print("TEST 2: SCALABILITY ACROSS VOCAB SIZES")
    print(f"{'='*80}")
    
    scalability_results = run_scalability_test(
        generator,
        vocab_sizes=[1000, 3000, 5000],
        corpus_size=5000
    )
    
    # Save results
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "benchmark_realistic_results.json"
    
    # Serialize
    def convert_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(item) for item in obj]
        return obj
    
    with open(output_file, 'w') as f:
        json.dump({
            'main_results': convert_types(results),
            'scalability_results': convert_types(scalability_results)
        }, f, indent=2)
    
    print(f"\n✓ Results saved to {output_file}")
    
    print(f"\n{'='*80}")
    print("BENCHMARK COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    run_comprehensive_benchmark()