"""
Information-Theoretic Fair BPE

Balances tokenization across languages by equalizing information density
(bits per token) rather than raw token count.

Mathematical foundation:
- Shannon entropy H(L) measures information content in bits
- Information per token: I(L) = H(L) / num_tokens
- Goal: Maximize min_L I(L) → ensure worst language has dense tokens

This guarantees training cost proportional to information content,
not language structure.
"""

import logging
import time
import numpy as np
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

from bytepiece.core.normalizer import Normalizer
from bytepiece.core.vocab import MergeRules, Vocabulary

logger = logging.getLogger(__name__)


class InformationTheoreticFairBPE:
    """
    Fair BPE based on information theory.
    
    Instead of equalizing tokens/word (fertility), equalizes bits/token
    (information density). This ensures training cost is proportional to
    semantic information, not language morphology.
    
    Key insight: Turkish and English expressing "from our houses" have
    the same semantic information (~30 bits), but Turkish uses more tokens
    in standard BPE. This algorithm compensates by making Turkish tokens
    more information-dense.
    
    Attributes:
        vocab_size: Target vocabulary size
        fairness_weight: λ parameter (0=standard BPE, 1=max fairness)
        normalizer: Text normalizer
        byte_fallback: Enable byte fallback
        verbose: Print training progress
    """
    
    def __init__(
        self,
        vocab_size: int,
        fairness_weight: float = 0.3,
        normalizer: Optional[Normalizer] = None,
        byte_fallback: bool = True,
        verbose: bool = False,
    ):
        """Initialize Information-Theoretic Fair BPE trainer.
        
        Args:
            vocab_size: Target vocabulary size
            fairness_weight: Balance between frequency and fairness (0-1)
                0.0 = Standard BPE (pure frequency)
                0.3 = Balanced (recommended)
                1.0 = Maximum information density fairness
            normalizer: Text normalizer (creates default if None)
            byte_fallback: Enable byte-fallback for full coverage
            verbose: Print training progress
        """
        if not 0 <= fairness_weight <= 1:
            raise ValueError(f"fairness_weight must be in [0, 1], got {fairness_weight}")
        
        self.vocab_size = vocab_size
        self.fairness_weight = fairness_weight
        self.normalizer = normalizer or Normalizer()
        self.byte_fallback = byte_fallback
        self.verbose = verbose
        
        # Will be populated during training
        self.vocab: Optional[Vocabulary] = None
        self.merge_rules: Optional[MergeRules] = None
        self.language_stats: Dict[str, Dict] = {}
        
        # Information theory metrics
        self.char_entropy: Dict[str, float] = {}  # H(L) at character level
        self.info_per_token: Dict[str, float] = {}  # bits/token
        
        # Performance optimization: cache pair counts per language
        self.pair_counts_cache: Dict[Tuple[str, Tuple[str, str]], int] = defaultdict(int)
    
    def train(
        self,
        corpora: Dict[str, List[str]],
        seed: int = 42
    ) -> Tuple[Vocabulary, MergeRules, Normalizer, Dict[str, Dict]]:
        """
        Train Information-Theoretic Fair BPE on multilingual corpus.
        
        Args:
            corpora: Dictionary mapping language name to list of texts
            seed: Random seed (for future stochastic extensions)
            
        Returns:
            Tuple of (vocabulary, merge_rules, normalizer, language_stats)
            
            language_stats includes:
                - total_tokens, total_words, fertility (standard)
                - char_entropy (bits of information at char level)
                - info_per_token (bits of information per token)
                - compression_ratio (for analysis)
        """
        if not corpora:
            raise ValueError("corpora cannot be empty")
        
        for lang, texts in corpora.items():
            if not texts:
                raise ValueError(f"Language '{lang}' has no texts")
        
        if self.verbose:
            logger.info(f"Training Info-Theoretic Fair BPE (λ={self.fairness_weight})")
            logger.info(f"  Languages: {list(corpora.keys())}")
            logger.info(f"  Target vocab size: {self.vocab_size}")
        
        # Initialize
        self.vocab = Vocabulary(byte_fallback=self.byte_fallback)
        self.merge_rules = MergeRules()
        
        # STEP 1: Compute character-level entropy for each language
        # This is the "information content" baseline
        if self.verbose:
            logger.info("  Computing character-level entropy...")
        
        for lang, texts in corpora.items():
            self.char_entropy[lang] = self._compute_char_entropy(texts)
            if self.verbose:
                logger.info(f"    {lang}: H={self.char_entropy[lang]:.2f} bits")
        
        # STEP 2: Standard BPE initialization
        word_freqs_by_lang = self._preprocess_corpora(corpora)
        self._build_base_vocabulary(word_freqs_by_lang)
        
        if self.verbose:
            logger.info(f"  Base vocab size: {len(self.vocab)}")
        
        # STEP 3: Initialize stats and cache
        self._initialize_language_stats(word_freqs_by_lang)
        self._initialize_pair_cache(word_freqs_by_lang)
        
        # STEP 4: Information-theoretic merge selection
        self._perform_merges(word_freqs_by_lang)
        
        if self.verbose:
            logger.info(f"  Final vocab size: {len(self.vocab)}")
            logger.info(f"  Total merges: {len(self.merge_rules)}")
            self._log_final_stats()
        
        return self.vocab, self.merge_rules, self.normalizer, self.language_stats
    
    def _compute_char_entropy(self, texts: List[str]) -> float:
        """
        Compute Shannon entropy at character level.
        
        H(X) = -Σ p(x) log₂(p(x))
        
        This measures the information content in bits.
        Higher entropy = more information/more random.
        
        Args:
            texts: List of text strings
            
        Returns:
            Entropy in bits
        """
        char_counts = Counter()
        total_chars = 0
        
        for text in texts:
            # Normalize first to get actual characters we'll tokenize
            normalized = self.normalizer.normalize(text)
            for char in normalized:
                char_counts[char] += 1
                total_chars += 1
        
        if total_chars == 0:
            return 0.0
        
        # Shannon entropy: -Σ p(x) log₂ p(x)
        entropy = 0.0
        for count in char_counts.values():
            p = count / total_chars
            if p > 0:
                entropy -= p * np.log2(p)
        
        return entropy
    
    def _preprocess_corpora(
        self,
        corpora: Dict[str, List[str]]
    ) -> Dict[str, Counter]:
        """Pre-tokenize and normalize all texts by language."""
        word_freqs_by_lang = {}
        
        for lang, texts in corpora.items():
            word_freqs = Counter()
            
            for text in texts:
                chunks = self.normalizer.pre_tokenize(text)
                
                for chunk in chunks:
                    normalized = self.normalizer.normalize(chunk)
                    tokens = tuple(self.vocab.encode_with_fallback(normalized))
                    word_freqs[tokens] += 1
            
            word_freqs_by_lang[lang] = word_freqs
            
            if self.verbose:
                logger.info(f"  {lang}: {sum(word_freqs.values())} words, "
                          f"{len(word_freqs)} unique")
        
        return word_freqs_by_lang
    
    def _build_base_vocabulary(self, word_freqs_by_lang: Dict[str, Counter]) -> None:
        """Extract all unique characters and add to vocabulary."""
        all_chars = set()
        
        for word_freqs in word_freqs_by_lang.values():
            for word_tuple in word_freqs:
                all_chars.update(word_tuple)
        
        self.vocab.add_tokens(sorted(all_chars))
    
    def _initialize_language_stats(self, word_freqs_by_lang: Dict[str, Counter]) -> None:
        """Initialize statistics per language."""
        for lang, word_freqs in word_freqs_by_lang.items():
            total_tokens = sum(len(word) * freq for word, freq in word_freqs.items())
            total_words = sum(word_freqs.values())
            total_chars = sum(sum(len(token) for token in word) * freq 
                            for word, freq in word_freqs.items())
            
            self.language_stats[lang] = {
                'total_tokens': total_tokens,
                'total_words': total_words,
                'total_chars': total_chars,
                'fertility': total_tokens / total_words if total_words > 0 else 0.0,
                'char_entropy': self.char_entropy[lang],
                'info_per_token': 0.0,  # Will be updated during training
            }
    
    def _initialize_pair_cache(self, word_freqs_by_lang: Dict[str, Counter]) -> None:
        """Precompute pair counts per language for O(1) lookup."""
        if self.verbose:
            logger.info("  Building pair count cache...")
        
        start = time.perf_counter()
        
        for lang, word_freqs in word_freqs_by_lang.items():
            for word_tuple, freq in word_freqs.items():
                for i in range(len(word_tuple) - 1):
                    pair = (word_tuple[i], word_tuple[i + 1])
                    self.pair_counts_cache[(lang, pair)] += freq
        
        if self.verbose:
            elapsed = time.perf_counter() - start
            logger.info(f"  Cache built in {elapsed:.2f}s "
                       f"({len(self.pair_counts_cache)} entries)")
    
    def _perform_merges(self, word_freqs_by_lang: Dict[str, Counter]) -> None:
        """Main training loop with information-theoretic selection."""
        base_size = len(self.vocab)
        num_merges = self.vocab_size - base_size
        
        if num_merges <= 0:
            if self.verbose:
                logger.info("  Target vocab size already reached")
            return
        
        start_time = time.perf_counter()
        
        for merge_idx in range(num_merges):
            # Compute pair frequencies
            pair_freqs = self._compute_pair_frequencies(word_freqs_by_lang)
            
            if not pair_freqs:
                if self.verbose:
                    logger.info(f"  No more pairs at iteration {merge_idx}")
                break
            
            # Select best pair with information-theoretic fairness
            best_pair = self._select_best_pair(pair_freqs)
            
            # Progress logging
            if self.verbose and (merge_idx % 500 == 0 or merge_idx == num_merges - 1):
                elapsed = time.perf_counter() - start_time
                self._update_info_per_token()
                min_info = min(self.info_per_token.values())
                max_info = max(self.info_per_token.values())
                logger.info(f"  Merge {merge_idx}/{num_merges} | "
                          f"Info/tok: [{min_info:.2f}, {max_info:.2f}] bits | "
                          f"Time: {elapsed:.1f}s")
            
            # Apply merge
            self.merge_rules.add_merge(best_pair)
            new_token = best_pair[0] + best_pair[1]
            self.vocab.add_token(new_token)
            
            # Update word frequencies and caches
            word_freqs_by_lang = self._apply_merge_to_all_languages(
                word_freqs_by_lang, best_pair
            )
    
    def _compute_pair_frequencies(
        self,
        word_freqs_by_lang: Dict[str, Counter]
    ) -> Counter:
        """Compute pair frequencies across all languages."""
        pair_freqs = Counter()
        
        for word_freqs in word_freqs_by_lang.values():
            for word_tuple, freq in word_freqs.items():
                for i in range(len(word_tuple) - 1):
                    pair = (word_tuple[i], word_tuple[i + 1])
                    pair_freqs[pair] += freq
        
        return pair_freqs
    
    def _update_info_per_token(self) -> None:
        """Update information per token for all languages."""
        for lang in self.language_stats.keys():
            total_tokens = self.language_stats[lang]['total_tokens']
            if total_tokens > 0:
                self.info_per_token[lang] = self.char_entropy[lang] / total_tokens
            else:
                self.info_per_token[lang] = 0.0
            
            # Update in stats too
            self.language_stats[lang]['info_per_token'] = self.info_per_token[lang]
    
    def _select_best_pair(self, pair_freqs: Counter) -> Tuple[str, str]:
        """
        Select best pair using information-theoretic fairness.
        
        Score = frequency + λ × fairness_bonus
        
        fairness_bonus rewards merges that increase min(info_per_token),
        i.e., help the language with lowest information density.
        
        This ensures the "most expensive" language gets denser tokens.
        """
        if self.fairness_weight == 0:
            # Standard BPE: just pick most frequent
            return pair_freqs.most_common(1)[0][0]
        
        # Update current info/token
        self._update_info_per_token()
        
        # Find minimum information density (worst language)
        current_min_info = min(self.info_per_token.values())
        
        best_pair = None
        best_score = float('-inf')
        
        for pair, freq in pair_freqs.items():
            # Base score: frequency (compression)
            base_score = float(freq)
            
            # Estimate new min_info after this merge
            new_min_info = float('inf')
            
            for lang in self.language_stats.keys():
                pair_count = self.pair_counts_cache.get((lang, pair), 0)
                new_tokens = self.language_stats[lang]['total_tokens'] - pair_count
                
                if new_tokens > 0:
                    new_info = self.char_entropy[lang] / new_tokens
                    new_min_info = min(new_min_info, new_info)
            
            # Fairness bonus: reward merges that increase min_info
            info_gain = new_min_info - current_min_info
            
            if info_gain > 0:
                # Helps worst language → strong reward
                fairness_bonus = info_gain * 10000
            else:
                # Doesn't help worst language → mild penalty
                fairness_bonus = info_gain * 2000
            
            # Combined score
            score = base_score + self.fairness_weight * fairness_bonus
            
            if score > best_score:
                best_score = score
                best_pair = pair
        
        return best_pair
    
    def _apply_merge_to_all_languages(
        self,
        word_freqs_by_lang: Dict[str, Counter],
        pair: Tuple[str, str]
    ) -> Dict[str, Counter]:
        """Apply merge to all languages and update statistics."""
        new_word_freqs_by_lang = {}
        
        for lang, word_freqs in word_freqs_by_lang.items():
            new_word_freqs = Counter()
            
            for word_tuple, freq in word_freqs.items():
                new_word = self._apply_merge(word_tuple, pair)
                new_word_freqs[new_word] += freq
            
            new_word_freqs_by_lang[lang] = new_word_freqs
            
            # Update language stats
            total_tokens = sum(len(word) * freq for word, freq in new_word_freqs.items())
            total_words = sum(new_word_freqs.values())
            
            self.language_stats[lang]['total_tokens'] = total_tokens
            self.language_stats[lang]['total_words'] = total_words
            self.language_stats[lang]['fertility'] = total_tokens / total_words if total_words > 0 else 0.0
        
        # Update pair cache incrementally
        self._update_pair_cache(word_freqs_by_lang, new_word_freqs_by_lang, pair)
        
        return new_word_freqs_by_lang
    
    def _update_pair_cache(
        self,
        old_word_freqs: Dict[str, Counter],
        new_word_freqs: Dict[str, Counter],
        merged_pair: Tuple[str, str]
    ) -> None:
        """Incrementally update pair cache after merge."""
        for lang in old_word_freqs.keys():
            # Remove old pairs
            for word_tuple, freq in old_word_freqs[lang].items():
                for i in range(len(word_tuple) - 1):
                    pair = (word_tuple[i], word_tuple[i + 1])
                    self.pair_counts_cache[(lang, pair)] -= freq
                    if self.pair_counts_cache[(lang, pair)] <= 0:
                        del self.pair_counts_cache[(lang, pair)]
            
            # Add new pairs
            for word_tuple, freq in new_word_freqs[lang].items():
                for i in range(len(word_tuple) - 1):
                    pair = (word_tuple[i], word_tuple[i + 1])
                    self.pair_counts_cache[(lang, pair)] += freq
    
    def _apply_merge(
        self,
        word: Tuple[str, ...],
        pair: Tuple[str, str]
    ) -> Tuple[str, ...]:
        """Apply merge to a single word."""
        if len(word) < 2:
            return word
        
        new_word = []
        i = 0
        
        while i < len(word):
            if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
                new_word.append(word[i] + word[i + 1])
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        
        return tuple(new_word)
    
    def _log_final_stats(self) -> None:
        """Log final training statistics."""
        self._update_info_per_token()
        
        logger.info("  Final language statistics:")
        for lang, stats in sorted(self.language_stats.items()):
            logger.info(f"    {lang}:")
            logger.info(f"      Fertility: {stats['fertility']:.4f} tokens/word")
            logger.info(f"      Info/token: {stats['info_per_token']:.2f} bits")
            logger.info(f"      Char entropy: {stats['char_entropy']:.2f} bits")
        
        # Summary metrics
        fertilities = [s['fertility'] for s in self.language_stats.values()]
        infos = [s['info_per_token'] for s in self.language_stats.values()]
        
        logger.info(f"  Fertility gap: {max(fertilities) - min(fertilities):.4f}")
        logger.info(f"  Info/token gap: {max(infos) - min(infos):.2f} bits")


def train_info_fair_bpe(
    corpora: Dict[str, List[str]],
    vocab_size: int,
    fairness_weight: float = 0.3,
    normalizer: Optional[Normalizer] = None,
    byte_fallback: bool = True,
    verbose: bool = False,
    seed: int = 42,
) -> Tuple[Vocabulary, MergeRules, Normalizer, Dict[str, Dict]]:
    """
    Convenience function to train Information-Theoretic Fair BPE.
    
    This is a mathematically principled variant of BPE that equalizes
    information density (bits/token) rather than raw token count.
    
    Based on Shannon's information theory, it ensures training cost
    is proportional to semantic information, not language structure.
    
    Args:
        corpora: Dictionary mapping language name to list of texts
        vocab_size: Target vocabulary size
        fairness_weight: Balance between frequency and fairness (0-1)
            0.0 = Standard BPE (pure frequency-based)
            0.3 = Balanced (recommended)
            1.0 = Maximum information density fairness
        normalizer: Text normalizer (creates default if None)
        byte_fallback: Enable byte-fallback for full coverage
        verbose: Print training progress
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (vocabulary, merge_rules, normalizer, language_stats)
        
        language_stats includes information-theoretic metrics:
        - char_entropy: Shannon entropy at character level (bits)
        - info_per_token: Information density (bits/token)
        - fertility: Traditional tokens/word metric
        
    Example:
        >>> corpora = {
        ...     "English": ["hello world", "test"],
        ...     "Turkish": ["merhaba dünya", "test"],
        ... }
        >>> vocab, merges, norm, stats = train_info_fair_bpe(
        ...     corpora=corpora,
        ...     vocab_size=1000,
        ...     fairness_weight=0.3,
        ...     verbose=True
        ... )
        >>> print(f"English: {stats['English']['info_per_token']:.2f} bits/token")
        >>> print(f"Turkish: {stats['Turkish']['info_per_token']:.2f} bits/token")
    """
    if verbose:
        logging.basicConfig(
            level=logging.INFO,
            format='%(message)s'
        )
    
    trainer = InformationTheoreticFairBPE(
        vocab_size=vocab_size,
        fairness_weight=fairness_weight,
        normalizer=normalizer,
        byte_fallback=byte_fallback,
        verbose=verbose,
    )
    
    return trainer.train(corpora, seed=seed)