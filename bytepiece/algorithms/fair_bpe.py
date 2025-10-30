"""
Fair BPE: Language-balanced tokenization algorithm.

Reduces language inequality by penalizing merges that increase fertility gaps.

Example usage:
    >>> from fair_bpe import train_fair_bpe
    >>> from bytepiece.algorithms.bpe import BPEEncoder
    >>> from bytepiece.core.io import save_model
    >>> 
    >>> # Prepare multilingual corpus
    >>> corpora = {
    ...     "English": ["hello world", "test"],
    ...     "Spanish": ["hola mundo", "prueba"],
    ...     "German": ["hallo welt", "test"],
    ... }
    >>> 
    >>> # Train Fair BPE
    >>> vocab, merges, normalizer, stats = train_fair_bpe(
    ...     corpora=corpora,
    ...     vocab_size=1000,
    ...     fairness_weight=0.3,
    ...     verbose=True,
    ... )
    >>> 
    >>> # Create encoder for tokenization
    >>> encoder = BPEEncoder(vocab, merges, normalizer)
    >>> 
    >>> # Tokenize text
    >>> tokens = encoder.encode("hello world")
    >>> 
    >>> # Save with fairness metadata
    >>> metadata = {
    ...     "algorithm": "fair_bpe",
    ...     "fairness_weight": 0.3,
    ...     "languages": list(stats.keys()),
    ...     "final_fertilities": {lang: s['fertility'] for lang, s in stats.items()},
    ... }
    >>> save_model(encoder, "fair_bpe.json", metadata=metadata)
"""

import logging
import time
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Set, Tuple

from bytepiece.core.normalizer import Normalizer
from bytepiece.core.vocab import MergeRules, Vocabulary

# Setup logging
logger = logging.getLogger(__name__)


class FairBPE:
    """
    Fair BPE tokenizer with language balancing.
    
    Strategy: Cost-based penalization
    - Penalizes merges that increase fertility gap between languages
    - Balances compression efficiency with fairness
    - Uses incremental caching for O(n) performance
    
    Attributes:
        vocab_size: Target vocabulary size
        fairness_weight: Lambda parameter (0=standard BPE, 1=max fairness)
        normalizer: Text normalizer
        byte_fallback: Enable byte fallback for unknown characters
        verbose: Enable progress logging
    """
    
    def __init__(
        self,
        vocab_size: int,
        fairness_weight: float = 0.3,
        normalizer: Optional[Normalizer] = None,
        byte_fallback: bool = True,
        verbose: bool = False,
    ):
        """Initialize Fair BPE trainer.
        
        Args:
            vocab_size: Target vocabulary size
            fairness_weight: Balance between frequency and fairness (0-1)
                0.0 = Standard BPE (max frequency)
                0.3 = Balanced (recommended)
                1.0 = Max fairness (may hurt compression)
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
        
        # Performance optimization: cache pair counts per language
        self.pair_counts_cache: Dict[Tuple[str, Tuple[str, str]], int] = defaultdict(int)
    
    def train(
        self,
        corpora: Dict[str, List[str]],
        seed: int = 42
    ) -> Tuple[Vocabulary, MergeRules, Normalizer, Dict[str, Dict]]:
        """
        Train Fair BPE on multilingual corpus.
        
        Args:
            corpora: Dictionary mapping language name to list of texts
                Example: {"English": ["hello", "world"], "Spanish": ["hola", "mundo"]}
            seed: Random seed for reproducibility (currently unused, for future extensions)
            
        Returns:
            Tuple of (vocabulary, merge_rules, normalizer, language_stats)
            
            language_stats format:
                {
                    "English": {
                        "total_tokens": 1000,
                        "total_words": 500,
                        "fertility": 2.0,
                    },
                    ...
                }
        
        Raises:
            ValueError: If corpora is empty or languages have no texts
        """
        if not corpora:
            raise ValueError("corpora cannot be empty")
        
        for lang, texts in corpora.items():
            if not texts:
                raise ValueError(f"Language '{lang}' has no texts")
        
        if self.verbose:
            logger.info(f"Training Fair BPE (λ={self.fairness_weight})")
            logger.info(f"  Languages: {list(corpora.keys())}")
            logger.info(f"  Target vocab size: {self.vocab_size}")
        
        # Initialize
        self.vocab = Vocabulary(byte_fallback=self.byte_fallback)
        self.merge_rules = MergeRules()
        
        # Pre-process each language separately
        word_freqs_by_lang = self._preprocess_corpora(corpora)
        
        # Build base vocabulary (all unique characters)
        self._build_base_vocabulary(word_freqs_by_lang)
        
        if self.verbose:
            logger.info(f"  Base vocab size: {len(self.vocab)}")
        
        # Initialize language statistics and pair count cache
        self._initialize_language_stats(word_freqs_by_lang)
        self._initialize_pair_cache(word_freqs_by_lang)
        
        # Iterative merging with fairness consideration
        self._perform_merges(word_freqs_by_lang)
        
        if self.verbose:
            logger.info(f"  Final vocab size: {len(self.vocab)}")
            logger.info(f"  Total merges: {len(self.merge_rules)}")
            self._log_final_stats()
        
        return self.vocab, self.merge_rules, self.normalizer, self.language_stats
    
    def _preprocess_corpora(
        self,
        corpora: Dict[str, List[str]]
    ) -> Dict[str, Counter]:
        """Pre-tokenize and normalize all texts by language.
        
        Returns:
            Dictionary mapping language -> Counter of word tuples
        """
        word_freqs_by_lang = {}
        
        for lang, texts in corpora.items():
            word_freqs = Counter()
            
            for text in texts:
                # Pre-tokenize (splits on whitespace/punctuation)
                chunks = self.normalizer.pre_tokenize(text)
                
                for chunk in chunks:
                    # Normalize (NFKC, lowercase, etc.)
                    normalized = self.normalizer.normalize(chunk)
                    
                    # Convert to character-level tokens
                    tokens = tuple(self.vocab.encode_with_fallback(normalized))
                    
                    word_freqs[tokens] += 1
            
            word_freqs_by_lang[lang] = word_freqs
            
            if self.verbose:
                logger.info(f"  {lang}: {sum(word_freqs.values())} words, "
                          f"{len(word_freqs)} unique")
        
        return word_freqs_by_lang
    
    def _build_base_vocabulary(self, word_freqs_by_lang: Dict[str, Counter]) -> None:
        """Extract all unique characters and add to vocabulary."""
        all_chars: Set[str] = set()
        
        for word_freqs in word_freqs_by_lang.values():
            for word_tuple in word_freqs:
                all_chars.update(word_tuple)
        
        self.vocab.add_tokens(sorted(all_chars))
    
    def _initialize_language_stats(self, word_freqs_by_lang: Dict[str, Counter]) -> None:
        """Initialize fertility statistics per language."""
        for lang, word_freqs in word_freqs_by_lang.items():
            total_tokens = sum(len(word) * freq for word, freq in word_freqs.items())
            total_words = sum(word_freqs.values())
            
            self.language_stats[lang] = {
                'total_tokens': total_tokens,
                'total_words': total_words,
                'fertility': total_tokens / total_words if total_words > 0 else 0.0,
            }
    
    def _initialize_pair_cache(self, word_freqs_by_lang: Dict[str, Counter]) -> None:
        """
        Precompute pair counts per language for O(1) lookup.
        
        Cache structure: {(lang, pair): count}
        This turns _estimate_fairness_penalty from O(n²) to O(1).
        """
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
        """Main training loop: iteratively merge best pairs."""
        base_size = len(self.vocab)
        num_merges = self.vocab_size - base_size
        
        if num_merges <= 0:
            if self.verbose:
                logger.info("  Target vocab size already reached")
            return
        
        start_time = time.perf_counter()
        
        for merge_idx in range(num_merges):
            # Compute pair frequencies (combined across all languages)
            pair_freqs = self._compute_pair_frequencies(word_freqs_by_lang)
            
            if not pair_freqs:
                if self.verbose:
                    logger.info(f"  No more pairs at iteration {merge_idx}")
                break
            
            # Select best pair with fairness consideration
            best_pair = self._select_best_pair(pair_freqs)
            
            # Progress logging
            if self.verbose and (merge_idx % 500 == 0 or merge_idx == num_merges - 1):
                elapsed = time.perf_counter() - start_time
                current_gap = self._compute_fertility_gap()
                logger.info(f"  Merge {merge_idx}/{num_merges} | "
                          f"Gap: {current_gap:.4f} | "
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
        """Compute pair frequencies across all languages (combined)."""
        pair_freqs = Counter()
        
        for word_freqs in word_freqs_by_lang.values():
            for word_tuple, freq in word_freqs.items():
                for i in range(len(word_tuple) - 1):
                    pair = (word_tuple[i], word_tuple[i + 1])
                    pair_freqs[pair] += freq
        
        return pair_freqs
    
    def _select_best_pair(self, pair_freqs: Counter) -> Tuple[str, str]:
        """
        Select best pair considering both frequency and fairness.
        
        Core innovation: score = frequency - λ × fairness_penalty
        
        Args:
            pair_freqs: Counter of pair frequencies across all languages
            
        Returns:
            Best pair to merge
        """
        if self.fairness_weight == 0:
            # Standard BPE: just pick most frequent
            return pair_freqs.most_common(1)[0][0]
        
        # Compute current fertility gap for penalty calculation
        current_gap = self._compute_fertility_gap()
        
        # Score each pair
        best_pair = None
        best_score = float('-inf')
        
        for pair, freq in pair_freqs.items():
            # Base score: frequency
            base_score = float(freq)
            
            # Fairness penalty: estimate impact on gap
            penalty = self._estimate_fairness_penalty(pair, current_gap)
            
            # Final score
            score = base_score - self.fairness_weight * penalty
            
            if score > best_score:
                best_score = score
                best_pair = pair
        
        return best_pair
    
    def _compute_fertility_gap(self) -> float:
        """Compute current fertility gap (max - min)."""
        fertilities = [stats['fertility'] for stats in self.language_stats.values()]
        return max(fertilities) - min(fertilities)
    
    def _estimate_fairness_penalty(
        self,
        pair: Tuple[str, str],
        current_gap: float
    ) -> float:
        """
        Estimate penalty/reward based on gap change.
        
        Positive = penalize (increases gap)
        Negative = reward (decreases gap)
        """
        new_fertilities = {}
        
        for lang in self.language_stats.keys():
            pair_count_in_lang = self.pair_counts_cache.get((lang, pair), 0)
            
            old_tokens = self.language_stats[lang]['total_tokens']
            old_words = self.language_stats[lang]['total_words']
            
            new_tokens = old_tokens - pair_count_in_lang
            new_fertility = new_tokens / old_words if old_words > 0 else 0.0
            
            new_fertilities[lang] = new_fertility
        
        new_gap = max(new_fertilities.values()) - min(new_fertilities.values())
        
        # CAMBIO CLAVE: Sin max(0, ...) para permitir penalties negativos
        gap_change = new_gap - current_gap  # Positivo = malo, Negativo = bueno
        
        total_occurrences = sum(
            self.pair_counts_cache.get((lang, pair), 0)
            for lang in self.language_stats.keys()
        )
        
        penalty = gap_change * total_occurrences
        
        return penalty
    
    def _apply_merge_to_all_languages(
        self,
        word_freqs_by_lang: Dict[str, Counter],
        pair: Tuple[str, str]
    ) -> Dict[str, Counter]:
        """
        Apply merge to all languages and update statistics and cache.
        
        Args:
            word_freqs_by_lang: Current word frequencies by language
            pair: Pair to merge
            
        Returns:
            Updated word frequencies
        """
        new_word_freqs_by_lang = {}
        
        for lang, word_freqs in word_freqs_by_lang.items():
            new_word_freqs = Counter()
            
            # Apply merge to each word
            for word_tuple, freq in word_freqs.items():
                new_word = self._apply_merge(word_tuple, pair)
                new_word_freqs[new_word] += freq
            
            new_word_freqs_by_lang[lang] = new_word_freqs
            
            # Update language stats
            total_tokens = sum(len(word) * freq for word, freq in new_word_freqs.items())
            total_words = sum(new_word_freqs.values())
            
            self.language_stats[lang] = {
                'total_tokens': total_tokens,
                'total_words': total_words,
                'fertility': total_tokens / total_words if total_words > 0 else 0.0,
            }
        
        # Update pair cache incrementally
        self._update_pair_cache(word_freqs_by_lang, new_word_freqs_by_lang, pair)
        
        return new_word_freqs_by_lang
    
    def _update_pair_cache(
        self,
        old_word_freqs: Dict[str, Counter],
        new_word_freqs: Dict[str, Counter],
        merged_pair: Tuple[str, str]
    ) -> None:
        """
        Incrementally update pair cache after merge.
        
        More efficient than rebuilding entire cache.
        """
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
        """
        Apply merge to a single word.
        
        Args:
            word: Word as tuple of tokens
            pair: Pair to merge
            
        Returns:
            Word with pair merged
        """
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
        logger.info("  Final language statistics:")
        for lang, stats in sorted(self.language_stats.items()):
            logger.info(f"    {lang}: fertility={stats['fertility']:.4f}")
        
        final_gap = self._compute_fertility_gap()
        logger.info(f"  Final fertility gap: {final_gap:.4f}")


def train_fair_bpe(
    corpora: Dict[str, List[str]],
    vocab_size: int,
    fairness_weight: float = 0.3,
    normalizer: Optional[Normalizer] = None,
    byte_fallback: bool = True,
    verbose: bool = False,
    seed: int = 42,
) -> Tuple[Vocabulary, MergeRules, Normalizer, Dict[str, Dict]]:
    """
    Convenience function to train Fair BPE.
    
    This is a language-balanced variant of BPE that reduces tokenization
    inequality across languages by penalizing merges that increase
    fertility gaps.
    
    Args:
        corpora: Dictionary mapping language name to list of texts
            Example: {"English": ["hello world"], "Spanish": ["hola mundo"]}
        vocab_size: Target vocabulary size
        fairness_weight: Balance between frequency and fairness (0-1)
            0.0 = Standard BPE (pure frequency-based)
            0.3 = Balanced (recommended default)
            1.0 = Maximum fairness (may hurt compression)
        normalizer: Text normalizer (creates default if None)
        byte_fallback: Enable byte-fallback for full coverage
        verbose: Print training progress
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (vocabulary, merge_rules, normalizer, language_stats)
        
        To use the trained tokenizer:
        >>> from bytepiece.algorithms.bpe import BPEEncoder
        >>> encoder = BPEEncoder(vocab, merge_rules, normalizer)
        >>> tokens = encoder.encode("your text here")
        
    Example:
        >>> corpora = {
        ...     "English": ["hello", "world"],
        ...     "Spanish": ["hola", "mundo"],
        ... }
        >>> vocab, merges, norm, stats = train_fair_bpe(
        ...     corpora=corpora,
        ...     vocab_size=1000,
        ...     fairness_weight=0.3,
        ... )
        >>> print(stats["English"]["fertility"])
        >>> print(stats["Spanish"]["fertility"])
    """
    # Setup logging
    if verbose:
        logging.basicConfig(
            level=logging.INFO,
            format='%(message)s'
        )
    
    fair_bpe = FairBPE(
        vocab_size=vocab_size,
        fairness_weight=fairness_weight,
        normalizer=normalizer,
        byte_fallback=byte_fallback,
        verbose=verbose,
    )
    
    return fair_bpe.train(corpora, seed=seed)