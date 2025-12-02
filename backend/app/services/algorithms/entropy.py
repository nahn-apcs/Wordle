"""
Entropy-based Wordle Solver (purely_maximize_information mode)
==============================================================
Based on 3b1b's approach: https://github.com/3b1b/videos/tree/master/_2022/wordle

This solver uses maximum entropy / information gain to choose the best guess.
It purely maximizes the expected information gained from each guess,
without considering word frequency priors.

Key optimizations from 3b1b:
1. Hardcoded first guess (pre-computed optimal: "TARES" or "SALET")
2. next_guess_map cache - reuse results for same game states
3. Pattern matrix for O(1) pattern lookups

The approach:
- For each potential guess, compute the probability distribution over all 243 possible patterns
- Calculate the entropy (expected information) of that distribution
- Choose the guess with maximum entropy
"""

from typing import List, Optional, Dict
import os
import json
import numpy as np
from scipy.stats import entropy as scipy_entropy
from .base import WordleSolver

# Try to import pattern matrix module
try:
    from .pattern_matrix import (
        ensure_pattern_matrix,
        get_pattern,
        filter_words_by_pattern,
        feedback_to_pattern,
        pattern_to_feedback,
        PATTERN_GRID_DATA,
    )
    HAS_PATTERN_MATRIX = True
except ImportError:
    HAS_PATTERN_MATRIX = False


# ============== Constants from 3b1b ==============

# Pre-computed optimal first guess (highest entropy)
# "TARES" or "SALET" are common choices
HARDCODED_FIRST_GUESS = "SALET"

# Data directory for caches
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "data")


# ============== Helper Functions from 3b1b ==============

def pattern_to_int_list(pattern: int) -> List[int]:
    """Convert pattern hash to list of colors (0=grey, 1=yellow, 2=green)."""
    result = []
    curr = pattern
    for _ in range(5):
        result.append(curr % 3)
        curr = curr // 3
    return result


def get_weights(words: List[str], priors: Dict[str, float]) -> np.ndarray:
    """
    Get probability weights for words based on priors.
    
    For purely_maximize_information mode, we use uniform priors (all 1s).
    """
    frequencies = np.array([priors.get(word, 0) for word in words])
    total = frequencies.sum()
    if total == 0:
        return np.zeros(frequencies.shape)
    return frequencies / total


def get_pattern_matrix(allowed_words: List[str], possible_words: List[str]) -> np.ndarray:
    """
    Get the pattern matrix between allowed guesses and possible answers.
    
    Returns an (n_guesses, n_answers) matrix where entry [i,j] is the
    pattern hash when guessing word i against answer j.
    """
    if not PATTERN_GRID_DATA:
        raise RuntimeError("Pattern matrix not loaded. Call ensure_pattern_matrix first.")
    
    grid = PATTERN_GRID_DATA['grid']
    w2i = PATTERN_GRID_DATA['words_to_index']
    
    indices1 = [w2i[w] for w in allowed_words if w in w2i]
    indices2 = [w2i[w] for w in possible_words if w in w2i]
    
    return grid[np.ix_(indices1, indices2)]


def get_pattern_distributions(allowed_words: List[str], possible_words: List[str], weights: np.ndarray) -> np.ndarray:
    """
    For each possible guess in allowed_words, this finds the probability
    distribution across all of the 3^5 wordle patterns you could see, assuming
    the possible answers are in possible_words with associated probabilities
    in weights.

    It considers the pattern hash grid between the two lists of words, and uses
    that to bucket together words from possible_words which would produce
    the same pattern, adding together their corresponding probabilities.
    
    Returns:
        np.ndarray: Shape (n_guesses, 243) probability distributions
    """
    pattern_matrix = get_pattern_matrix(allowed_words, possible_words)
    
    n = len(allowed_words)
    distributions = np.zeros((n, 3**5))
    n_range = np.arange(n)
    for j, prob in enumerate(weights):
        distributions[n_range, pattern_matrix[:, j]] += prob
    return distributions


def entropy_of_distributions(distributions: np.ndarray, atol: float = 1e-12) -> np.ndarray:
    """
    Calculate entropy of probability distributions.
    
    Args:
        distributions: Array of probability distributions
        atol: Absolute tolerance for zero check
        
    Returns:
        np.ndarray: Entropy values in bits (base 2)
    """
    axis = len(distributions.shape) - 1
    return scipy_entropy(distributions, base=2, axis=axis)


def get_entropies(allowed_words: List[str], possible_words: List[str], weights: np.ndarray) -> np.ndarray:
    """
    Calculate expected information gain (entropy) for each guess.
    
    Args:
        allowed_words: List of possible guesses
        possible_words: List of remaining possible answers
        weights: Probability weights for possible_words
        
    Returns:
        np.ndarray: Entropy (expected information) for each guess
    """
    if weights.sum() == 0:
        return np.zeros(len(allowed_words))
    distributions = get_pattern_distributions(allowed_words, possible_words, weights)
    return entropy_of_distributions(distributions)


def get_possible_words(guess: str, pattern: int, word_list: List[str]) -> List[str]:
    """
    Filter word list to only words consistent with guess+pattern.
    """
    all_patterns = get_pattern_matrix([guess], word_list).flatten()
    return list(np.array(word_list)[all_patterns == pattern])


def optimal_guess(allowed_words: List[str], 
                  possible_words: List[str], 
                  priors: Dict[str, float],
                  purely_maximize_information: bool = True) -> str:
    """
    Find the optimal guess using purely_maximize_information strategy.
    
    This is the core of 3b1b's entropy-based approach:
    - If only one word remains, guess it
    - Otherwise, calculate entropy for all allowed guesses
    - Return the guess with maximum entropy (maximum expected information)
    
    Args:
        allowed_words: All valid guesses
        possible_words: Remaining possible answers
        priors: Prior probabilities (uniform for this mode)
        purely_maximize_information: Use max entropy strategy
        
    Returns:
        str: The optimal guess
    """
    if purely_maximize_information:
        if len(possible_words) == 1:
            return possible_words[0]
        
        weights = get_weights(possible_words, priors)
        ents = get_entropies(allowed_words, possible_words, weights)
        return allowed_words[np.argmax(ents)]
    
    # Fallback
    return possible_words[0] if possible_words else allowed_words[0]


# ============== Solver Class ==============

class EntropySolver(WordleSolver):
    """
    Entropy-based Wordle solver using 3b1b's purely_maximize_information approach.
    
    Key optimizations:
    1. Hardcoded first guess - skips expensive entropy calculation for first move
    2. next_guess_map cache - reuses results for identical game states
    3. Pattern matrix for O(1) lookups
    
    Usage:
        solver = EntropySolver()
        
        # First guess (instant - hardcoded)
        guess = solver.make_prediction()
        
        # Update with feedback
        solver.update_feedback(guess, "BYBGB")
        
        # Next guess
        next_guess = solver.make_prediction()
    """
    
    def __init__(self, word_list: Optional[List[str]] = None, word_length: int = 5):
        super().__init__(word_list, word_length)
        
        self.use_numpy = HAS_PATTERN_MATRIX
        
        # Uniform priors for purely_maximize_information mode
        self.priors = {w: 1 for w in self.word_list}
        
        # ============== Key optimization from 3b1b ==============
        # Cache for next guess decisions: phash -> guess
        # phash = "guess1" + "pattern1" + "guess2" + "pattern2" + ...
        self.next_guess_map: Dict[str, str] = {}
        
        # Track guesses and patterns for cache key
        self.guess_history: List[str] = []
        self.pattern_history: List[int] = []
        
        if self.use_numpy:
            try:
                # Ensure pattern matrix is loaded
                ensure_pattern_matrix(self.word_list)
                
                # IMPORTANT: Use words from pattern matrix to ensure correct indexing
                matrix_words = PATTERN_GRID_DATA['words']
                self.remaining_words_np = np.array(matrix_words)
                self.remaining_indices = np.arange(len(matrix_words))
                
                # Update word_list to match matrix order
                self.word_list = list(matrix_words)
                self.remaining_words = self.word_list.copy()
                
                # All words are allowed as guesses
                self.allowed_words = self.word_list.copy()
                
            except Exception as e:
                print(f"Warning: Could not load pattern matrix: {e}")
                print("Falling back to brute force mode.")
                self.use_numpy = False
                self.allowed_words = self.word_list.copy()
        else:
            self.allowed_words = self.word_list.copy()
    
    def _get_phash(self) -> str:
        """
        Generate cache key from guess/pattern history.
        Format from 3b1b: "guess1" + "pattern1_digits" + "guess2" + "pattern2_digits" + ...
        """
        parts = []
        for g, p in zip(self.guess_history, self.pattern_history):
            parts.append(g)
            parts.append("".join(map(str, pattern_to_int_list(p))))
        return "".join(parts)
    
    def make_prediction(self) -> str:
        """
        Generate next guess using purely_maximize_information strategy.
        
        Optimizations:
        1. First guess: return hardcoded optimal (instant)
        2. Check cache for previously computed results
        3. Only compute if cache miss
        
        Returns:
            str: The predicted word (uppercase)
        """
        if not self.use_numpy:
            # Fallback: return first remaining word
            if self.remaining_words:
                return self.remaining_words[0]
            raise Exception("No valid predictions available")
        
        # ============== Optimization 1: Hardcoded first guess ==============
        if len(self.guess_history) == 0:
            return HARDCODED_FIRST_GUESS
        
        # Get current remaining words
        if len(self.remaining_indices) == 0:
            raise Exception("No valid predictions available")
        
        possible_words = [self.remaining_words_np[i] for i in self.remaining_indices]
        
        # If only one word left, return it
        if len(possible_words) == 1:
            return possible_words[0]
        
        # ============== Optimization 2: Check cache ==============
        phash = self._get_phash()
        if phash in self.next_guess_map:
            return self.next_guess_map[phash]
        
        # ============== Cache miss: compute optimal guess ==============
        guess = optimal_guess(
            self.allowed_words,
            possible_words,
            self.priors,
            purely_maximize_information=True
        )
        
        # Store in cache for future use
        self.next_guess_map[phash] = guess
        
        return guess
    
    def update_feedback(self, guess: str, colors: str):
        """
        Update solver state with feedback.
        
        Args:
            guess: The word that was guessed
            colors: Feedback string ('G','Y','B')
        """
        guess = guess.upper()
        colors = colors.upper()
        
        # Store in history (for base class)
        self.history.append(guess)
        self.colors.append(colors)
        
        # Convert colors to pattern hash
        pattern = feedback_to_pattern(colors)
        
        # Store for cache key generation
        self.guess_history.append(guess)
        self.pattern_history.append(pattern)
        
        if self.use_numpy:
            # Fast path: use pattern matrix
            if PATTERN_GRID_DATA:
                w2i = PATTERN_GRID_DATA['words_to_index']
                grid = PATTERN_GRID_DATA['grid']
                
                if guess in w2i:
                    guess_idx = w2i[guess]
                    
                    # Get patterns for remaining words
                    patterns = grid[guess_idx, self.remaining_indices]
                    
                    # Keep only words with matching pattern
                    mask = patterns == pattern
                    self.remaining_indices = self.remaining_indices[mask]
            
            # Update remaining words list for compatibility
            self.remaining_words = [self.remaining_words_np[i] for i in self.remaining_indices]
        else:
            # Slow path: filter manually
            self.remaining_words = [w for w in self.remaining_words if self.possible_solution(w)]
    
    def reset(self):
        """Reset solver to initial state."""
        super().reset()
        
        # Reset cache-related state
        self.guess_history = []
        self.pattern_history = []
        # Note: Keep next_guess_map cache across games (it's still valid!)
        
        if self.use_numpy and PATTERN_GRID_DATA:
            matrix_words = PATTERN_GRID_DATA['words']
            self.remaining_words_np = np.array(matrix_words)
            self.remaining_indices = np.arange(len(matrix_words))
            self.remaining_words = list(matrix_words)
    
    def get_remaining_count(self) -> int:
        """Get number of remaining possible words."""
        if self.use_numpy:
            return len(self.remaining_indices)
        return len(self.remaining_words)
    
    def get_top_candidates(self, n: int = 10) -> List[tuple]:
        """
        Get top n candidates ranked by entropy.
        
        Returns:
            List of (word, entropy) tuples
        """
        if not self.use_numpy:
            return [(w, 0.0) for w in self.remaining_words[:n]]
        
        possible_words = [self.remaining_words_np[i] for i in self.remaining_indices]
        weights = get_weights(possible_words, self.priors)
        ents = get_entropies(self.allowed_words, possible_words, weights)
        
        # Get top n by entropy
        top_indices = np.argsort(ents)[-n:][::-1]
        return [(self.allowed_words[i], float(ents[i])) for i in top_indices]
    
    def score_feedback(self, guess: str, target: str) -> str:
        """
        Return feedback string for a guess against target.
        
        Args:
            guess: The guessed word
            target: The target word
            
        Returns:
            str: Feedback string (e.g., "GYBBB")
        """
        if self.use_numpy and HAS_PATTERN_MATRIX:
            pattern = get_pattern(guess, target)
            return pattern_to_feedback(pattern)
        
        # Fallback: use base class method
        return self.compute_feedback(guess, target)


# ============== Simulation Functions (from 3b1b) ==============

def simulate_game(answer: str, first_guess: str = None, max_guesses: int = 6) -> dict:
    """
    Simulate a single Wordle game using entropy-based strategy.
    
    Args:
        answer: The target word
        first_guess: Optional first guess (uses HARDCODED_FIRST_GUESS if not provided)
        max_guesses: Maximum number of guesses allowed
        
    Returns:
        dict: Game result with score, guesses, patterns
    """
    solver = EntropySolver()
    answer = answer.upper()
    
    guesses = []
    patterns = []
    
    for i in range(max_guesses):
        if i == 0 and first_guess:
            guess = first_guess.upper()
        else:
            guess = solver.make_prediction()
        
        guesses.append(guess)
        
        if guess == answer:
            return {
                'score': i + 1,
                'won': True,
                'answer': answer,
                'guesses': guesses,
                'patterns': patterns,
            }
        
        feedback = solver.score_feedback(guess, answer)
        pattern = feedback_to_pattern(feedback)
        patterns.append(pattern)
        solver.update_feedback(guess, feedback)
    
    return {
        'score': max_guesses + 1,
        'won': False,
        'answer': answer,
        'guesses': guesses,
        'patterns': patterns,
    }


if __name__ == "__main__":
    # Test the solver
    import time
    
    solver = EntropySolver()
    print(f"Loaded {len(solver.word_list)} words")
    print(f"Pattern matrix available: {solver.use_numpy}")
    print(f"Hardcoded first guess: {HARDCODED_FIRST_GUESS}")
    
    # Test first guess (should be instant)
    print("\n--- First guess (hardcoded, should be instant) ---")
    start = time.time()
    guess1 = solver.make_prediction()
    elapsed1 = time.time() - start
    print(f"First guess: {guess1}")
    print(f"Time: {elapsed1:.4f}s")
    
    # Test with a sample word
    test_answer = "CRANE"
    print(f"\n--- Simulating game with answer: {test_answer} ---")
    
    start = time.time()
    result = simulate_game(test_answer)
    elapsed = time.time() - start
    
    print(f"Score: {result['score']}")
    print(f"Won: {result['won']}")
    print(f"Guesses: {result['guesses']}")
    print(f"Time: {elapsed:.2f}s")
    
    # Test cache effectiveness
    print("\n--- Testing cache (same game should be faster) ---")
    start = time.time()
    result2 = simulate_game(test_answer)
    elapsed2 = time.time() - start
    print(f"Second run time: {elapsed2:.2f}s")
