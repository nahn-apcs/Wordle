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

from typing import List, Dict
import os
import numpy as np
from scipy.stats import entropy as scipy_entropy

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


def stochastic_optimal_guess(allowed_words: List[str], 
                             possible_words: List[str], 
                             priors: Dict[str, float],
                             top_m: int = 5) -> str:
    """
    Find a good guess by randomly selecting from top M candidates.
    
    Stochastic version of optimal_guess:
    - If only one word remains, guess it
    - Otherwise, calculate entropy for all allowed guesses
    - Randomly select from top M candidates with highest entropy
    
    Args:
        allowed_words: All valid guesses
        possible_words: Remaining possible answers
        priors: Prior probabilities (uniform for this mode)
        top_m: Number of top candidates to randomly select from
        
    Returns:
        str: A randomly selected guess from top M
    """
    import random
    
    if len(possible_words) == 1:
        return possible_words[0]
    
    weights = get_weights(possible_words, priors)
    ents = get_entropies(allowed_words, possible_words, weights)
    
    # Get top M indices
    m = min(top_m, len(allowed_words))
    top_indices = np.argsort(ents)[-m:][::-1]
    
    # Randomly select from top M
    selected_idx = random.choice(top_indices)
    return allowed_words[selected_idx]
