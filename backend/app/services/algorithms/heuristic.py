"""
Heuristic Functions for Wordle Solvers
======================================
Implements the splitting heuristic for evaluating guess quality.

h(g) = Σ p(x)(S-p(x)) + Σ p_i(g_i)(S-p_i(g_i)) - n*S²
       letter-based      position-based           penalty

Where:
- S = number of remaining candidate words
- p(x) = number of candidates containing letter x
- p_i(g_i) = number of candidates with letter g_i at position i
- n = number of "dead letters" (letters not in any candidate)
"""

import numpy as np
from typing import List, Tuple, Dict, Optional


def build_letter_stats(candidates: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Build letter frequency statistics for candidates.
    
    Args:
        candidates: Array of encoded words (N x 5), each letter as 0-25
        
    Returns:
        presence: Array of size 26, presence[x] = count of candidates containing letter x
        pos_counts: List of 5 arrays, pos_counts[i][x] = count of candidates with letter x at position i
    """
    if len(candidates) == 0:
        return np.zeros(26, dtype=np.int32), [np.zeros(26, dtype=np.int32) for _ in range(5)]
    
    N = len(candidates)
    
    # Position-based counts: how many candidates have letter x at position i
    pos_counts = []
    for i in range(5):
        counts = np.bincount(candidates[:, i], minlength=26).astype(np.int32)
        pos_counts.append(counts)
    
    # Presence counts: how many candidates contain letter x (anywhere)
    # Vectorized: create binary mask for each letter presence, then sum
    # Shape: (N, 26) - presence_matrix[i, j] = 1 if word i contains letter j
    presence_matrix = np.zeros((N, 26), dtype=np.int32)
    for i in range(5):
        np.add.at(presence_matrix, (np.arange(N), candidates[:, i]), 1)
    
    # Convert to binary (each letter counted once per word)
    presence_matrix = (presence_matrix > 0).astype(np.int32)
    presence = presence_matrix.sum(axis=0)
    
    return presence, pos_counts


def compute_heuristic(
    guess_encoded: np.ndarray,
    presence: np.ndarray,
    pos_counts: List[np.ndarray],
    S: int,
    wpos: float = 0.5
) -> float:
    """
    Compute heuristic score for a single guess.
    
    h(g) = Σ p(x)(S-p(x)) + wpos * Σ p_i(g_i)(S-p_i(g_i)) - n*S²
    
    Args:
        guess_encoded: Encoded guess word (5 letters as 0-25)
        presence: Letter presence counts
        pos_counts: Position-based letter counts
        S: Number of remaining candidates
        wpos: Weight for position-based term (default 0.5)
        
    Returns:
        Heuristic score (higher is better)
    """
    if S == 0:
        return -float('inf')
    
    score = 0.0
    bad_penalty = S * S
    
    # Letter-based splitting: Σ p(x)(S-p(x))
    # Only count unique letters in guess
    seen = set()
    for x in guess_encoded:
        if x not in seen:
            seen.add(x)
            p = presence[x]
            if p == 0:
                score -= bad_penalty  # Dead letter penalty
            else:
                score += p * (S - p)
    
    # Position-based splitting: Σ p_i(g_i)(S-p_i(g_i))
    for i in range(5):
        x = guess_encoded[i]
        p = pos_counts[i][x]
        score += wpos * p * (S - p)
    
    return score


def compute_heuristic_batch(
    guesses_encoded: np.ndarray,
    presence: np.ndarray,
    pos_counts: List[np.ndarray],
    S: int,
    wpos: float = 0.5
) -> np.ndarray:
    """
    Compute heuristic scores for multiple guesses (vectorized).
    
    Args:
        guesses_encoded: Array of encoded guesses (M x 5)
        presence: Letter presence counts
        pos_counts: Position-based letter counts
        S: Number of remaining candidates
        wpos: Weight for position-based term
        
    Returns:
        Array of scores (M,)
    """
    if S == 0:
        return np.full(len(guesses_encoded), -float('inf'))
    
    M = len(guesses_encoded)
    scores = np.zeros(M, dtype=np.float64)
    bad_penalty = S * S
    
    # Position-based term (fully vectorized)
    for i in range(5):
        letter_at_pos = guesses_encoded[:, i]  # (M,)
        p = pos_counts[i][letter_at_pos]  # (M,)
        scores += wpos * p * (S - p)
    
    # Letter-based term (vectorized)
    # For each guess, we need unique letters. 
    # Approach: compute letter contribution for all 5 positions, 
    # but only count each unique letter once.
    # 
    # Create a mask for "first occurrence" of each letter in each word
    # E.g., "HELLO" -> positions [0,1,2,3,4], letters [H,E,L,L,O]
    # First occurrence mask: [True, True, True, False, True]
    
    # Get presence values for all letters in guesses
    all_letters = guesses_encoded.flatten()  # (M*5,)
    all_presence = presence[all_letters].reshape(M, 5)  # (M, 5)
    
    # Create first-occurrence mask using numpy
    # For each row, mark first occurrence of each letter as True
    sorted_indices = np.argsort(guesses_encoded, axis=1)
    sorted_letters = np.take_along_axis(guesses_encoded, sorted_indices, axis=1)
    
    # Detect where sorted letter differs from previous (first occurrence in sorted order)
    first_in_sorted = np.ones((M, 5), dtype=bool)
    first_in_sorted[:, 1:] = sorted_letters[:, 1:] != sorted_letters[:, :-1]
    
    # Map back to original positions
    first_occurrence = np.zeros((M, 5), dtype=bool)
    np.put_along_axis(first_occurrence, sorted_indices, first_in_sorted, axis=1)
    
    # Calculate letter contribution: p * (S - p) for present letters, -bad_penalty for dead
    letter_contrib = np.where(
        all_presence > 0,
        all_presence * (S - all_presence),
        -bad_penalty
    )
    
    # Sum only first occurrences
    scores += np.sum(letter_contrib * first_occurrence, axis=1)
    
    return scores


def encode_words(words: List[str]) -> np.ndarray:
    """
    Encode words as numpy array of letter indices (0-25).
    
    Args:
        words: List of 5-letter words (uppercase)
        
    Returns:
        Array of shape (N, 5) with letter indices
    """
    if not words:
        return np.empty((0, 5), dtype=np.uint8)
    
    encoded = np.array([[ord(c) - ord('A') for c in w.upper()] for w in words], dtype=np.uint8)
    return encoded


def get_unique_letters(word_encoded: np.ndarray) -> np.ndarray:
    """Get unique letters in an encoded word."""
    return np.unique(word_encoded)
