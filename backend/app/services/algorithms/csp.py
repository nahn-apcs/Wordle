"""
CSP (Constraint Satisfaction Problem) / Brute Force Solver for Wordle
======================================================================
Uses numpy + precomputed pattern matrix for fast word filtering.

Based on 3b1b's approach: https://github.com/3b1b/videos/tree/master/_2022/wordle

The approach:
- Pre-compute pattern matrix between all word pairs
- Use numpy for fast filtering of remaining words
- For each guess+feedback, filter to words with matching pattern
"""

from typing import List, Optional
from collections import Counter
import numpy as np
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


class CSPSolver(WordleSolver):
    """
    Brute Force / CSP based Wordle solver.
    
    Uses precomputed pattern matrix for fast O(1) pattern lookups when available.
    Falls back to brute force computation if pattern matrix not available.
    
    Usage:
        solver = CSPSolver()
        
        # First guess
        guess = solver.make_prediction()
        
        # Update with feedback
        solver.update_feedback(guess, "BYBGB")
        
        # Next guess
        next_guess = solver.make_prediction()
    """
    
    def __init__(self, word_list: Optional[List[str]] = None, word_length: int = 5):
        super().__init__(word_list, word_length)
        
        self.use_numpy = HAS_PATTERN_MATRIX
        
        if self.use_numpy:
            try:
                # Ensure pattern matrix is loaded
                ensure_pattern_matrix(self.word_list)
                
                # IMPORTANT: Use words from pattern matrix to ensure correct indexing
                # The pattern matrix indices correspond to PATTERN_GRID_DATA['words']
                matrix_words = PATTERN_GRID_DATA['words']
                self.remaining_words_np = np.array(matrix_words)
                self.remaining_indices = np.arange(len(matrix_words))
                
                # Update word_list to match matrix order
                self.word_list = list(matrix_words)
                self.remaining_words = self.word_list.copy()
            except Exception as e:
                print(f"Warning: Could not load pattern matrix: {e}")
                print("Falling back to brute force mode.")
                self.use_numpy = False
    
    def make_prediction(self) -> str:
        """
        Generate next guess using brute force approach.
        
        Returns the first word that is consistent with all past guesses.
        
        Returns:
            str: The predicted word (uppercase)
        """
        if self.use_numpy:
            # Fast path: use numpy
            if len(self.remaining_indices) == 0:
                raise Exception("No valid predictions available")
            return self.remaining_words_np[self.remaining_indices[0]]
        else:
            # Slow path: brute force
            for word in self.word_list:
                if self.possible_solution(word):
                    return word
            raise Exception("No valid predictions available")
    
    def update_feedback(self, guess: str, colors: str):
        """
        Update solver state with feedback.
        
        Args:
            guess: The word that was guessed
            colors: Feedback string ('G','Y','B')
        """
        guess = guess.upper()
        colors = colors.upper()
        
        # Store in history
        self.history.append(guess)
        self.colors.append(colors)
        
        if self.use_numpy:
            # Fast path: use pattern matrix
            pattern = feedback_to_pattern(colors)
            
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
    
    def score_feedback(self, guess: str, target: str) -> str:
        """
        Return feedback string using:
        'G' = green (correct place)
        'Y' = yellow (in word, wrong place)
        'B' = black/gray (not in word)
        
        This correctly handles repeated letters by counting non-green target letters.
        
        Args:
            guess: The guessed word
            target: The target word (potential answer)
            
        Returns:
            str: Feedback string (e.g., "GYBBB")
        """
        if self.use_numpy and HAS_PATTERN_MATRIX:
            pattern = get_pattern(guess, target)
            return pattern_to_feedback(pattern)
        
        # Fallback: compute manually
        guess = guess.upper()
        target = target.upper()
        n = len(guess)
        feedback = [''] * n

        # First pass: mark greens and count remaining letters in target
        remaining = Counter()
        for i in range(n):
            if guess[i] == target[i]:
                feedback[i] = 'G'
            else:
                remaining[target[i]] += 1

        # Second pass: mark Y or B using remaining counts
        for i in range(n):
            if feedback[i] == 'G':
                continue
            g = guess[i]
            if remaining[g] > 0:
                feedback[i] = 'Y'
                remaining[g] -= 1
            else:
                feedback[i] = 'B'
        
        return ''.join(feedback)

    def possible_solution(self, word: str) -> bool:
        """
        Return True if `word` is consistent with all past guesses and recorded colors.
        
        A word is consistent if: for each past guess, computing the feedback
        (as if this word were the target) gives the same result as the actual
        feedback we received.
        
        Args:
            word: Candidate word to check
            
        Returns:
            bool: True if word is consistent with all history
        """
        word = word.upper()
        
        # Quick length check
        if any(len(word) != len(g) for g in self.history):
            return False

        # Check consistency with each past guess
        for past_guess, color_feedback in zip(self.history, self.colors):
            computed = self.score_feedback(past_guess, word)
            if computed != color_feedback:
                return False
        
        return True


# Alias for backward compatibility
BruteForceSolver = CSPSolver
