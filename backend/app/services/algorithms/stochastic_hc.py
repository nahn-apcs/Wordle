"""
Stochastic Hill Climbing Solver for Wordle
==========================================
Randomly selects from the top-M guesses with highest heuristic scores.

This adds randomness to avoid getting stuck in local optima and provides
variety in gameplay while still making reasonably good guesses.
"""

import numpy as np
from typing import List, Optional, Set

from .base import WordleSolver
from .heuristic import (
    build_letter_stats,
    compute_heuristic_batch,
    encode_words,
)

# Try to import pattern matrix for fast feedback computation
try:
    from .pattern_matrix import (
        ensure_pattern_matrix,
        get_pattern,
        feedback_to_pattern,
        pattern_to_feedback,
        PATTERN_GRID_DATA,
    )
    HAS_PATTERN_MATRIX = True
except ImportError:
    HAS_PATTERN_MATRIX = False


# Default number of top candidates to randomly choose from
DEFAULT_TOP_M = 10


class StochasticHillClimbingSolver(WordleSolver):
    """
    Stochastic Hill Climbing Wordle solver.
    
    Instead of always picking the best guess, randomly selects from
    the top-M guesses with highest heuristic scores. This provides
    variety while still making good guesses.
    
    Usage:
        solver = StochasticHillClimbingSolver(top_m=10)
        guess = solver.make_prediction()
        solver.update_feedback(guess, "GYBBB")
    """
    
    def __init__(
        self, 
        word_list: Optional[List[str]] = None, 
        word_length: int = 5,
        top_m: int = DEFAULT_TOP_M,
        seed: Optional[int] = None
    ):
        """
        Initialize solver.
        
        Args:
            word_list: List of valid words
            word_length: Length of words (default 5)
            top_m: Number of top candidates to randomly choose from
            seed: Random seed for reproducibility
        """
        super().__init__(word_list, word_length)
        
        self.top_m = top_m
        self.rng = np.random.default_rng(seed)
        
        self.use_pattern_matrix = HAS_PATTERN_MATRIX
        
        # Encode all words for fast heuristic computation
        self.all_words_encoded = encode_words(self.word_list)
        self.word_to_idx = {w: i for i, w in enumerate(self.word_list)}
        
        # Track guessed words to avoid repeating (as index mask)
        self.guessed: Set[str] = set()
        self.guessed_mask = np.zeros(len(self.word_list), dtype=bool)
        
        # Remaining candidates (indices into word_list)
        self.candidate_indices = np.arange(len(self.word_list))
        
        if self.use_pattern_matrix:
            try:
                ensure_pattern_matrix(self.word_list)
                # Use words from pattern matrix for consistency
                if PATTERN_GRID_DATA:
                    matrix_words = list(PATTERN_GRID_DATA['words'])
                    self.word_list = matrix_words
                    self.all_words_encoded = encode_words(matrix_words)
                    self.word_to_idx = {w: i for i, w in enumerate(matrix_words)}
                    self.candidate_indices = np.arange(len(matrix_words))
                    self.guessed_mask = np.zeros(len(matrix_words), dtype=bool)
                    self.remaining_words = matrix_words.copy()
            except Exception as e:
                print(f"Warning: Could not load pattern matrix: {e}")
                self.use_pattern_matrix = False
    
    def reset(self):
        """Reset solver for new game."""
        super().reset()
        self.guessed.clear()
        self.guessed_mask[:] = False
        self.candidate_indices = np.arange(len(self.word_list))
        if self.use_pattern_matrix and PATTERN_GRID_DATA:
            self.remaining_words = list(PATTERN_GRID_DATA['words'])
    
    def make_prediction(self) -> str:
        """
        Randomly select from top-M guesses with highest heuristic scores.
        
        Returns:
            str: Selected guess word (uppercase)
        """
        S = len(self.candidate_indices)
        
        # If only one candidate left, return it
        if S == 1:
            word = self.word_list[self.candidate_indices[0]]
            self.guessed.add(word)
            return word
        
        # If no candidates (shouldn't happen), return first unguessed word
        if S == 0:
            for w in self.word_list:
                if w not in self.guessed:
                    self.guessed.add(w)
                    return w
            return self.word_list[0]
        
        # Build statistics from current candidates
        candidates_encoded = self.all_words_encoded[self.candidate_indices]
        presence, pos_counts = build_letter_stats(candidates_encoded)
        
        # Find valid guesses (not yet guessed) - O(n) numpy operation
        valid_indices = np.where(~self.guessed_mask)[0]
        
        if len(valid_indices) == 0:
            word = self.word_list[self.candidate_indices[0]]
            return word
        
        # Compute heuristic for all valid guesses
        valid_guesses_encoded = self.all_words_encoded[valid_indices]
        scores = compute_heuristic_batch(valid_guesses_encoded, presence, pos_counts, S)
        
        # Select from top-M (or fewer if not enough valid guesses)
        # M = min(top_m, number of valid guesses)
        M = min(self.top_m, len(valid_indices))
        
        
        # Get indices of top-M scores
        top_m_local_indices = np.argpartition(scores, -M)[-M:]
        
        # Randomly select one from top-M
        selected_local_idx = self.rng.choice(top_m_local_indices)
        selected_idx = valid_indices[selected_local_idx]
        selected_word = self.word_list[selected_idx]
        
        self.guessed.add(selected_word)
        self.guessed_mask[selected_idx] = True
        return selected_word
    
    def update_feedback(self, guess: str, colors: str):
        """
        Update solver state with feedback.
        
        Args:
            guess: The guessed word
            colors: Feedback string ('G', 'Y', 'B')
        """
        guess = guess.upper()
        colors = colors.upper()
        
        self.history.append(guess)
        self.colors.append(colors)
        self.guessed.add(guess)
        if guess in self.word_to_idx:
            self.guessed_mask[self.word_to_idx[guess]] = True
        
        if self.use_pattern_matrix and PATTERN_GRID_DATA:
            # Fast path: use pattern matrix
            pattern = feedback_to_pattern(colors)
            w2i = PATTERN_GRID_DATA['words_to_index']
            grid = PATTERN_GRID_DATA['grid']
            
            if guess in w2i:
                guess_idx = w2i[guess]
                patterns = grid[guess_idx, self.candidate_indices]
                mask = patterns == pattern
                self.candidate_indices = self.candidate_indices[mask]
        else:
            # Slow path: filter manually
            new_candidates = []
            for idx in self.candidate_indices:
                word = self.word_list[idx]
                if self._matches_feedback(word, guess, colors):
                    new_candidates.append(idx)
            self.candidate_indices = np.array(new_candidates)
        
        # Update remaining_words for compatibility
        self.remaining_words = [self.word_list[i] for i in self.candidate_indices]
    
    def _matches_feedback(self, candidate: str, guess: str, colors: str) -> bool:
        """Check if candidate matches the feedback for guess."""
        expected_colors = self.score_feedback(guess, candidate)
        return expected_colors == colors
    
    def score_feedback(self, guess: str, target: str) -> str:
        """Compute feedback string for guess against target."""
        if self.use_pattern_matrix and HAS_PATTERN_MATRIX:
            pattern = get_pattern(guess, target)
            return pattern_to_feedback(pattern)
        
        # Fallback: compute manually
        guess = guess.upper()
        target = target.upper()
        feedback = ['B'] * 5
        target_chars = list(target)
        
        # First pass: mark greens
        for i in range(5):
            if guess[i] == target_chars[i]:
                feedback[i] = 'G'
                target_chars[i] = None
        
        # Second pass: mark yellows
        for i in range(5):
            if feedback[i] == 'G':
                continue
            if guess[i] in target_chars:
                feedback[i] = 'Y'
                idx = target_chars.index(guess[i])
                target_chars[idx] = None
        
        return ''.join(feedback)
    
    def get_remaining_count(self) -> int:
        """Get number of remaining candidates."""
        return len(self.candidate_indices)
