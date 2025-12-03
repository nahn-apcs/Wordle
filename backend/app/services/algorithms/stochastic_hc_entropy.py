"""
Stochastic Hill Climbing with Entropy Heuristic
================================================
Uses entropy (information gain) as the heuristic function.
Randomly selects from TOP M candidates (stochastic).

Based on 3b1b's approach with stochastic selection.
"""

from typing import List, Optional
import numpy as np
import random
from .base import WordleSolver

# Import hàm entropy từ entropy.py
from .entropy import (
    get_weights,
    get_entropies,
    HARDCODED_FIRST_GUESS,
)

# Import pattern matrix utilities
from .pattern_matrix import (
    ensure_pattern_matrix,
    feedback_to_pattern,
    PATTERN_GRID_DATA,
)


DEFAULT_TOP_M = 5


class StochasticHCEntropySolver(WordleSolver):
    """
    Stochastic Hill Climbing solver using Entropy as heuristic.
    
    Strategy: Randomly select from TOP M candidates with highest entropy.
    This adds randomness - same state may produce different guesses.
    
    NOTE: Không cache vì mỗi lần gọi cần random khác.
    Chấp nhận chậm hơn HC để đổi lấy tính đa dạng.
    """
    
    def __init__(self, word_list: Optional[List[str]] = None, word_length: int = 5, top_m: int = DEFAULT_TOP_M):
        super().__init__(word_list, word_length)
        
        self.top_m = top_m
        self.priors = {w: 1 for w in self.word_list}
        
        self.guess_history: List[str] = []
        self.pattern_history: List[int] = []
        
        # Load pattern matrix
        ensure_pattern_matrix(self.word_list)
        matrix_words = PATTERN_GRID_DATA['words']
        self.remaining_words_np = np.array(matrix_words)
        self.remaining_indices = np.arange(len(matrix_words))
        self.word_list = list(matrix_words)
        self.remaining_words = self.word_list.copy()
        self.allowed_words = self.word_list.copy()
    
    def make_prediction(self) -> str:
        """Randomly select from TOP M candidates with highest entropy."""
        # First guess: hardcoded (optimal)
        if len(self.guess_history) == 0:
            return HARDCODED_FIRST_GUESS
        
        if len(self.remaining_indices) == 0:
            raise Exception("No valid predictions available")
        
        possible_words = [self.remaining_words_np[i] for i in self.remaining_indices]
        
        if len(possible_words) == 1:
            return possible_words[0]
        
        # Compute entropies mỗi lần (không cache)
        weights = get_weights(possible_words, self.priors)
        ents = get_entropies(self.allowed_words, possible_words, weights)
        
        # Get top M indices
        m = min(self.top_m, len(self.allowed_words))
        top_indices = np.argsort(ents)[-m:][::-1]
        
        # Random select from top M
        selected_idx = random.choice(top_indices)
        return self.allowed_words[selected_idx]
    
    def update_feedback(self, guess: str, colors: str):
        """Update solver state with feedback."""
        guess = guess.upper()
        colors = colors.upper()
        
        self.history.append(guess)
        self.colors.append(colors)
        
        pattern = feedback_to_pattern(colors)
        self.guess_history.append(guess)
        self.pattern_history.append(pattern)
        
        w2i = PATTERN_GRID_DATA['words_to_index']
        grid = PATTERN_GRID_DATA['grid']
        
        if guess in w2i:
            guess_idx = w2i[guess]
            patterns = grid[guess_idx, self.remaining_indices]
            mask = patterns == pattern
            self.remaining_indices = self.remaining_indices[mask]
        
        self.remaining_words = [self.remaining_words_np[i] for i in self.remaining_indices]
    
    def reset(self):
        """Reset solver to initial state."""
        super().reset()
        self.guess_history = []
        self.pattern_history = []
        
        matrix_words = PATTERN_GRID_DATA['words']
        self.remaining_words_np = np.array(matrix_words)
        self.remaining_indices = np.arange(len(matrix_words))
        self.remaining_words = list(matrix_words)
    
    def get_remaining_count(self) -> int:
        return len(self.remaining_indices)
