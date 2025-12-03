"""
Hill Climbing with Entropy Heuristic
=====================================
Uses entropy (information gain) as the heuristic function.
Always selects the word with MAXIMUM entropy (greedy/deterministic).

Based on 3b1b's approach: https://github.com/3b1b/videos/tree/master/_2022/wordle
"""

from typing import List, Optional, Dict
import numpy as np
from .base import WordleSolver

# Import hàm entropy từ entropy.py
from .entropy import (
    optimal_guess,
    get_weights,
    get_entropies,
    pattern_to_int_list,
    HARDCODED_FIRST_GUESS,
)

# Import pattern matrix utilities
from .pattern_matrix import (
    ensure_pattern_matrix,
    feedback_to_pattern,
    PATTERN_GRID_DATA,
)


class HillClimbingEntropySolver(WordleSolver):
    """
    Hill Climbing solver using Entropy as heuristic.
    
    Strategy: Always select the word with MAXIMUM entropy (greedy).
    This is deterministic - same state always produces same guess.
    Uses cache for speed.
    """
    
    def __init__(self, word_list: Optional[List[str]] = None, word_length: int = 5):
        super().__init__(word_list, word_length)
        
        self.priors = {w: 1 for w in self.word_list}
        
        # Cache for decisions (deterministic -> can cache)
        self.next_guess_map: Dict[str, str] = {}
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
    
    def _get_phash(self) -> str:
        """Generate cache key from history."""
        parts = []
        for g, p in zip(self.guess_history, self.pattern_history):
            parts.append(g)
            parts.append("".join(map(str, pattern_to_int_list(p))))
        return "".join(parts)
    
    def make_prediction(self) -> str:
        """Select word with MAXIMUM entropy (Hill Climbing - greedy)."""
        # First guess: hardcoded
        if len(self.guess_history) == 0:
            return HARDCODED_FIRST_GUESS
        
        if len(self.remaining_indices) == 0:
            raise Exception("No valid predictions available")
        
        possible_words = [self.remaining_words_np[i] for i in self.remaining_indices]
        
        if len(possible_words) == 1:
            return possible_words[0]
        
        # Check cache
        phash = self._get_phash()
        if phash in self.next_guess_map:
            return self.next_guess_map[phash]
        
        # Dùng optimal_guess từ entropy.py (argmax entropy)
        guess = optimal_guess(
            self.allowed_words,
            possible_words,
            self.priors,
            purely_maximize_information=True
        )
        
        self.next_guess_map[phash] = guess
        return guess
    
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
        # Keep cache across games
        
        matrix_words = PATTERN_GRID_DATA['words']
        self.remaining_words_np = np.array(matrix_words)
        self.remaining_indices = np.arange(len(matrix_words))
        self.remaining_words = list(matrix_words)
    
    def get_top_candidates(self, n: int = 10) -> List[tuple]:
        """Get top n candidates ranked by entropy."""
        possible_words = [self.remaining_words_np[i] for i in self.remaining_indices]
        weights = get_weights(possible_words, self.priors)
        ents = get_entropies(self.allowed_words, possible_words, weights)
        
        top_indices = np.argsort(ents)[-n:][::-1]
        return [(self.allowed_words[i], float(ents[i])) for i in top_indices]
