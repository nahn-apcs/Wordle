"""
Simulated Annealing Solver for Wordle (Heuristic-based)
=======================================================
Uses Simulated Annealing with splitting heuristic as energy function.

The energy function measures how well a guess splits the remaining candidates:
h(g) = Σ p(x)(S-p(x)) + Σ p_i(g_i)(S-p_i(g_i)) - n*S²

Higher energy = better guess (more information gain)
SA allows accepting worse guesses with probability exp(delta/T)
"""

import math
import random
import numpy as np
from typing import List, Optional, Set

from .base import WordleSolver
from .heuristic import (
    build_letter_stats,
    compute_heuristic,
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


class SASolver(WordleSolver):
    """
    Simulated Annealing Wordle solver using splitting heuristic.
    
    Uses Metropolis criterion to occasionally accept worse guesses,
    allowing exploration of the search space to potentially find
    better global solutions.
    
    Usage:
        solver = SASolver()
        guess = solver.make_prediction()
        solver.update_feedback(guess, "GYBBB")
    """
    
    def __init__(self, 
                 word_list: Optional[List[str]] = None, 
                 word_length: int = 5,
                 max_iterations: int = 500,
                 initial_temperature: float = 50_000_000.0,  # Scale with heuristic (80M-280M range)
                 cooling_rate: float = 0.99,
                 top_k: int = 10):
        super().__init__(word_list, word_length)
        
        self.use_pattern_matrix = HAS_PATTERN_MATRIX
        
        # Encode all words for fast heuristic computation
        self.all_words_encoded = encode_words(self.word_list)
        self.word_to_idx = {w: i for i, w in enumerate(self.word_list)}
        
        # Track guessed words to avoid repeating
        self.guessed: Set[str] = set()
        self.guessed_mask = np.zeros(len(self.word_list), dtype=bool)
        
        # Remaining candidates (indices into word_list)
        self.candidate_indices = np.arange(len(self.word_list))
        
        # SA parameters
        self.max_iterations = max_iterations
        self.initial_temperature = initial_temperature
        self.temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.top_k = top_k
        
        if self.use_pattern_matrix:
            try:
                ensure_pattern_matrix(self.word_list)
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
        self.temperature = self.initial_temperature
        if self.use_pattern_matrix and PATTERN_GRID_DATA:
            self.remaining_words = list(PATTERN_GRID_DATA['words'])
    
    def _get_energy(self, word_idx: int, presence: np.ndarray, 
                    pos_counts: List[np.ndarray], S: int) -> float:
        """
        Calculate energy (heuristic score) for a word.
        Higher energy = better guess.
        """
        word_encoded = self.all_words_encoded[word_idx]
        return compute_heuristic(word_encoded, presence, pos_counts, S)
    
    def make_prediction(self) -> str:
        """
        Select guess using TRUE Simulated Annealing with heuristic as energy.
        
        SA Algorithm:
        1. Start with a random initial solution
        2. Generate neighbor by random swap
        3. If neighbor is better: accept
        4. If neighbor is worse: accept with probability exp(-(E_old - E_new)/T)
           (Note: higher heuristic = better, so we want to MAXIMIZE)
        5. Cool down temperature
        6. Repeat until temperature is cold
        
        Returns:
            str: Selected guess word (uppercase)
        """
        S = len(self.candidate_indices)
        
        # If only one candidate left, return it - SOLVED!
        if S == 1:
            word = self.word_list[self.candidate_indices[0]]
            self.guessed.add(word)
            return word
        
        # If no candidates, return first unguessed word
        if S == 0:
            for w in self.word_list:
                if w not in self.guessed:
                    self.guessed.add(w)
                    return w
            return self.word_list[0]
        
        # Build statistics from current candidates
        candidates_encoded = self.all_words_encoded[self.candidate_indices]
        presence, pos_counts = build_letter_stats(candidates_encoded)
        
        # Get valid indices (not yet guessed) - ALWAYS use full word list
        valid_indices = np.where(~self.guessed_mask)[0]
        
        if len(valid_indices) == 0:
            word = self.word_list[self.candidate_indices[0]]
            return word
        
        n = len(valid_indices)
        
        # ============== TRUE SIMULATED ANNEALING ==============
        
        # PRE-COMPUTE all heuristics once
        search_encoded = self.all_words_encoded[valid_indices]
        all_energies = compute_heuristic_batch(search_encoded, presence, pos_counts, S)
        
        # SELECT TOP-K candidates with highest heuristic
        k = min(self.top_k, n)
        top_k_local_indices = np.argsort(all_energies)[-k:][::-1]  # Descending
        top_k_energies = all_energies[top_k_local_indices]
        
        # SA only on top-k candidates
        # Initialize: random starting solution from top-k
        current_k_idx = random.randint(0, k - 1)
        current_energy = top_k_energies[current_k_idx]
        
        # Track best solution found
        best_k_idx = current_k_idx
        best_energy = current_energy
        
        # SA parameters
        T = self.initial_temperature
        
        for iteration in range(self.max_iterations):
            # Generate neighbor: random word from TOP-K only
            neighbor_k_idx = random.randint(0, k - 1)
            neighbor_energy = top_k_energies[neighbor_k_idx]  # O(1) lookup!
            
            # Calculate energy difference
            # We want to MAXIMIZE heuristic, so delta = new - old
            delta_E = neighbor_energy - current_energy
            
            # Acceptance decision
            if delta_E > 0:
                # Neighbor is better - always accept
                accept = True
            else:
                # Neighbor is worse - accept with probability exp(delta_E / T)
                accept_prob = math.exp(delta_E / T) if T > 0 else 0
                accept = random.random() < accept_prob
            
            if accept:
                current_k_idx = neighbor_k_idx
                current_energy = neighbor_energy
                
                # Update best if this is the best we've seen
                if current_energy > best_energy:
                    best_k_idx = current_k_idx
                    best_energy = current_energy
            
            # Cool down
            T *= self.cooling_rate
            
            # Early termination if temperature is very cold
            if T < 0.001:
                break
        
        # Map back to original index
        best_local_idx = top_k_local_indices[best_k_idx]
        best_idx = valid_indices[best_local_idx]
        best_word = self.word_list[best_idx]
        self.guessed.add(best_word)
        self.guessed_mask[best_idx] = True
        return best_word
    
    def update_feedback(self, guess: str, colors: str):
        """
        Update solver state with feedback.
        """
        guess = guess.upper()
        colors = colors.upper()
        
        self.history.append(guess)
        self.colors.append(colors)
        self.guessed.add(guess)
        if guess in self.word_to_idx:
            self.guessed_mask[self.word_to_idx[guess]] = True
        
        if self.use_pattern_matrix and PATTERN_GRID_DATA:
            pattern = feedback_to_pattern(colors)
            w2i = PATTERN_GRID_DATA['words_to_index']
            grid = PATTERN_GRID_DATA['grid']
            
            if guess in w2i:
                guess_idx = w2i[guess]
                patterns = grid[guess_idx, self.candidate_indices]
                mask = patterns == pattern
                self.candidate_indices = self.candidate_indices[mask]
        else:
            new_candidates = []
            for idx in self.candidate_indices:
                word = self.word_list[idx]
                if self._matches_feedback(word, guess, colors):
                    new_candidates.append(idx)
            self.candidate_indices = np.array(new_candidates)
        
        self.remaining_words = [self.word_list[i] for i in self.candidate_indices]
    
    def _matches_feedback(self, candidate: str, guess: str, colors: str) -> bool:
        """Check if candidate matches the feedback for guess."""
        expected_colors = self.compute_feedback(guess, candidate)
        return expected_colors == colors
    
    def get_remaining_count(self) -> int:
        """Get number of remaining candidates."""
        return len(self.candidate_indices)
