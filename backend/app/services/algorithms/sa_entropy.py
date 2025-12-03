"""
Simulated Annealing Solver for Wordle (Entropy-based)
=====================================================
Uses Simulated Annealing with entropy (information gain) as energy function.

Based on 3b1b's entropy approach but with SA optimization:
- Energy = entropy of pattern distribution (expected information gain)
- Higher entropy = better guess (more information revealed)
- Boltzmann distribution: P(word) ∝ exp(entropy / T)
"""

import math
import random
import numpy as np
from typing import List, Optional, Set, Dict, Tuple

from .base import WordleSolver
from .entropy import (
    get_weights,
    get_entropies,
)

# Try to import pattern matrix
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


# Pre-computed top 10 first-guess words with their entropy values
# These are the best opening words - no need to compute at runtime
FIRST_GUESS_TOP10: List[Tuple[str, float]] = [
    ("TARES", 6.1594),
    ("LARES", 6.1148),
    ("RALES", 6.0968),
    ("RATES", 6.0841),
    ("RANES", 6.0768),
    ("NARES", 6.0749),
    ("REAIS", 6.0496),
    ("TERAS", 6.0474),
    ("SOARE", 6.0437),
    ("TALES", 6.0142),
]


class SAEntropySolver(WordleSolver):
    """
    Simulated Annealing Wordle solver using entropy as energy.
    
    Uses Boltzmann distribution for weighted random selection:
    - Higher entropy = higher probability of being selected
    - First turn uses pre-computed top 10 words
    
    Usage:
        solver = SAEntropySolver()
        guess = solver.make_prediction()
        solver.update_feedback(guess, "GYBBB")
    """
    
    def __init__(self, 
                 word_list: Optional[List[str]] = None, 
                 word_length: int = 5,
                 max_iterations: int = 500,
                 initial_temperature: float = 2.0,
                 cooling_rate: float = 0.99,
                 top_k: int = 10):
        super().__init__(word_list, word_length)
        
        self.use_pattern_matrix = HAS_PATTERN_MATRIX
        
        self.word_to_idx = {w: i for i, w in enumerate(self.word_list)}
        
        # Track guessed words
        self.guessed: Set[str] = set()
        
        # Uniform priors for entropy calculation
        self.priors: Dict[str, float] = {w: 1.0 for w in self.word_list}
        
        # SA parameters
        self.max_iterations = max_iterations
        self.initial_temperature = initial_temperature
        self.temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.top_k = top_k
        
        # Pre-computed entropy cache for first turn (expensive)
        self._first_turn_entropies: Optional[np.ndarray] = None
        
        if self.use_pattern_matrix:
            try:
                ensure_pattern_matrix(self.word_list)
                if PATTERN_GRID_DATA:
                    matrix_words = list(PATTERN_GRID_DATA['words'])
                    self.word_list = matrix_words
                    self.word_to_idx = {w: i for i, w in enumerate(matrix_words)}
                    self.remaining_words = matrix_words.copy()
                    self.priors = {w: 1.0 for w in matrix_words}
            except Exception as e:
                print(f"Warning: Could not load pattern matrix: {e}")
                self.use_pattern_matrix = False
    
    def reset(self):
        """Reset solver for new game."""
        super().reset()
        self.guessed.clear()
        self.temperature = self.initial_temperature
        if self.use_pattern_matrix and PATTERN_GRID_DATA:
            self.remaining_words = list(PATTERN_GRID_DATA['words'])
    
    def _get_entropy(self, word: str, possible_words: List[str]) -> float:
        """
        Calculate entropy for a single word guess.
        """
        if len(possible_words) == 0:
            return 0.0
        
        weights = get_weights(possible_words, self.priors)
        ents = get_entropies([word], possible_words, weights)
        return ents[0] if len(ents) > 0 else 0.0
    
    def _get_entropies_batch(self, words: List[str], possible_words: List[str]) -> np.ndarray:
        """
        Calculate entropies for multiple words at once (more efficient).
        """
        if len(possible_words) == 0:
            return np.zeros(len(words))
        
        weights = get_weights(possible_words, self.priors)
        return get_entropies(words, possible_words, weights)
    
    def _boltzmann_select(self, words: List[str], energies: np.ndarray, temp: float) -> str:
        """
        Select a word using Boltzmann distribution.
        P(word_i) ∝ exp(energy_i / T)
        """
        # Shift to avoid overflow
        shifted = energies - energies.max()
        weights = np.exp(shifted / temp)
        probs = weights / weights.sum()
        
        selected_idx = np.random.choice(len(words), p=probs)
        return words[selected_idx]
    
    def make_prediction(self) -> str:
        """
        Select guess using TRUE Simulated Annealing.
        
        SA Algorithm:
        1. Start with a random initial solution
        2. Generate neighbor by random swap
        3. If neighbor is better: accept
        4. If neighbor is worse: accept with probability exp(-(E_old - E_new)/T)
           (Note: higher entropy = better, so we want to MAXIMIZE)
        5. Cool down temperature
        6. Repeat until temperature is cold
        
        Returns:
            str: Selected guess word (uppercase)
        """
        # First turn: SA on pre-computed top 10 words
        if len(self.history) == 0:
            words = [w for w, _ in FIRST_GUESS_TOP10]
            energies = np.array([e for _, e in FIRST_GUESS_TOP10])
            
            # SA on top 10
            n = len(words)
            current_idx = random.randint(0, n - 1)
            current_energy = energies[current_idx]
            best_idx = current_idx
            best_energy = current_energy
            
            T = self.initial_temperature
            for _ in range(self.max_iterations):
                neighbor_idx = random.randint(0, n - 1)
                neighbor_energy = energies[neighbor_idx]
                delta_E = neighbor_energy - current_energy
                
                if delta_E > 0 or random.random() < math.exp(delta_E / T):
                    current_idx = neighbor_idx
                    current_energy = neighbor_energy
                    if current_energy > best_energy:
                        best_idx = current_idx
                        best_energy = current_energy
                
                T *= self.cooling_rate
                if T < 0.001:
                    break
            
            first_guess = words[best_idx]
            self.guessed.add(first_guess)
            return first_guess
        
        S = len(self.remaining_words)
        
        # If only one candidate left, return it - SOLVED!
        if S == 1:
            word = self.remaining_words[0]
            self.guessed.add(word)
            return word
        
        # If no candidates, fallback
        if S == 0:
            for w in self.word_list:
                if w not in self.guessed:
                    self.guessed.add(w)
                    return w
            return self.word_list[0]
        
        # Use FULL word list for guessing (not just remaining)
        valid_words = [w for w in self.word_list if w not in self.guessed]
        
        if len(valid_words) == 0:
            return self.remaining_words[0]
        
        n = len(valid_words)
        
        # ============== TRUE SIMULATED ANNEALING ==============
        
        # PRE-COMPUTE all entropies once (expensive but only once per turn)
        all_entropies = self._get_entropies_batch(valid_words, self.remaining_words)
        
        # SELECT TOP-K candidates with highest entropy
        k = min(self.top_k, n)
        top_k_indices = np.argsort(all_entropies)[-k:][::-1]  # Descending
        top_k_words = [valid_words[i] for i in top_k_indices]
        top_k_energies = all_entropies[top_k_indices]
        
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
            # We want to MAXIMIZE entropy, so delta = new - old
            # If delta > 0: neighbor is better
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
        
        best_word = top_k_words[best_k_idx]
        self.guessed.add(best_word)
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
        
        if self.use_pattern_matrix and PATTERN_GRID_DATA:
            pattern = feedback_to_pattern(colors)
            w2i = PATTERN_GRID_DATA['words_to_index']
            grid = PATTERN_GRID_DATA['grid']
            
            if guess in w2i:
                guess_idx = w2i[guess]
                # Get indices of remaining words
                remaining_indices = np.array([w2i[w] for w in self.remaining_words if w in w2i])
                if len(remaining_indices) > 0:
                    patterns = grid[guess_idx, remaining_indices]
                    mask = patterns == pattern
                    # Filter remaining words
                    self.remaining_words = [self.remaining_words[i] for i, m in enumerate(mask) if m]
        else:
            # Slow path: filter manually
            self.remaining_words = [
                w for w in self.remaining_words 
                if self.is_consistent(w, guess, colors)
            ]
    
    def get_remaining_count(self) -> int:
        """Get number of remaining candidates."""
        return len(self.remaining_words)
