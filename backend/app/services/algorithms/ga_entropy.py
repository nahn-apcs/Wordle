"""
Genetic Algorithm Solver (Entropy Fitness)
=========================================
This solver mirrors the heuristic GA but uses Wordle information gain
(entropy) as the fitness function. The pre-computed pattern matrix makes
entropy evaluation O(1) per guess, enabling population-based search while
remaining fully compliant with Wordle's rules.

Highlights
----------
- First move: random pick from a hand-crafted top-10 entropy list to avoid
  recomputing expensive full-matrix entropies on turn one.
- Fitness: expected information gain over the remaining candidate set.
- Crossover & mutation always return valid words present in the dictionary.
- Pattern matrix is mandatory for entropy evaluation; the solver falls back
  to heuristic GA behaviour if the matrix is unavailable.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Set

import numpy as np
from scipy.stats import entropy as scipy_entropy

from .base import WordleSolver

try:
    from .pattern_matrix import (
        ensure_pattern_matrix,
        feedback_to_pattern,
        PATTERN_GRID_DATA,
    )

    HAS_PATTERN_MATRIX = True
except ImportError:  # pragma: no cover
    HAS_PATTERN_MATRIX = False


FIRST_GUESS_TOP10 = [
    "TARES",
    "LARES",
    "RALES",
    "RATES",
    "NARES",
    "TALES",
    "REAIS",
    "TEARS",
    "ARLES",
    "SALET",
]


class GAEntropySolver(WordleSolver):
    """Genetic Algorithm solver that maximises entropy/information gain."""

    def __init__(
        self,
        word_list: Optional[List[str]] = None,
        word_length: int = 5,
        population_size: int = 60,
        generations: int = 35,
        tournament_size: int = 3,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.25,
        elitism_count: int = 2,
    ) -> None:
        super().__init__(word_list, word_length)

        self.use_pattern_matrix = HAS_PATTERN_MATRIX
        if not self.use_pattern_matrix:
            raise RuntimeError("GAEntropySolver requires the pattern matrix module")

        ensure_pattern_matrix(self.word_list)
        if not PATTERN_GRID_DATA:
            raise RuntimeError("Pattern matrix not loaded; run ensure_pattern_matrix first")

        matrix_words = [w for w in PATTERN_GRID_DATA["words"]]
        self.word_list = matrix_words
        self.word_to_idx: Dict[str, int] = {w: i for i, w in enumerate(matrix_words)}
        self.valid_words_set: Set[str] = set(matrix_words)
        self._build_char_position_lookup()

        self.guessed: Set[str] = set()
        self.guessed_mask = np.zeros(len(self.word_list), dtype=bool)
        self.candidate_indices = np.arange(len(self.word_list), dtype=np.int32)

        self.population_size = population_size
        self.generations = generations
        self.tournament_size = tournament_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism_count = elitism_count

    # ------------------------------------------------------------------
    # Builders
    # ------------------------------------------------------------------
    def _build_char_position_lookup(self) -> None:
        self.char_at_pos: List[Dict[str, Set[int]]] = [{} for _ in range(5)]
        for idx, word in enumerate(self.word_list):
            for pos, letter in enumerate(word):
                self.char_at_pos[pos].setdefault(letter, set()).add(idx)

    def reset(self) -> None:
        super().reset()
        self.guessed.clear()
        self.guessed_mask[:] = False
        self.candidate_indices = np.arange(len(self.word_list), dtype=np.int32)
        if PATTERN_GRID_DATA:
            self.remaining_words = [w for w in PATTERN_GRID_DATA["words"]]

    # ------------------------------------------------------------------
    # GA utilities
    # ------------------------------------------------------------------
    def _compute_entropy_batch(self, guess_indices: np.ndarray) -> np.ndarray:
        if not (self.use_pattern_matrix and PATTERN_GRID_DATA):
            return np.zeros(len(guess_indices))

        grid = PATTERN_GRID_DATA["grid"]
        if len(self.candidate_indices) == 0:
            return np.zeros(len(guess_indices))

        patterns = grid[np.ix_(guess_indices, self.candidate_indices)]
        n_candidates = patterns.shape[1]
        n_guesses = patterns.shape[0]

        distributions = np.zeros((n_guesses, 243), dtype=np.float64)
        row_indices = np.repeat(np.arange(n_guesses), n_candidates)
        np.add.at(distributions, (row_indices, patterns.flatten()), 1.0)
        distributions /= n_candidates

        # Replace all-zero rows (possible if n_candidates==0) to avoid NaNs
        zero_rows = np.isclose(distributions.sum(axis=1), 0.0)
        distributions[zero_rows, 0] = 1.0

        return scipy_entropy(distributions, base=2, axis=1)

    def _tournament_select(self, population: np.ndarray, fitness: np.ndarray) -> int:
        size = min(self.tournament_size, len(population))
        contenders = np.random.choice(len(population), size=size, replace=False)
        return int(contenders[np.argmax(fitness[contenders])])

    def _crossover(self, parent1_idx: int, parent2_idx: int) -> Optional[int]:
        word1 = self.word_list[parent1_idx]
        word2 = self.word_list[parent2_idx]
        split = random.randint(1, self.word_length - 1)
        children = [word1[:split] + word2[split:], word2[:split] + word1[split:]]

        valid_children = [
            self.word_to_idx[ch]
            for ch in children
            if ch in self.valid_words_set and ch not in self.guessed
        ]

        if not valid_children:
            return None
        return random.choice(valid_children)

    def _mutate(self, word_idx: int) -> Optional[int]:
        word = self.word_list[word_idx]
        pos = random.randint(0, self.word_length - 1)

        candidates: Optional[Set[int]] = None
        for i in range(self.word_length):
            if i == pos:
                continue
            letter = word[i]
            bucket = self.char_at_pos[i].get(letter)
            if bucket is None:
                return None
            candidates = bucket.copy() if candidates is None else candidates & bucket
            if not candidates:
                return None

        candidates.discard(word_idx)
        for guessed_word in list(self.guessed):
            idx = self.word_to_idx.get(guessed_word)
            if idx is not None:
                candidates.discard(idx)

        if not candidates:
            return None
        return random.choice(tuple(candidates))

    def _initialize_population(self, valid_indices: np.ndarray) -> np.ndarray:
        n_valid = len(valid_indices)
        pop_size = min(self.population_size, n_valid)

        if n_valid <= pop_size:
            return valid_indices.copy()

        entropies = self._compute_entropy_batch(valid_indices)
        n_elite = pop_size // 2
        elite_local = np.argsort(entropies)[-n_elite:]

        remaining = np.setdiff1d(np.arange(n_valid), elite_local, assume_unique=True)
        n_random = pop_size - n_elite
        if len(remaining) > 0:
            random_local = np.random.choice(remaining, size=min(n_random, len(remaining)), replace=False)
        else:
            random_local = np.array([], dtype=int)

        if len(random_local) < n_random:
            filler = np.random.choice(elite_local, size=n_random - len(random_local), replace=True)
            random_local = np.concatenate([random_local, filler])

        combined = np.concatenate([elite_local, random_local])
        return valid_indices[combined]

    # ------------------------------------------------------------------
    # Core GA loop
    # ------------------------------------------------------------------
    def make_prediction(self) -> str:
        S = len(self.candidate_indices)

        if S == 1:
            word = self.word_list[self.candidate_indices[0]]
            self._mark_guessed(word)
            return word

        if S == 0:
            fallback = self._fallback_word()
            self._mark_guessed(fallback)
            return fallback

        if not self.history:
            first_choices = [w for w in FIRST_GUESS_TOP10 if w in self.valid_words_set]
            guess = random.choice(first_choices) if first_choices else self.word_list[0]
            self._mark_guessed(guess)
            return guess

        valid_indices = np.where(~self.guessed_mask)[0]
        if len(valid_indices) == 0:
            fallback = self.word_list[self.candidate_indices[0]]
            self._mark_guessed(fallback)
            return fallback

        if len(valid_indices) <= self.population_size:
            entropies = self._compute_entropy_batch(valid_indices)
            best_idx = valid_indices[int(np.argmax(entropies))]
            best_word = self.word_list[best_idx]
            self._mark_guessed(best_word)
            return best_word

        population = self._initialize_population(valid_indices)

        for _ in range(self.generations):
            fitness = self._compute_entropy_batch(population)
            elite_local = np.argsort(fitness)[-self.elitism_count :]
            new_population = list(population[elite_local])

            while len(new_population) < len(population):
                p1_local = self._tournament_select(population, fitness)
                p2_local = self._tournament_select(population, fitness)
                parent1_idx = population[p1_local]
                parent2_idx = population[p2_local]

                child_idx = None
                if random.random() < self.crossover_rate:
                    child_idx = self._crossover(parent1_idx, parent2_idx)
                if child_idx is None:
                    child_idx = parent1_idx if random.random() < 0.5 else parent2_idx

                if random.random() < self.mutation_rate:
                    mutant = self._mutate(child_idx)
                    if mutant is not None:
                        child_idx = mutant

                child_word = self.word_list[child_idx]
                if child_word in self.guessed:
                    fallback_idx = self._random_valid_index(valid_indices)
                    new_population.append(fallback_idx)
                else:
                    new_population.append(child_idx)

            population = np.array(new_population[: len(population)], dtype=np.int32)

        final_entropy = self._compute_entropy_batch(population)
        best_idx = population[int(np.argmax(final_entropy))]
        best_word = self.word_list[best_idx]
        self._mark_guessed(best_word)
        return best_word

    # ------------------------------------------------------------------
    # Feedback updates
    # ------------------------------------------------------------------
    def update_feedback(self, guess: str, colors: str) -> None:
        guess = guess.upper()
        colors = colors.upper()

        self.history.append(guess)
        self.colors.append(colors)
        self._mark_guessed(guess)

        pattern = feedback_to_pattern(colors)
        w2i = PATTERN_GRID_DATA["words_to_index"]
        if guess in w2i:
            grid = PATTERN_GRID_DATA["grid"]
            guess_idx = w2i[guess]
            patterns = grid[guess_idx, self.candidate_indices]
            mask = patterns == pattern
            self.candidate_indices = self.candidate_indices[mask]

        self.remaining_words = [self.word_list[i] for i in self.candidate_indices]

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    def _mark_guessed(self, word: str) -> None:
        self.guessed.add(word)
        idx = self.word_to_idx.get(word)
        if idx is not None:
            self.guessed_mask[idx] = True

    def _fallback_word(self) -> str:
        for word in self.word_list:
            if word not in self.guessed:
                return word
        return self.word_list[0]

    def _random_valid_index(self, valid_indices: np.ndarray) -> int:
        unguessed = [idx for idx in valid_indices if not self.guessed_mask[idx]]
        if not unguessed:
            return int(valid_indices[0])
        return int(random.choice(unguessed))

    def get_remaining_count(self) -> int:
        return len(self.candidate_indices)
