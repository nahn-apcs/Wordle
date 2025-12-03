"""
Genetic Algorithm Solver (Heuristic Fitness)
===========================================
Uses a classical Genetic Algorithm to search the global word list while
evaluating every candidate with the splitting heuristic. The GA explores
the whole space of allowable guesses (not only the remaining answers),
but every fitness evaluation is computed using the current remaining
candidate set so that the heuristic rewards guesses that maximally split
the possible answers.

Key characteristics
-------------------
- Population initialisation: half top-scoring (by heuristic), half random.
- Selection: tournament selection (size configurable).
- Crossover: single split point; child kept only if it is a valid word.
- Mutation: change exactly one position; only valid words are kept.
- Elitism: best individuals automatically copied to the next generation.
- Backend: numpy for fast vectorised heuristic computation plus the
  pre-computed pattern matrix (when available) for O(1) feedback updates.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Set

import numpy as np

from .base import WordleSolver
from .heuristic import (
	build_letter_stats,
	compute_heuristic_batch,
	encode_words,
)

try:  # Pattern matrix accelerates feedback filtering dramatically
	from .pattern_matrix import (
		ensure_pattern_matrix,
		feedback_to_pattern,
		PATTERN_GRID_DATA,
	)

	HAS_PATTERN_MATRIX = True
except ImportError:  # pragma: no cover - environment without pattern matrix
	HAS_PATTERN_MATRIX = False


class GASolver(WordleSolver):
	"""Genetic Algorithm solver that uses the splitting heuristic as fitness."""

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

		# Encoded representations for numpy friendly computations
		self.all_words_encoded = encode_words(self.word_list)
		self.word_to_idx: Dict[str, int] = {w: i for i, w in enumerate(self.word_list)}
		self.valid_words_set: Set[str] = set(self.word_list)
		self._build_char_position_lookup()

		# State tracking
		self.guessed: Set[str] = set()
		self.guessed_mask = np.zeros(len(self.word_list), dtype=bool)
		self.candidate_indices = np.arange(len(self.word_list), dtype=np.int32)

		# GA parameters
		self.population_size = population_size
		self.generations = generations
		self.tournament_size = tournament_size
		self.crossover_rate = crossover_rate
		self.mutation_rate = mutation_rate
		self.elitism_count = elitism_count

		if self.use_pattern_matrix:
			try:
				ensure_pattern_matrix(self.word_list)
				if PATTERN_GRID_DATA:
					matrix_words = [w for w in PATTERN_GRID_DATA["words"]]
					self.word_list = matrix_words
					self.all_words_encoded = encode_words(matrix_words)
					self.word_to_idx = {w: i for i, w in enumerate(matrix_words)}
					self.valid_words_set = set(matrix_words)
					self._build_char_position_lookup()
					self.candidate_indices = np.arange(len(matrix_words), dtype=np.int32)
					self.guessed_mask = np.zeros(len(matrix_words), dtype=bool)
					self.remaining_words = matrix_words.copy()
			except Exception as exc:  # pragma: no cover - informative only
				print(f"Warning: Could not load pattern matrix: {exc}")
				self.use_pattern_matrix = False

	# ------------------------------------------------------------------
	# Helper builders
	# ------------------------------------------------------------------
	def _build_char_position_lookup(self) -> None:
		"""Pre-compute indices for words sharing a letter at each position."""

		self.char_at_pos: List[Dict[str, Set[int]]] = [{} for _ in range(5)]
		for idx, word in enumerate(self.word_list):
			for pos, letter in enumerate(word):
				if letter not in self.char_at_pos[pos]:
					self.char_at_pos[pos][letter] = set()
				self.char_at_pos[pos][letter].add(idx)

	def reset(self) -> None:
		super().reset()
		self.guessed.clear()
		self.guessed_mask[:] = False
		self.candidate_indices = np.arange(len(self.word_list), dtype=np.int32)
		if self.use_pattern_matrix and PATTERN_GRID_DATA:
			self.remaining_words = [w for w in PATTERN_GRID_DATA["words"]]

	# ------------------------------------------------------------------
	# Genetic operators
	# ------------------------------------------------------------------
	def _compute_fitness_batch(
		self,
		population_indices: np.ndarray,
		presence: np.ndarray,
		pos_counts: List[np.ndarray],
		S: int,
	) -> np.ndarray:
		search_slice = self.all_words_encoded[population_indices]
		return compute_heuristic_batch(search_slice, presence, pos_counts, S)

	def _tournament_select(self, population: np.ndarray, fitness: np.ndarray) -> int:
		size = min(self.tournament_size, len(population))
		contenders = np.random.choice(len(population), size=size, replace=False)
		winner_local = contenders[np.argmax(fitness[contenders])]
		return int(winner_local)

	def _crossover(self, parent1_idx: int, parent2_idx: int) -> Optional[int]:
		word1 = self.word_list[parent1_idx]
		word2 = self.word_list[parent2_idx]

		split = random.randint(1, self.word_length - 1)
		children = [
			word1[:split] + word2[split:],
			word2[:split] + word1[split:],
		]

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

	def _initialize_population(
		self,
		valid_indices: np.ndarray,
		presence: np.ndarray,
		pos_counts: List[np.ndarray],
		S: int,
	) -> np.ndarray:
		n_valid = len(valid_indices)
		pop_size = min(self.population_size, n_valid)

		if n_valid <= pop_size:
			return valid_indices.copy()

		scores = self._compute_fitness_batch(valid_indices, presence, pos_counts, S)
		n_elite = pop_size // 2
		elite_local = np.argsort(scores)[-n_elite:]

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
			fallback = self._pick_random_word()
			self._mark_guessed(fallback)
			return fallback

		candidates_encoded = self.all_words_encoded[self.candidate_indices]
		presence, pos_counts = build_letter_stats(candidates_encoded)

		valid_indices = np.where(~self.guessed_mask)[0]
		if len(valid_indices) == 0:
			fallback = self.word_list[self.candidate_indices[0]]
			self._mark_guessed(fallback)
			return fallback

		if len(valid_indices) <= self.population_size:
			fitness = self._compute_fitness_batch(valid_indices, presence, pos_counts, S)
			best_idx = valid_indices[int(np.argmax(fitness))]
			best_word = self.word_list[best_idx]
			self._mark_guessed(best_word)
			return best_word

		population = self._initialize_population(valid_indices, presence, pos_counts, S)

		for _ in range(self.generations):
			fitness = self._compute_fitness_batch(population, presence, pos_counts, S)
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
					mutated = self._mutate(child_idx)
					if mutated is not None:
						child_idx = mutated

				child_word = self.word_list[child_idx]
				if child_word in self.guessed:
					fallback_idx = self._random_valid_index(valid_indices)
					new_population.append(fallback_idx)
				else:
					new_population.append(child_idx)

			population = np.array(new_population[: len(population)], dtype=np.int32)

		final_fitness = self._compute_fitness_batch(population, presence, pos_counts, S)
		best_idx = population[int(np.argmax(final_fitness))]
		best_word = self.word_list[best_idx]
		self._mark_guessed(best_word)
		return best_word

	# ------------------------------------------------------------------
	# Feedback handling
	# ------------------------------------------------------------------
	def update_feedback(self, guess: str, colors: str) -> None:
		guess = guess.upper()
		colors = colors.upper()

		self.history.append(guess)
		self.colors.append(colors)
		self._mark_guessed(guess)

		if self.use_pattern_matrix and PATTERN_GRID_DATA:
			pattern = feedback_to_pattern(colors)
			w2i = PATTERN_GRID_DATA["words_to_index"]
			if guess in w2i:
				grid = PATTERN_GRID_DATA["grid"]
				guess_idx = w2i[guess]
				patterns = grid[guess_idx, self.candidate_indices]
				mask = patterns == pattern
				self.candidate_indices = self.candidate_indices[mask]
		else:
			mask = [
				idx
				for idx in self.candidate_indices
				if self._matches_feedback(self.word_list[idx], guess, colors)
			]
			self.candidate_indices = np.array(mask, dtype=np.int32)

		self.remaining_words = [self.word_list[i] for i in self.candidate_indices]

	def _matches_feedback(self, candidate: str, guess: str, colors: str) -> bool:
		return self.compute_feedback(guess, candidate) == colors

	# ------------------------------------------------------------------
	# Small helpers
	# ------------------------------------------------------------------
	def _mark_guessed(self, word: str) -> None:
		self.guessed.add(word)
		idx = self.word_to_idx.get(word)
		if idx is not None:
			self.guessed_mask[idx] = True

	def _pick_random_word(self) -> str:
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
