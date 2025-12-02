"""
Base Algorithm Class for Wordle Solver
======================================
Provides common interface for all solver algorithms.
Works both locally and via FastAPI.
"""

import os
from abc import ABC, abstractmethod
from typing import List, Optional
from collections import Counter


class WordleSolver(ABC):
    """
    Abstract base class for all Wordle solver algorithms.
    
    Usage:
        # Local usage
        solver = CSPSolver()
        guess = solver.make_prediction()
        solver.update_feedback(guess, "GYBBB")
        
        # FastAPI usage  
        solver = CSPSolver()
        guess = solver.make_prediction()
        # ... send to frontend, get feedback
        solver.update_feedback(guess, colors)
    """
    
    def __init__(self, word_list: Optional[List[str]] = None, word_length: int = 5):
        """
        Initialize solver with word list.
        
        Args:
            word_list: List of valid words. If None, loads from default file.
            word_length: Length of words (default 5)
        """
        self.word_length = word_length
        self.history: List[str] = []  # Past guesses
        self.colors: List[str] = []   # Feedback for each guess ('G','Y','B')
        
        if word_list is not None:
            self.word_list = [w.upper() for w in word_list if len(w) == word_length]
        else:
            self.word_list = self._load_default_words()
        
        self.remaining_words = self.word_list.copy()
    
    def _load_default_words(self) -> List[str]:
        """Load words from default dictionary file."""
        # Try multiple possible paths
        possible_paths = [
            os.path.join(os.path.dirname(__file__), "../../../data/words.txt"),
            os.path.join(os.path.dirname(__file__), "../../../../data/words.txt"),
            "data/words.txt",
            "../data/words.txt",
        ]
        
        for path in possible_paths:
            abs_path = os.path.abspath(path)
            if os.path.exists(abs_path):
                with open(abs_path, 'r') as f:
                    words = [line.strip().upper() for line in f if len(line.strip()) == self.word_length]
                return words
        
        raise FileNotFoundError(f"Could not find words.txt in any of: {possible_paths}")
    
    @abstractmethod
    def make_prediction(self) -> str:
        """
        Generate the next guess based on current state.
        
        Returns:
            str: The predicted word (uppercase)
        """
        pass
    
    def update_feedback(self, guess: str, colors: str):
        """
        Update solver state with feedback from a guess.
        
        Args:
            guess: The word that was guessed (uppercase)
            colors: Feedback string using 'G' (green), 'Y' (yellow), 'B' (black/gray)
        """
        guess = guess.upper()
        colors = colors.upper()
        
        self.history.append(guess)
        self.colors.append(colors)
        
        # Filter remaining words based on new feedback
        self.remaining_words = [
            word for word in self.remaining_words 
            if self.is_consistent(word, guess, colors)
        ]
    
    def reset(self):
        """Reset solver to initial state for a new game."""
        self.history = []
        self.colors = []
        self.remaining_words = self.word_list.copy()
    
    @staticmethod
    def compute_feedback(guess: str, target: str) -> str:
        """
        Compute feedback string for a guess against a target.
        
        Uses standard Wordle rules with proper handling of repeated letters:
        - 'G' = green (correct position)
        - 'Y' = yellow (in word, wrong position)  
        - 'B' = black/gray (not in word or already matched)
        
        Args:
            guess: The guessed word
            target: The actual target word
            
        Returns:
            str: Feedback string (e.g., "GYBBB")
        """
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
    
    @staticmethod
    def is_consistent(word: str, guess: str, colors: str) -> bool:
        """
        Check if a word is consistent with given guess and feedback.
        
        Args:
            word: Candidate word to check
            guess: The guessed word
            colors: Feedback string ('G','Y','B')
            
        Returns:
            bool: True if word could produce this feedback
        """
        return WordleSolver.compute_feedback(guess, word) == colors
    
    def possible_solution(self, word: str) -> bool:
        """
        Check if a word is consistent with ALL past guesses.
        
        Args:
            word: Candidate word to check
            
        Returns:
            bool: True if word is consistent with all history
        """
        if len(word) != self.word_length:
            return False
        
        for past_guess, color_feedback in zip(self.history, self.colors):
            computed = self.compute_feedback(past_guess, word)
            if computed != color_feedback:
                return False
        return True
    
    @property
    def num_remaining(self) -> int:
        """Number of remaining possible words."""
        return len(self.remaining_words)
    
    @property
    def num_guesses(self) -> int:
        """Number of guesses made so far."""
        return len(self.history)
