#!/usr/bin/env python3
"""
Wordle Benchmark Runner
=======================
Run experiments on Wordle solver algorithms locally.

Usage:
    python run_benchmark.py --algorithms entropy,csp --start 0 --end 100
    python run_benchmark.py --algorithms all --start 0 --end 2315
    python run_benchmark.py --algorithms hill_climb --start 500 --end 600
"""

import argparse
import json
import time
import os
import sys
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, asdict
from enum import Enum

try:
    from tqdm import tqdm
except ImportError:
    print("⚠️  tqdm not installed. Run: pip install tqdm")
    sys.exit(1)

# Import algorithms from app/services/algorithms/
# Add backend to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.services.algorithms import CSPSolver, HillClimbingSolver, StochasticHillClimbingSolver


# ============== WORDLE GAME SIMULATION ==============

class LetterState(Enum):
    CORRECT = "🟩"   # Green - right letter, right position
    PRESENT = "🟨"   # Yellow - right letter, wrong position
    ABSENT = "⬛"    # Gray - letter not in word


@dataclass
class GuessResult:
    guess: str
    feedback: List[LetterState]
    
    def is_win(self) -> bool:
        return all(s == LetterState.CORRECT for s in self.feedback)
    
    def feedback_str(self) -> str:
        return "".join(s.value for s in self.feedback)


class WordleGame:
    """Simulates a Wordle game with a target word."""
    
    def __init__(self, target: str, max_guesses: int = 100):
        self.target = target.upper()
        self.max_guesses = max_guesses  # 100 = practically unlimited
        self.guesses: List[GuessResult] = []
        self.is_over = False
        self.is_won = False
    
    def make_guess(self, guess: str) -> GuessResult:
        """Make a guess and get feedback."""
        if self.is_over:
            raise ValueError("Game is already over")
        
        guess = guess.upper()
        feedback = self._compute_feedback(guess)
        result = GuessResult(guess=guess, feedback=feedback)
        self.guesses.append(result)
        
        if result.is_win():
            self.is_over = True
            self.is_won = True
        elif len(self.guesses) >= self.max_guesses:
            self.is_over = True
            self.is_won = False
        
        return result
    
    def _compute_feedback(self, guess: str) -> List[LetterState]:
        """Compute Wordle feedback for a guess."""
        feedback = [LetterState.ABSENT] * 5
        target_chars = list(self.target)
        guess_chars = list(guess)
        
        # First pass: mark correct (green)
        for i in range(5):
            if guess_chars[i] == target_chars[i]:
                feedback[i] = LetterState.CORRECT
                target_chars[i] = None  # Mark as used
        
        # Second pass: mark present (yellow)
        for i in range(5):
            if feedback[i] == LetterState.CORRECT:
                continue
            if guess_chars[i] in target_chars:
                feedback[i] = LetterState.PRESENT
                # Remove first occurrence
                idx = target_chars.index(guess_chars[i])
                target_chars[idx] = None
        
        return feedback
    
    @property
    def num_guesses(self) -> int:
        return len(self.guesses)


# ============== WORD LIST MANAGER ==============

class WordManager:
    """Manages word list and filtering based on feedback."""
    
    def __init__(self, words_file: str = "data/words.txt"):
        self.all_words = self._load_words(words_file)
        self.remaining_words = self.all_words.copy()
    
    def _load_words(self, filepath: str) -> List[str]:
        """Load words from file."""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(script_dir, filepath)
        
        with open(full_path, 'r') as f:
            words = [line.strip().upper() for line in f if len(line.strip()) == 5]
        return words
    
    def reset(self):
        """Reset to full word list."""
        self.remaining_words = self.all_words.copy()
    
    def filter_words(self, guess: str, feedback: List[LetterState]) -> List[str]:
        """Filter remaining words based on guess feedback."""
        guess = guess.upper()
        filtered = []
        
        for word in self.remaining_words:
            if self._matches_feedback(word, guess, feedback):
                filtered.append(word)
        
        self.remaining_words = filtered
        return filtered
    
    def _matches_feedback(self, candidate: str, guess: str, feedback: List[LetterState]) -> bool:
        """Check if a candidate word matches the feedback."""
        simulated = self._simulate_feedback(candidate, guess)
        return simulated == feedback
    
    def _simulate_feedback(self, target: str, guess: str) -> List[LetterState]:
        """Simulate feedback as if target was the answer."""
        feedback = [LetterState.ABSENT] * 5
        target_chars = list(target)
        
        for i in range(5):
            if guess[i] == target_chars[i]:
                feedback[i] = LetterState.CORRECT
                target_chars[i] = None
        
        for i in range(5):
            if feedback[i] == LetterState.CORRECT:
                continue
            if guess[i] in target_chars:
                feedback[i] = LetterState.PRESENT
                idx = target_chars.index(guess[i])
                target_chars[idx] = None
        
        return feedback


# ============== BASE ALGORITHM ==============

class BaseAlgorithm:
    """Base class for Wordle solver algorithms."""
    
    name: str = "base"
    
    def __init__(self, word_manager: WordManager):
        self.word_manager = word_manager
    
    def get_guess(self) -> str:
        """Return the next guess. Override in subclass."""
        raise NotImplementedError
    
    def update(self, guess: str, feedback: List[LetterState]):
        """Update state based on feedback."""
        self.word_manager.filter_words(guess, feedback)
    
    def reset(self):
        """Reset algorithm state for new game."""
        self.word_manager.reset()


# ============== ALGORITHM IMPLEMENTATIONS ==============
# Adapter classes to wrap app/services/algorithms/ solvers for benchmark

import random

class CSPAlgorithm(BaseAlgorithm):
    """Wrapper for CSPSolver from app.services.algorithms.csp"""
    name = "csp"
    
    def __init__(self, word_manager: WordManager):
        super().__init__(word_manager)
        # Initialize the actual CSP solver with the word list
        self.solver = CSPSolver(word_list=word_manager.all_words)
    
    def get_guess(self) -> str:
        return self.solver.make_prediction()
    
    def update(self, guess: str, feedback: List[LetterState]):
        # Convert feedback to colors string (G/Y/B)
        color_map = {
            LetterState.CORRECT: 'G',
            LetterState.PRESENT: 'Y', 
            LetterState.ABSENT: 'B'
        }
        colors = ''.join(color_map[f] for f in feedback)
        self.solver.update_feedback(guess, colors)
        # Also update word_manager for consistency
        super().update(guess, feedback)
    
    def reset(self):
        super().reset()
        self.solver.reset()


class HillClimbingAlgorithm(BaseAlgorithm):
    """Wrapper for HillClimbingSolver - greedy selection with heuristic"""
    name = "hill_climb"
    
    def __init__(self, word_manager: WordManager):
        super().__init__(word_manager)
        self.solver = HillClimbingSolver(word_list=word_manager.all_words)
    
    def get_guess(self) -> str:
        return self.solver.make_prediction()
    
    def update(self, guess: str, feedback: List[LetterState]):
        color_map = {
            LetterState.CORRECT: 'G',
            LetterState.PRESENT: 'Y', 
            LetterState.ABSENT: 'B'
        }
        colors = ''.join(color_map[f] for f in feedback)
        self.solver.update_feedback(guess, colors)
        super().update(guess, feedback)
    
    def reset(self):
        super().reset()
        self.solver.reset()


class StochasticHCAlgorithm(BaseAlgorithm):
    """Wrapper for StochasticHillClimbingSolver - random from top-M guesses"""
    name = "stochastic_hc"
    
    def __init__(self, word_manager: WordManager):
        super().__init__(word_manager)
        self.solver = StochasticHillClimbingSolver(word_list=word_manager.all_words, top_m=10)
    
    def get_guess(self) -> str:
        return self.solver.make_prediction()
    
    def update(self, guess: str, feedback: List[LetterState]):
        color_map = {
            LetterState.CORRECT: 'G',
            LetterState.PRESENT: 'Y', 
            LetterState.ABSENT: 'B'
        }
        colors = ''.join(color_map[f] for f in feedback)
        self.solver.update_feedback(guess, colors)
        super().update(guess, feedback)
    
    def reset(self):
        super().reset()
        self.solver.reset()


class SimulatedAnnealingAlgorithm(BaseAlgorithm):
    """Placeholder for SA - will import from app.services.algorithms.simulated_annealing"""
    name = "sa"
    
    def get_guess(self) -> str:
        remaining = self.word_manager.remaining_words
        return remaining[0] if remaining else "CRANE"


class GeneticAlgorithm(BaseAlgorithm):
    """Placeholder for GA - will import from app.services.algorithms.genetic"""
    name = "genetic"
    
    def get_guess(self) -> str:
        remaining = self.word_manager.remaining_words
        if not remaining:
            return "CRANE"
        return random.choice(remaining)


class EntropyAlgorithm(BaseAlgorithm):
    """Placeholder for Entropy - will import from app.services.algorithms.entropy"""
    name = "entropy"
    
    def __init__(self, word_manager: WordManager):
        super().__init__(word_manager)
        self.guess_count = 0
    
    def get_guess(self) -> str:
        self.guess_count += 1
        if self.guess_count == 1:
            return "CRANE"
        remaining = self.word_manager.remaining_words
        return remaining[0] if remaining else "CRANE"
    
    def reset(self):
        super().reset()
        self.guess_count = 0


# ============== ALGORITHM REGISTRY ==============
# TODO: Update these imports when algorithms are implemented in app/services/algorithms/

ALGORITHMS: Dict[str, type] = {
    "csp": CSPAlgorithm,
    "hill_climb": HillClimbingAlgorithm,
    "stochastic_hc": StochasticHCAlgorithm,
    "sa": SimulatedAnnealingAlgorithm,
    "genetic": GeneticAlgorithm,
    "entropy": EntropyAlgorithm,
}


# ============== BENCHMARK RUNNER ==============

@dataclass
class GameResult:
    word: str
    guesses: int
    won: bool
    time_ms: float
    guess_sequence: List[str]


@dataclass 
class BenchmarkResult:
    algorithm: str
    total_words: int
    wins: int
    losses: int
    win_rate: float
    avg_guesses: float
    total_time_sec: float
    avg_time_ms: float
    distribution: Dict[int, int]  # guesses -> count
    results: List[GameResult]


def run_single_game(algorithm: BaseAlgorithm, target_word: str, max_guesses: int = 100) -> GameResult:
    """Run a single Wordle game with the algorithm. max_guesses=100 means practically unlimited."""
    algorithm.reset()
    game = WordleGame(target_word, max_guesses)
    
    start_time = time.perf_counter()
    guess_sequence = []
    
    while not game.is_over:
        guess = algorithm.get_guess()
        guess_sequence.append(guess)
        result = game.make_guess(guess)
        algorithm.update(guess, result.feedback)
    
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    
    return GameResult(
        word=target_word,
        guesses=game.num_guesses,
        won=game.is_won,
        time_ms=elapsed_ms,
        guess_sequence=guess_sequence
    )


def run_benchmark(
    algorithm_names: List[str],
    word_manager: WordManager,
    start_idx: int = 0,
    end_idx: Optional[int] = None,
    max_guesses: int = 100,  # 100 = practically unlimited
    verbose: bool = True
) -> Dict[str, BenchmarkResult]:
    """Run benchmark for specified algorithms on word range."""
    
    words = word_manager.all_words
    if end_idx is None:
        end_idx = len(words)
    
    test_words = words[start_idx:end_idx]
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"WORDLE BENCHMARK")
        print(f"{'='*60}")
        print(f"Words: {start_idx} to {end_idx} ({len(test_words)} words)")
        print(f"Algorithms: {', '.join(algorithm_names)}")
        print(f"Max guesses: {'unlimited' if max_guesses >= 100 else max_guesses}")
        print(f"{'='*60}\n")
    
    all_results = {}
    
    for algo_name in algorithm_names:
        if algo_name not in ALGORITHMS:
            print(f"⚠️  Unknown algorithm: {algo_name}, skipping...")
            continue
        
        if verbose:
            print(f"\n▶ Running {algo_name}...")
        
        AlgoClass = ALGORITHMS[algo_name]
        algorithm = AlgoClass(word_manager)
        
        results: List[GameResult] = []
        # Extended distribution: 1-10, and 11+ (failed or very long)
        distribution = {}
        
        start_time = time.perf_counter()
        
        # Use tqdm for progress bar
        word_iterator = tqdm(test_words, desc=f"  {algo_name}", disable=not verbose, 
                            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
        
        for target in word_iterator:
            # Reset algorithm state for each word (independent games)
            result = run_single_game(algorithm, target, max_guesses)
            results.append(result)
            
            # Track distribution for any number of guesses
            # Win = solved in ≤6 guesses, Lose = solved in >6 guesses or not solved
            guesses = result.guesses
            distribution[guesses] = distribution.get(guesses, 0) + 1
        
        total_time = time.perf_counter() - start_time
        
        # Win = solved in ≤6 guesses (like original Wordle rules)
        wins = sum(1 for r in results if r.won and r.guesses <= 6)
        losses = len(results) - wins
        # Calculate average guesses (only for won games makes more sense, but include all)
        avg_guesses = sum(r.guesses for r in results) / len(results) if results else 0
        avg_guesses_won = sum(r.guesses for r in results if r.won and r.guesses <= 6) / wins if wins else 0
        avg_time = sum(r.time_ms for r in results) / len(results) if results else 0
        
        benchmark_result = BenchmarkResult(
            algorithm=algo_name,
            total_words=len(test_words),
            wins=wins,
            losses=losses,
            win_rate=wins / len(test_words) * 100 if test_words else 0,
            avg_guesses=avg_guesses,
            total_time_sec=total_time,
            avg_time_ms=avg_time,
            distribution=distribution,
            results=results
        )
        
        all_results[algo_name] = benchmark_result
        
        if verbose:
            print(f"  ✓ {algo_name}: {wins}/{len(test_words)} wins ({benchmark_result.win_rate:.1f}%)")
            print(f"    Avg guesses (all): {avg_guesses:.2f}, Avg guesses (won only): {avg_guesses_won:.2f}")
    
    return all_results


def save_results(results: Dict[str, BenchmarkResult], output_path: str):
    """Save benchmark results to JSON file."""
    
    # Convert to serializable format
    for algo_name, result in results.items():
        # Convert distribution to sorted format with "solved_in_X" keys
        # Include all values from 1 to max (no gaps)
        dist_sorted = {}
        numeric_keys = [k for k in result.distribution.keys() if isinstance(k, int)]
        if numeric_keys:
            max_guesses_dist = max(numeric_keys)
            for k in range(1, max_guesses_dist + 1):
                dist_sorted[f"solved_in_{k}"] = result.distribution.get(k, 0)
        
        # Calculate additional stats - win = solved in ≤6 guesses
        wins = result.wins
        won_results = [r for r in result.results if r.won and r.guesses <= 6]
        won_guesses = [r.guesses for r in won_results]
        avg_guesses_won = sum(won_guesses) / len(won_guesses) if won_guesses else 0
        min_guesses = min(won_guesses) if won_guesses else 0
        max_guesses = max(won_guesses) if won_guesses else 0
        
        data = {
            "algorithm": result.algorithm,
            "total_words": result.total_words,
            "wins": result.wins,
            "losses": result.losses,
            "win_rate": result.win_rate,
            "avg_guesses": result.avg_guesses,
            "avg_guesses_won": avg_guesses_won,
            "min_guesses": min_guesses,
            "max_guesses": max_guesses,
            "total_time_sec": result.total_time_sec,
            "avg_time_ms": result.avg_time_ms,
            "distribution": dist_sorted,
        }
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_path}")


def print_summary(results: Dict[str, BenchmarkResult]):
    """Print summary table of results."""
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"{'Algorithm':<15} {'Win Rate':>10} {'Avg Guesses':>12} {'Avg Time':>12} {'Total Time':>12}")
    print(f"{'-'*80}")
    
    for algo_name, result in sorted(results.items(), key=lambda x: -x[1].win_rate):
        print(f"{algo_name:<15} {result.win_rate:>9.1f}% {result.avg_guesses:>12.2f} {result.avg_time_ms:>10.1f}ms {result.total_time_sec:>10.1f}s")
    
    print(f"{'='*80}")
    
    # Distribution - dynamic columns based on actual data
    print(f"\nGUESS DISTRIBUTION (Win = ≤6 guesses, Fail = >6 guesses):")
    
    # Build header - always show 1-6 for wins, then Fail for >6
    header = f"{'Algorithm':<15}"
    for i in range(1, 7):
        header += f" {i:>5}"
    header += f" {'Fail':>6}"
    print(header)
    print(f"{'-'*80}")
    
    for algo_name, result in sorted(results.items(), key=lambda x: -x[1].win_rate):
        d = result.distribution
        row = f"{algo_name:<15}"
        for i in range(1, 7):
            row += f" {d.get(i, 0):>5}"
        # Fail = sum of guesses > 6
        fail_count = sum(v for k, v in d.items() if isinstance(k, int) and k > 6)
        row += f" {fail_count:>6}"
        print(row)
    
    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(
        description="Run Wordle solver benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_benchmark.py --algorithms entropy,csp --start 0 --end 100
  python run_benchmark.py --algorithms all --end 500
  python run_benchmark.py --algorithms hill_climb,sa,genetic --start 1000 --end 1500
        """
    )
    
    parser.add_argument(
        "--algorithms", "-a",
        type=str,
        default="all",
        help=f"Comma-separated list of algorithms or 'all'. Available: {', '.join(ALGORITHMS.keys())}"
    )
    
    parser.add_argument(
        "--start", "-s",
        type=int,
        default=0,
        help="Start word index (default: 0)"
    )
    
    parser.add_argument(
        "--end", "-e",
        type=int,
        default=None,
        help="End word index (default: all words)"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output file path (default: results/benchmark_<timestamp>.json)"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output"
    )
    
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available algorithms and exit"
    )
    
    args = parser.parse_args()
    
    if args.list:
        print("Available algorithms:")
        for name in ALGORITHMS.keys():
            print(f"  - {name}")
        sys.exit(0)
    
    # Parse algorithms
    if args.algorithms.lower() == "all":
        algo_names = list(ALGORITHMS.keys())
    else:
        algo_names = [a.strip() for a in args.algorithms.split(",")]
    
    # Load words
    word_manager = WordManager()
    print(f"📚 Loaded {len(word_manager.all_words)} words")
    
    # Validate range
    max_words = len(word_manager.all_words)
    start = max(0, args.start)
    end = min(max_words, args.end) if args.end else max_words
    
    if start >= end:
        print(f"❌ Invalid range: start={start}, end={end}")
        sys.exit(1)
    
    # Run benchmark
    results = run_benchmark(
        algorithm_names=algo_names,
        word_manager=word_manager,
        start_idx=start,
        end_idx=end,
        verbose=not args.quiet
    )
    
    # Print summary
    print_summary(results)
    
    # Save results - one file per algorithm
    for algo_name, result in results.items():
        if args.output:
            output_path = args.output
        else:
            output_path = f"results/summary_{algo_name}.json"
        
        save_results({algo_name: result}, output_path)


if __name__ == "__main__":
    main()