"""
Wordle Solver Algorithms
========================
Collection of algorithms for solving Wordle puzzles.

Available Algorithms:
- CSPSolver: Constraint Satisfaction Problem approach (brute force with numpy)
- HillClimbingSolver: Greedy selection based on splitting heuristic
- StochasticHillClimbingSolver: Random selection from top-M guesses

Usage:
    from app.services.algorithms import CSPSolver, HillClimbingSolver
    
    solver = HillClimbingSolver()
    guess = solver.make_prediction()
    solver.update_feedback(guess, "GYBBB")
    
Pattern Matrix:
    from app.services.algorithms import ensure_pattern_matrix
    
    # Pre-generate pattern matrix on startup
    ensure_pattern_matrix()
"""

from .base import WordleSolver
from .csp import CSPSolver, BruteForceSolver
from .hill_climbing import HillClimbingSolver
from .stochastic_hc import StochasticHillClimbingSolver
from .hill_climbing_entropy import HillClimbingEntropySolver
from .stochastic_hc_entropy import StochasticHCEntropySolver
from .pattern_matrix import ensure_pattern_matrix, get_pattern, get_pattern_batch

__all__ = [
    'WordleSolver',
    'CSPSolver', 
    'BruteForceSolver',
    'HillClimbingSolver',
    'StochasticHillClimbingSolver',
    'HillClimbingEntropySolver',
    'StochasticHCEntropySolver',
    'ensure_pattern_matrix',
    'get_pattern',
    'get_pattern_batch',
]
