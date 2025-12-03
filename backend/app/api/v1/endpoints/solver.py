"""
Solver API Endpoints
====================
API endpoints for Wordle solver algorithms.

Endpoints:
- POST /predict: Get next guess from algorithm
- POST /update: Update algorithm state with feedback
- POST /reset: Reset algorithm to initial state
- GET /algorithms: List available algorithms
- POST /benchmark/step: Run one step of benchmark (for streaming)
"""

from typing import Dict, List, Optional, Literal
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.algorithms import (
    CSPSolver, 
    WordleSolver, 
    HillClimbingSolver, 
    StochasticHillClimbingSolver,
    HillClimbingEntropySolver,
    StochasticHCEntropySolver,
)

router = APIRouter()

# ============== Session Management ==============
# Store solver instances per session (in production, use Redis or similar)
# Key: session_id, Value: solver instance
_sessions: Dict[str, WordleSolver] = {}

# Available algorithms
ALGORITHMS = {
    "csp": CSPSolver,
    "hill_climb": HillClimbingSolver,
    "stochastic_hc": StochasticHillClimbingSolver,
    "hc_entropy": HillClimbingEntropySolver,
    "stochastic_hc_entropy": StochasticHCEntropySolver,
}


# ============== Request/Response Models ==============

class AlgorithmInfo(BaseModel):
    id: str
    name: str
    description: str


class PredictRequest(BaseModel):
    session_id: str = Field(..., description="Unique session identifier")
    algorithm: str = Field(default="csp", description="Algorithm to use")


class PredictResponse(BaseModel):
    guess: str
    remaining_count: int
    guess_number: int


class UpdateRequest(BaseModel):
    session_id: str = Field(..., description="Unique session identifier")
    guess: str = Field(..., description="The word that was guessed")
    feedback: str = Field(..., description="Feedback string: G=green, Y=yellow, B=black")


class UpdateResponse(BaseModel):
    success: bool
    remaining_count: int
    guess_number: int


class ResetRequest(BaseModel):
    session_id: str = Field(..., description="Unique session identifier")
    algorithm: str = Field(default="csp", description="Algorithm to use after reset")


class ResetResponse(BaseModel):
    success: bool
    remaining_count: int


class BenchmarkStepRequest(BaseModel):
    session_id: str = Field(..., description="Unique session identifier")
    algorithm: str = Field(default="csp", description="Algorithm to use")
    target_word: str = Field(..., description="Target word to solve")


class BenchmarkStepResponse(BaseModel):
    word: str
    guesses: int
    won: bool
    time_ms: float
    sequence: List[str]


class CandidatesRequest(BaseModel):
    session_id: str = Field(..., description="Unique session identifier")
    algorithm: str = Field(default="csp", description="Algorithm to use")
    top_k: int = Field(default=5, description="Number of top candidates to return")


class CandidateWord(BaseModel):
    word: str
    score: float


class CandidatesResponse(BaseModel):
    candidates: List[CandidateWord]
    remaining_count: int


# ============== Helper Functions ==============

def get_or_create_solver(session_id: str, algorithm: str = "csp") -> WordleSolver:
    """Get existing solver or create new one for session."""
    if session_id not in _sessions:
        if algorithm not in ALGORITHMS:
            raise HTTPException(status_code=400, detail=f"Unknown algorithm: {algorithm}")
        _sessions[session_id] = ALGORITHMS[algorithm]()
    return _sessions[session_id]


def create_solver(session_id: str, algorithm: str = "csp") -> WordleSolver:
    """Create new solver for session (replaces existing)."""
    if algorithm not in ALGORITHMS:
        raise HTTPException(status_code=400, detail=f"Unknown algorithm: {algorithm}")
    _sessions[session_id] = ALGORITHMS[algorithm]()
    return _sessions[session_id]


# ============== Endpoints ==============

@router.get("/algorithms", response_model=List[AlgorithmInfo])
async def list_algorithms():
    """List all available algorithms."""
    return [
        AlgorithmInfo(id="csp", name="CSP", description="Constraint Satisfaction Problem solver"),
        AlgorithmInfo(id="hill_climb", name="Hill Climbing", description="Hill climbing optimization"),
        AlgorithmInfo(id="stochastic_hc", name="Stochastic Hill Climbing", description="Stochastic hill climbing"),
        AlgorithmInfo(id="sa", name="Simulated Annealing", description="Simulated annealing optimization"),
        AlgorithmInfo(id="genetic", name="Genetic Algorithm", description="Genetic algorithm solver"),
        AlgorithmInfo(id="entropy", name="Entropy-based", description="Maximum entropy information gain"),
    ]


@router.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    Get next guess prediction from the algorithm.
    Creates a new session if it doesn't exist.
    """
    solver = get_or_create_solver(request.session_id, request.algorithm)
    
    try:
        guess = solver.make_prediction()
        return PredictResponse(
            guess=guess,
            remaining_count=solver.num_remaining,
            guess_number=solver.num_guesses + 1,  # Next guess number
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/update", response_model=UpdateResponse)
async def update(request: UpdateRequest):
    """
    Update algorithm state with guess feedback.
    Must call predict first to create session.
    """
    if request.session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found. Call /predict first.")
    
    solver = _sessions[request.session_id]
    
    # Validate feedback
    feedback = request.feedback.upper()
    if not all(c in "GYB" for c in feedback):
        raise HTTPException(status_code=400, detail="Feedback must contain only G, Y, B characters")
    
    if len(feedback) != 5:
        raise HTTPException(status_code=400, detail="Feedback must be 5 characters")
    
    try:
        solver.update_feedback(request.guess.upper(), feedback)
        return UpdateResponse(
            success=True,
            remaining_count=solver.num_remaining,
            guess_number=solver.num_guesses,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reset", response_model=ResetResponse)
async def reset(request: ResetRequest):
    """
    Reset algorithm to initial state for a new game.
    Can optionally change the algorithm.
    """
    solver = create_solver(request.session_id, request.algorithm)
    
    return ResetResponse(
        success=True,
        remaining_count=solver.num_remaining,
    )


@router.post("/candidates", response_model=CandidatesResponse)
async def get_candidates(request: CandidatesRequest):
    """
    Get top candidate words with scores.
    Used for the ANALYSIS panel hints.
    """
    solver = get_or_create_solver(request.session_id, request.algorithm)
    
    # Get remaining words and score them
    remaining = solver.remaining_words[:100]  # Limit for performance
    
    # Score words (simple frequency-based scoring for now)
    from collections import Counter
    all_letters = "".join(remaining)
    freq = Counter(all_letters)
    
    scored = []
    for word in remaining:
        unique = set(word)
        score = sum(freq.get(c, 0) for c in unique) / 100  # Normalize
        scored.append(CandidateWord(word=word, score=round(score, 1)))
    
    # Sort by score descending
    scored.sort(key=lambda x: -x.score)
    
    return CandidatesResponse(
        candidates=scored[:request.top_k],
        remaining_count=solver.num_remaining,
    )


@router.post("/benchmark/step", response_model=BenchmarkStepResponse)
async def benchmark_step(request: BenchmarkStepRequest):
    """
    Run a single game for benchmark.
    Used by StatisticsPanel to run tests one word at a time.
    """
    import time
    
    # Create fresh solver for this benchmark run
    if request.algorithm not in ALGORITHMS:
        raise HTTPException(status_code=400, detail=f"Unknown algorithm: {request.algorithm}")
    
    solver = ALGORITHMS[request.algorithm]()
    target = request.target_word.upper()
    
    start_time = time.perf_counter()
    sequence = []
    won = False
    max_guesses = 6
    
    for _ in range(max_guesses):
        guess = solver.make_prediction()
        sequence.append(guess)
        
        if guess == target:
            won = True
            break
        
        # Compute feedback
        feedback = solver.compute_feedback(guess, target)
        solver.update_feedback(guess, feedback)
    
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    
    return BenchmarkStepResponse(
        word=target,
        guesses=len(sequence),
        won=won,
        time_ms=round(elapsed_ms, 2),
        sequence=sequence,
    )


@router.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a session to free memory."""
    if session_id in _sessions:
        del _sessions[session_id]
        return {"success": True, "message": "Session deleted"}
    return {"success": False, "message": "Session not found"}
