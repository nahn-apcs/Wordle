"""
FastAPI Main Application
========================
Entry point for the Wordle Solver API.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.endpoints import solver
from app.services.algorithms import ensure_pattern_matrix


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load pattern matrix on startup."""
    print("🚀 Starting server...")
    print("📊 Loading pattern matrix (this may take a moment on first run)...")
    
    # Load words and ensure pattern matrix is ready
    import os
    words_path = os.path.join(os.path.dirname(__file__), "../data/words.txt")
    with open(words_path, 'r') as f:
        words = [line.strip().upper() for line in f if len(line.strip()) == 5]
    
    ensure_pattern_matrix(words)
    print("✅ Pattern matrix loaded! Server ready.")
    
    yield  # Server is running
    
    print("👋 Shutting down server...")


app = FastAPI(
    title="Wordle Solver API",
    description="API for Wordle solving algorithms",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(solver.router, prefix="/api/v1/solver", tags=["solver"])


@app.get("/")
async def root():
    return {"message": "Wordle Solver API", "version": "1.0.0"}


@app.get("/health")
async def health():
    return {"status": "healthy"}
