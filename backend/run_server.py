#!/usr/bin/env python3
"""
Run the Wordle Solver API server.

Usage:
    python run_server.py
    
Or with uvicorn directly:
    uvicorn app.main:app --reload --port 8000
"""

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
