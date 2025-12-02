# Wordle+ AI Lab Edition

🎮 A Wordle game with AI solver algorithms for educational purposes.

## 📁 Project Structure

```
Wordle/
├── backend/          # FastAPI backend with solver algorithms
│   ├── app/
│   │   ├── api/v1/endpoints/solver.py   # API endpoints
│   │   └── services/algorithms/         # Solver algorithms
│   ├── data/words.txt                   # Word dictionary
│   ├── results/                         # Benchmark results
│   ├── run_server.py                    # Run API server
│   └── run_benchmark.py                 # Run benchmark tests
│
└── frontend/wordle/  # Svelte frontend
    └── src/
        ├── App.svelte                   # Main app with Play Mode
        └── components/
            ├── Game.svelte              # Game board
            ├── StatisticsPanel.svelte   # Benchmark statistics
            └── GuessProgressChart.svelte # Progress visualization
```

## 🚀 Quick Start

### 1. Backend Setup

```bash
cd backend

# Create virtual environment (optional but recommended)
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run API server
python run_server.py
```

API will run at: `http://localhost:8000`

### 2. Frontend Setup

```bash
cd frontend/wordle

# Install dependencies
npm install

# Run development server
npm run dev
```

Frontend will run at: `http://localhost:5173`

## 🎯 Features

### Play Modes

- **👤 Human Mode**: Player guesses manually with AI hints
- **🤖 AI Agent Mode**: AI plays automatically using selected algorithm

### Statistics Panel

- Run benchmarks on multiple words
- Display guess distribution charts
- Detailed result logs


## 📊 Run Benchmark (Local)

```bash
cd backend

# Run all algorithms on 100 words
python run_benchmark.py --algorithms all --start 0 --end 100

# Run specific algorithm
python run_benchmark.py --algorithms csp,hill_climb --start 0 --end 500

# Options
python run_benchmark.py --help
```

Results will be saved to `backend/results/summary_<algorithm>.json`

## 🛠️ Development

### Backend

```bash
cd backend

# Run with auto-reload
uvicorn app.main:app --reload --port 8000

# Or use run_server.py
python run_server.py
```

### Frontend

```bash
cd frontend/wordle

# Development
npm run dev

# Build for production
npm run build
```

## 📝 Notes

- Make sure the backend is running before using AI features
- Word dictionary contains ~14,855 five-letter words
- Feedback format: `G` = Green (correct), `Y` = Yellow (wrong position), `B` = Black/Gray (not in word)

## 📈 Benchmark Results

Benchmark results on all 14,855 words:

| Algorithm | Win Rate | Avg Guesses | Time/Word | Total Time |
|-----------|----------|-------------|-----------|------------|
| CSP (Brute Force) | 74.0% | 5.65 | 14.6ms | 242.9s |
| Hill Climbing | 98.5% | 4.39 | 25.8ms | 408.3s |
| Stochastic Hill Climbing | 95.9% | 4.57 | 26.9ms | 425.2s |

### Guess Distribution

| Algorithm | 1 | 2 | 3 | 4 | 5 | 6 | Fail (>6) |
|-----------|---|------|------|------|------|------|-----------|
| CSP | 1 | 116 | 1,022 | 2,782 | 3,847 | 3,221 | 3,866 |
| Hill Climbing | 1 | 33 | 1,725 | 7,120 | 4,634 | 1,114 | 228 |
| Stochastic HC | 0 | 32 | 1,570 | 6,198 | 4,867 | 1,576 | 612 |

- **Win**: Solved within ≤6 guesses
- **Fail**: Required >6 guesses to solve

## 👥 Contributors

- 23125005 - Xỉn Quý Hùng
- 23125014 - Nguyễn Thành Nhân
- 23125021 - Đoàn Đức Tuấn
- 23125074 - Thới Gia Nghi
