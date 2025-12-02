"""
Pattern Matrix Generator for Wordle
====================================
Pre-computes the pattern matrix between all word pairs for fast lookup.

Based on 3b1b's approach: https://github.com/3b1b/videos/tree/master/_2022/wordle

The pattern matrix is a N×N matrix where N is the number of words.
pattern_matrix[i, j] = the pattern hash when guessing word[i] against answer word[j]

Pattern encoding (ternary → integer):
- 0 = grey/absent
- 1 = yellow/present  
- 2 = green/correct
- Pattern hash = sum(color[i] * 3^i) for i in 0..4
"""

import os
import numpy as np
from typing import List, Optional, Dict
import itertools as it

# Constants
MISS = np.uint8(0)      # Grey - letter not in word
MISPLACED = np.uint8(1) # Yellow - letter in word but wrong position
EXACT = np.uint8(2)     # Green - letter in correct position

# File paths
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "data")
PATTERN_MATRIX_FILE = os.path.join(DATA_DIR, "pattern_matrix.npy")
WORDS_INDEX_FILE = os.path.join(DATA_DIR, "words_index.npy")

# Global cache for pattern matrix
PATTERN_GRID_DATA: Dict = {}


def words_to_int_arrays(words: List[str]) -> np.ndarray:
    """Convert list of words to numpy array of ASCII codes."""
    return np.array([[ord(c) for c in w.upper()] for w in words], dtype=np.uint8)


def generate_pattern_matrix(words1: List[str], words2: List[str]) -> np.ndarray:
    """
    Generate pattern matrix between two word lists.
    
    This computes the pairwise patterns between all words in words1 (as guesses)
    and all words in words2 (as answers).
    
    Pattern encoding: grey=0, yellow=1, green=2
    Stored as single integer: pattern = sum(color[i] * 3^i)
    
    Args:
        words1: List of guess words
        words2: List of answer words
        
    Returns:
        np.ndarray: Shape (len(words1), len(words2)) with pattern hashes
    """
    # Number of letters/words
    nl = len(words1[0])  # Should be 5
    nw1 = len(words1)
    nw2 = len(words2)
    
    # Convert to int arrays
    word_arr1, word_arr2 = words_to_int_arrays(words1), words_to_int_arrays(words2)
    
    # equality_grid[a, b, i, j] = True when words1[a][i] == words2[b][j]
    equality_grid = np.zeros((nw1, nw2, nl, nl), dtype=bool)
    for i, j in it.product(range(nl), range(nl)):
        equality_grid[:, :, i, j] = np.equal.outer(word_arr1[:, i], word_arr2[:, j])
    
    # full_pattern_matrix[a, b, i] = color for position i when guessing a against answer b
    full_pattern_matrix = np.zeros((nw1, nw2, nl), dtype=np.uint8)
    
    # Green pass - exact matches
    for i in range(nl):
        matches = equality_grid[:, :, i, i].flatten()
        full_pattern_matrix[:, :, i].flat[matches] = EXACT
        
        # Mark these letters as used
        for k in range(nl):
            equality_grid[:, :, k, i].flat[matches] = False
            equality_grid[:, :, i, k].flat[matches] = False
    
    # Yellow pass - misplaced letters
    for i, j in it.product(range(nl), range(nl)):
        matches = equality_grid[:, :, i, j].flatten()
        full_pattern_matrix[:, :, i].flat[matches] = MISPLACED
        
        # Mark as used
        for k in range(nl):
            equality_grid[:, :, k, j].flat[matches] = False
            equality_grid[:, :, i, k].flat[matches] = False
    
    # Convert 5-color pattern to single integer hash
    # pattern = color[0] + color[1]*3 + color[2]*9 + color[3]*27 + color[4]*81
    pattern_matrix = np.dot(
        full_pattern_matrix,
        (3 ** np.arange(nl)).astype(np.uint8)
    )
    
    return pattern_matrix


def generate_pattern_matrix_chunked(words: List[str], chunk_size: int = 5000) -> np.ndarray:
    """
    Generate full pattern matrix in chunks to manage memory.
    
    Args:
        words: List of all words
        chunk_size: Size of each chunk
        
    Returns:
        np.ndarray: Full N×N pattern matrix
    """
    n = len(words)
    pattern_matrix = np.zeros((n, n), dtype=np.uint8)
    
    for i in range(0, n, chunk_size):
        i_end = min(i + chunk_size, n)
        for j in range(0, n, chunk_size):
            j_end = min(j + chunk_size, n)
            
            chunk = generate_pattern_matrix(words[i:i_end], words[j:j_end])
            pattern_matrix[i:i_end, j:j_end] = chunk
            
            print(f"  Generated chunk [{i}:{i_end}, {j}:{j_end}]")
    
    return pattern_matrix


def generate_and_save_pattern_matrix(words: List[str], force: bool = False) -> np.ndarray:
    """
    Generate pattern matrix and save to file.
    
    Args:
        words: List of all words
        force: If True, regenerate even if file exists
        
    Returns:
        np.ndarray: Pattern matrix
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    
    if not force and os.path.exists(PATTERN_MATRIX_FILE) and os.path.exists(WORDS_INDEX_FILE):
        print("Pattern matrix already exists. Loading from file...")
        return load_pattern_matrix(words)
    
    print(f"Generating pattern matrix for {len(words)} words...")
    print("This may take a few minutes...")
    
    # Sort words for consistent indexing
    words = sorted([w.upper() for w in words])
    
    pattern_matrix = generate_pattern_matrix_chunked(words)
    
    # Save matrix and word index
    np.save(PATTERN_MATRIX_FILE, pattern_matrix)
    np.save(WORDS_INDEX_FILE, np.array(words))
    
    print(f"Saved pattern matrix to {PATTERN_MATRIX_FILE}")
    print(f"Saved words index to {WORDS_INDEX_FILE}")
    
    return pattern_matrix


def load_pattern_matrix(words: Optional[List[str]] = None) -> np.ndarray:
    """
    Load pattern matrix from file.
    
    Args:
        words: Optional list of words to verify against saved index
        
    Returns:
        np.ndarray: Pattern matrix
    """
    global PATTERN_GRID_DATA
    
    if PATTERN_GRID_DATA:
        return PATTERN_GRID_DATA['grid']
    
    if not os.path.exists(PATTERN_MATRIX_FILE):
        raise FileNotFoundError(
            f"Pattern matrix not found at {PATTERN_MATRIX_FILE}. "
            "Run generate_and_save_pattern_matrix() first."
        )
    
    PATTERN_GRID_DATA['grid'] = np.load(PATTERN_MATRIX_FILE)
    PATTERN_GRID_DATA['words'] = np.load(WORDS_INDEX_FILE)
    PATTERN_GRID_DATA['words_to_index'] = {
        w: i for i, w in enumerate(PATTERN_GRID_DATA['words'])
    }
    
    print(f"Loaded pattern matrix: {PATTERN_GRID_DATA['grid'].shape}")
    
    return PATTERN_GRID_DATA['grid']


def ensure_pattern_matrix(words: List[str]) -> np.ndarray:
    """
    Ensure pattern matrix exists, generating if needed.
    
    Args:
        words: List of words to use
        
    Returns:
        np.ndarray: Pattern matrix
    """
    if os.path.exists(PATTERN_MATRIX_FILE) and os.path.exists(WORDS_INDEX_FILE):
        return load_pattern_matrix(words)
    else:
        return generate_and_save_pattern_matrix(words)


def get_pattern(guess: str, answer: str) -> int:
    """
    Get pattern hash for a guess-answer pair.
    
    Args:
        guess: The guessed word
        answer: The target word
        
    Returns:
        int: Pattern hash (0-242)
    """
    global PATTERN_GRID_DATA
    
    guess = guess.upper()
    answer = answer.upper()
    
    if PATTERN_GRID_DATA:
        w2i = PATTERN_GRID_DATA['words_to_index']
        if guess in w2i and answer in w2i:
            return int(PATTERN_GRID_DATA['grid'][w2i[guess], w2i[answer]])
    
    # Fallback: compute directly
    return int(generate_pattern_matrix([guess], [answer])[0, 0])


def get_pattern_batch(guess: str, answers: List[str]) -> np.ndarray:
    """
    Get patterns for one guess against multiple answers.
    
    Args:
        guess: The guessed word
        answers: List of potential answer words
        
    Returns:
        np.ndarray: Array of pattern hashes
    """
    global PATTERN_GRID_DATA
    
    guess = guess.upper()
    
    if PATTERN_GRID_DATA:
        w2i = PATTERN_GRID_DATA['words_to_index']
        if guess in w2i:
            guess_idx = w2i[guess]
            answer_indices = [w2i[a.upper()] for a in answers if a.upper() in w2i]
            return PATTERN_GRID_DATA['grid'][guess_idx, answer_indices]
    
    # Fallback
    return generate_pattern_matrix([guess], [a.upper() for a in answers])[0]


def pattern_to_string(pattern: int) -> str:
    """
    Convert pattern hash to emoji string.
    
    Args:
        pattern: Pattern hash (0-242)
        
    Returns:
        str: Emoji representation (e.g., "🟩🟨⬛⬛⬛")
    """
    d = {MISS: "⬛", MISPLACED: "🟨", EXACT: "🟩"}
    result = []
    for _ in range(5):
        result.append(d[pattern % 3])
        pattern //= 3
    return "".join(result)


def pattern_to_feedback(pattern: int) -> str:
    """
    Convert pattern hash to feedback string (G/Y/B).
    
    Args:
        pattern: Pattern hash (0-242)
        
    Returns:
        str: Feedback string (e.g., "GYBBB")
    """
    d = {MISS: "B", MISPLACED: "Y", EXACT: "G"}
    result = []
    for _ in range(5):
        result.append(d[pattern % 3])
        pattern //= 3
    return "".join(result)


def feedback_to_pattern(feedback: str) -> int:
    """
    Convert feedback string to pattern hash.
    
    Args:
        feedback: Feedback string using G/Y/B
        
    Returns:
        int: Pattern hash
    """
    d = {"B": MISS, "Y": MISPLACED, "G": EXACT}
    pattern = 0
    for i, c in enumerate(feedback.upper()):
        pattern += d[c] * (3 ** i)
    return pattern


def filter_words_by_pattern(guess: str, pattern: int, word_list: List[str]) -> List[str]:
    """
    Filter word list to only words consistent with guess+pattern.
    
    Args:
        guess: The guessed word
        pattern: Pattern hash received
        word_list: List of candidate words
        
    Returns:
        List[str]: Filtered words
    """
    global PATTERN_GRID_DATA
    
    guess = guess.upper()
    
    if PATTERN_GRID_DATA:
        w2i = PATTERN_GRID_DATA['words_to_index']
        if guess in w2i:
            guess_idx = w2i[guess]
            grid = PATTERN_GRID_DATA['grid']
            
            # Get indices for word list
            indices = np.array([w2i[w.upper()] for w in word_list if w.upper() in w2i])
            
            # Filter by pattern match
            patterns = grid[guess_idx, indices]
            matching_indices = indices[patterns == pattern]
            
            words_array = PATTERN_GRID_DATA['words']
            return [words_array[i] for i in matching_indices]
    
    # Fallback: compute patterns directly
    patterns = generate_pattern_matrix([guess], [w.upper() for w in word_list])[0]
    return [w for w, p in zip(word_list, patterns) if p == pattern]


if __name__ == "__main__":
    # Test/generate pattern matrix
    import sys
    
    # Load word list
    words_file = os.path.join(DATA_DIR, "words.txt")
    if os.path.exists(words_file):
        with open(words_file, 'r') as f:
            words = [line.strip().upper() for line in f if len(line.strip()) == 5]
    else:
        print(f"Words file not found: {words_file}")
        sys.exit(1)
    
    print(f"Loaded {len(words)} words")
    
    # Generate or load matrix
    if "--force" in sys.argv:
        generate_and_save_pattern_matrix(words, force=True)
    else:
        ensure_pattern_matrix(words)
    
    # Test
    test_guess = "CRANE"
    test_answer = "SLANT"
    pattern = get_pattern(test_guess, test_answer)
    print(f"\nTest: {test_guess} vs {test_answer}")
    print(f"Pattern hash: {pattern}")
    print(f"Pattern string: {pattern_to_string(pattern)}")
    print(f"Pattern feedback: {pattern_to_feedback(pattern)}")
