/**
 * Wordle Solver API Client
 * =========================
 * Client for communicating with the FastAPI backend.
 */

const API_BASE = "http://localhost:8000/api/v1/solver";

export interface PredictResponse {
	guess: string;
	remaining_count: number;
	guess_number: number;
}

export interface UpdateResponse {
	success: boolean;
	remaining_count: number;
	guess_number: number;
}

export interface ResetResponse {
	success: boolean;
	remaining_count: number;
}

export interface CandidateWord {
	word: string;
	score: number;
}

export interface CandidatesResponse {
	candidates: CandidateWord[];
	remaining_count: number;
}

export interface BenchmarkStepResponse {
	word: string;
	guesses: number;
	won: boolean;
	time_ms: number;
	sequence: string[];
}

export interface AlgorithmInfo {
	id: string;
	name: string;
	description: string;
}

export type Algorithm = "csp" | "hill_climb" | "stochastic_hc" | "hc_entropy" | "stochastic_hc_entropy" | "sa" | "sa_entropy";

/**
 * Generate a unique session ID
 */
export function generateSessionId(): string {
	return `session_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`;
}

/**
 * Get next guess prediction from the algorithm
 */
export async function predict(sessionId: string, algorithm: Algorithm = "csp"): Promise<PredictResponse> {
	const response = await fetch(`${API_BASE}/predict`, {
		method: "POST",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify({
			session_id: sessionId,
			algorithm: algorithm,
		}),
	});

	if (!response.ok) {
		const error = await response.json();
		throw new Error(error.detail || "Failed to get prediction");
	}

	return response.json();
}

/**
 * Update algorithm state with feedback
 * @param feedback - String of G (green), Y (yellow), B (black)
 */
export async function update(sessionId: string, guess: string, feedback: string): Promise<UpdateResponse> {
	const response = await fetch(`${API_BASE}/update`, {
		method: "POST",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify({
			session_id: sessionId,
			guess: guess.toUpperCase(),
			feedback: feedback.toUpperCase(),
		}),
	});

	if (!response.ok) {
		const error = await response.json();
		throw new Error(error.detail || "Failed to update state");
	}

	return response.json();
}

/**
 * Reset algorithm to initial state for a new game
 */
export async function reset(sessionId: string, algorithm: Algorithm = "csp"): Promise<ResetResponse> {
	const response = await fetch(`${API_BASE}/reset`, {
		method: "POST",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify({
			session_id: sessionId,
			algorithm: algorithm,
		}),
	});

	if (!response.ok) {
		const error = await response.json();
		throw new Error(error.detail || "Failed to reset");
	}

	return response.json();
}

/**
 * Get top candidate words with scores
 */
export async function getCandidates(sessionId: string, algorithm: Algorithm = "csp", topK: number = 5): Promise<CandidatesResponse> {
	const response = await fetch(`${API_BASE}/candidates`, {
		method: "POST",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify({
			session_id: sessionId,
			algorithm: algorithm,
			top_k: topK,
		}),
	});

	if (!response.ok) {
		const error = await response.json();
		throw new Error(error.detail || "Failed to get candidates");
	}

	return response.json();
}

/**
 * Run a single benchmark game
 */
export async function benchmarkStep(
	sessionId: string,
	algorithm: Algorithm,
	targetWord: string
): Promise<BenchmarkStepResponse> {
	const response = await fetch(`${API_BASE}/benchmark/step`, {
		method: "POST",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify({
			session_id: sessionId,
			algorithm: algorithm,
			target_word: targetWord.toUpperCase(),
		}),
	});

	if (!response.ok) {
		const error = await response.json();
		throw new Error(error.detail || "Failed to run benchmark step");
	}

	return response.json();
}

/**
 * List available algorithms
 */
export async function listAlgorithms(): Promise<AlgorithmInfo[]> {
	const response = await fetch(`${API_BASE}/algorithms`);

	if (!response.ok) {
		throw new Error("Failed to list algorithms");
	}

	return response.json();
}

/**
 * Delete a session
 */
export async function deleteSession(sessionId: string): Promise<void> {
	await fetch(`${API_BASE}/session/${sessionId}`, {
		method: "DELETE",
	});
}

/**
 * Convert Wordle game state to feedback string
 * @param state Array of letter states from the game
 * @returns Feedback string (e.g., "GYBBB")
 */
export function stateToFeedback(state: string[]): string {
	return state
		.map((s) => {
			if (s === "🟩") return "G";
			if (s === "🟨") return "Y";
			return "B";
		})
		.join("");
}
