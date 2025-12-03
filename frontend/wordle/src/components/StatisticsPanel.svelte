<script lang="ts">
	import { onMount, onDestroy } from "svelte";
	import { benchmarkStep, generateSessionId, type Algorithm } from "../api";
	import { words } from "../utils";

	const algorithms: { value: Algorithm; label: string }[] = [
		{ value: "csp", label: "CSP" },
		{ value: "hill_climb", label: "Hill Climbing" },
		{ value: "stochastic_hc", label: "Stochastic Hill Climbing" },
		{ value: "hc_entropy", label: "Hill Climbing Entropy" },
		{ value: "stochastic_hc_entropy", label: "Stochastic HC Entropy" },
		{ value: "sa", label: "Simulated Annealing" },
		{ value: "sa_entropy", label: "Simulated Annealing Entropy" },
		{ value: "ga", label: "Genetic Algorithm" },
		{ value: "ga_entropy", label: "Genetic Algorithm Entropy" },
	];

	let selectedAlgo: Algorithm = "csp";
	let wordCount = 200;
	const maxWords = words.words.length;

	// Distribution data
	let distribution = [0, 0, 0, 0, 0, 0, 0]; // 1, 2, 3, 4, 5, 6, >6

	// Results log
	interface ResultEntry {
		word: string;
		time: number;
		guesses: number;
		ok: boolean;
	}
	let results: ResultEntry[] = [];

	// Running state
	let isRunning = false;
	let progress = 0;
	let shouldStop = false;
	let sessionId = generateSessionId();

	// Run algorithm benchmark using FastAPI
	async function runAlgorithm() {
		if (isRunning) return;

		isRunning = true;
		shouldStop = false;
		progress = 0;
		results = [];
		distribution = [0, 0, 0, 0, 0, 0, 0];
		sessionId = generateSessionId();

		try {
			// Get test words from dictionary
			const testWords = words.words.slice(0, wordCount).map(w => w.toUpperCase());
			
			for (let i = 0; i < testWords.length; i++) {
				if (shouldStop) break;

				const targetWord = testWords[i];
				
				try {
					// Call API for benchmark step
					const result = await benchmarkStep(sessionId, selectedAlgo, targetWord);
					
					const ok = result.won;
					const guesses = result.guesses;
					
					results = [
						...results,
						{ 
							word: result.word, 
							time: result.time_ms, 
							guesses: guesses, 
							ok: ok 
						},
					];

					// Update distribution
					if (ok && guesses <= 6) {
						distribution[guesses - 1]++;
					} else {
						distribution[6]++;
					}
					distribution = [...distribution];
				} catch (apiError) {
					console.error(`API error for word ${targetWord}:`, apiError);
					// Continue with next word on error
				}

				progress = ((i + 1) / testWords.length) * 100;
			}
		} catch (e) {
			console.error("Algorithm run error:", e);
		} finally {
			isRunning = false;
			shouldStop = false;
		}
	}

	function stopAlgorithm() {
		shouldStop = true;
	}

	function formatTime(ms: number): string {
		return ms.toFixed(1) + "ms";
	}

	$: maxDist = Math.max(...distribution, 1);
</script>

<div class="stats-panel">
	<div class="card-label">📈 STATISTICS</div>

	<div class="content">
		<!-- Algorithm Selection -->
		<div class="field">
			<div class="field-label">Algorithm</div>
			<select bind:value={selectedAlgo} disabled={isRunning}>
				{#each algorithms as algo}
					<option value={algo.value}>{algo.label}</option>
				{/each}
			</select>
		</div>

		<!-- Word Count Slider -->
		<div class="field">
			<div class="field-label">Test Words</div>
			<div class="slider-box">
				<div class="slider-labels">
					<span>1</span>
					<span class="slider-value">{wordCount}</span>
					<span>{maxWords}</span>
				</div>
				<input
					type="range"
					min="1"
					max={maxWords}
					bind:value={wordCount}
					disabled={isRunning}
					class="slider"
				/>
			</div>
		</div>

		<!-- Run Button -->
		<button
			class="run-btn"
			class:running={isRunning}
			on:click={(e) => { isRunning ? stopAlgorithm() : runAlgorithm(); e.currentTarget.blur(); }}
		>
			{#if isRunning}
				<span class="stop-icon">⏹</span> STOP ({progress.toFixed(0)}%)
			{:else}
				<span class="play-icon">▷</span> RUN
			{/if}
		</button>

		<!-- Distribution Chart - Horizontal Bars -->
		<div class="section">
			<div class="section-title">📊 GUESS DISTRIBUTION</div>
			<div class="dist-rows">
				{#each distribution as count, i}
					<div class="dist-row">
						<div class="dist-label">{i < 6 ? i + 1 : "X"}</div>
						<div class="dist-bar-wrap">
							<div 
								class="dist-bar-h" 
								class:fail={i === 6}
								style="width: {Math.max(8, (count / maxDist) * 100)}%"
							>
								<span class="dist-count">{count}</span>
							</div>
						</div>
					</div>
				{/each}
			</div>
		</div>

		<!-- Results Log -->
		<div class="section results-section">
			<div class="section-title">📋 RESULTS LOG</div>
			<div class="results-header">
				<span>WORD</span>
				<span>Time</span>
				<span>Tries</span>
				<span>OK</span>
			</div>
			<div class="results-list">
				{#each results.slice(-50).reverse() as r}
					<div class="result-row" class:fail={!r.ok}>
						<span class="result-word">{r.word}</span>
						<span>{formatTime(r.time)}</span>
						<span>{r.guesses}</span>
						<span>{r.ok ? "✓" : "✗"}</span>
					</div>
				{/each}
				{#if results.length === 0}
					<div class="no-results">No results yet</div>
				{/if}
			</div>
		</div>
	</div>
</div>

<style>
	.stats-panel {
		height: 100%;
		display: flex;
		flex-direction: column;
		padding: 18px;
		box-sizing: border-box;
		position: relative;
		background: linear-gradient(145deg, var(--color-background, #fff), var(--color-tone-7, #f8f8f8));
		border-radius: 20px;
		overflow: hidden;
	}

	.stats-panel::before {
		content: '';
		position: absolute;
		top: 0;
		left: 0;
		right: 0;
		height: 4px;
		background: linear-gradient(90deg, #f59e0b, #ef4444);
		border-radius: 20px 20px 0 0;
	}

	.card-label {
		font-size: 0.85rem;
		font-weight: 900;
		letter-spacing: 0.12em;
		color: var(--color-tone-1, #333);
		margin-bottom: 18px;
		margin-top: 4px;
		display: flex;
		align-items: center;
		gap: 8px;
	}

	.card-label::before {
		content: '';
		width: 4px;
		height: 18px;
		background: linear-gradient(180deg, #f59e0b, #ef4444);
		border-radius: 2px;
	}

	.content {
		flex: 1;
		display: flex;
		flex-direction: column;
		gap: 16px;
		overflow-y: auto;
	}

	.field {
		margin-bottom: 2px;
	}

	.field-label {
		font-size: 0.7rem;
		font-weight: 900;
		letter-spacing: 0.08em;
		text-transform: uppercase;
		color: var(--color-tone-2, #555);
		margin-bottom: 10px;
	}

	.section {
		background: var(--color-background, #fff);
		border-radius: 14px;
		padding: 14px;
		border: 1px solid var(--color-tone-6, #e5e5e5);
	}

	.section-title {
		font-size: 0.78rem;
		font-weight: 900;
		letter-spacing: 0.1em;
		color: var(--color-tone-1, #333);
		margin-bottom: 12px;
		text-transform: uppercase;
		display: flex;
		align-items: center;
		gap: 8px;
	}

	.section-title::before {
		content: '';
		width: 4px;
		height: 16px;
		background: linear-gradient(180deg, #f59e0b, #ef4444);
		border-radius: 2px;
	}

	select {
		width: 100%;
		padding: 14px 16px;
		border-radius: 14px;
		border: 2px solid transparent;
		background: 
			linear-gradient(var(--color-background, #fff), var(--color-background, #fff)) padding-box,
			linear-gradient(135deg, rgba(245, 158, 11, 0.3), rgba(239, 68, 68, 0.3)) border-box;
		font-size: 0.95rem;
		font-weight: 700;
		color: var(--color-tone-1, #333);
		cursor: pointer;
		outline: none;
		transition: all 0.25s ease;
		box-shadow: 0 4px 12px rgba(245, 158, 11, 0.08);
	}

	select:hover:not(:disabled) {
		background: 
			linear-gradient(var(--color-background, #fff), var(--color-background, #fff)) padding-box,
			linear-gradient(135deg, rgba(245, 158, 11, 0.5), rgba(239, 68, 68, 0.5)) border-box;
		box-shadow: 0 6px 16px rgba(245, 158, 11, 0.15);
	}

	select:focus {
		background: 
			linear-gradient(var(--color-background, #fff), var(--color-background, #fff)) padding-box,
			linear-gradient(135deg, #f59e0b, #ef4444) border-box;
		box-shadow: 0 8px 20px rgba(245, 158, 11, 0.2);
	}

	select:disabled {
		opacity: 0.6;
		cursor: not-allowed;
	}

	.slider-box {
		background: linear-gradient(135deg, rgba(245, 158, 11, 0.06), rgba(239, 68, 68, 0.06));
		padding: 18px;
		border: 1px solid rgba(245, 158, 11, 0.15);
		border-radius: 14px;
	}

	.slider-labels {
		display: flex;
		justify-content: space-between;
		align-items: center;
		margin-bottom: 14px;
		font-weight: 700;
		color: var(--color-tone-3, #888);
		font-size: 0.75rem;
	}

	.slider-value {
		font-size: 1.6rem;
		font-weight: 900;
		background: linear-gradient(135deg, #f59e0b, #ef4444);
		-webkit-background-clip: text;
		-webkit-text-fill-color: transparent;
		background-clip: text;
	}

	.slider {
		width: 100%;
		height: 8px;
		-webkit-appearance: none;
		appearance: none;
		background: linear-gradient(90deg, rgba(245, 158, 11, 0.2), rgba(239, 68, 68, 0.2));
		border-radius: 999px;
		outline: none;
		cursor: pointer;
	}

	.slider::-webkit-slider-thumb {
		-webkit-appearance: none;
		appearance: none;
		width: 24px;
		height: 24px;
		background: linear-gradient(135deg, #f59e0b, #ef4444);
		border-radius: 50%;
		cursor: pointer;
		box-shadow: 0 4px 12px rgba(245, 158, 11, 0.4);
		transition: transform 0.2s, box-shadow 0.2s;
		border: 3px solid #fff;
	}

	.slider::-webkit-slider-thumb:hover {
		transform: scale(1.15);
		box-shadow: 0 6px 18px rgba(245, 158, 11, 0.5);
	}

	.slider:disabled {
		opacity: 0.5;
		cursor: not-allowed;
	}

	.run-btn {
		width: 100%;
		padding: 16px;
		border: none;
		border-radius: 14px;
		background: linear-gradient(135deg, #f59e0b, #ef4444);
		color: white;
		font-size: 0.9rem;
		font-weight: 900;
		letter-spacing: 0.1em;
		text-transform: uppercase;
		cursor: pointer;
		display: flex;
		align-items: center;
		justify-content: center;
		gap: 10px;
		transition: all 0.25s ease;
		box-shadow: 0 6px 20px rgba(245, 158, 11, 0.35);
		position: relative;
		overflow: hidden;
	}

	.run-btn::before {
		content: '';
		position: absolute;
		top: 0;
		left: -100%;
		width: 100%;
		height: 100%;
		background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
		transition: left 0.5s ease;
	}

	.run-btn:hover::before {
		left: 100%;
	}

	.run-btn:hover {
		transform: translateY(-2px);
		box-shadow: 0 10px 28px rgba(245, 158, 11, 0.45);
	}

	.run-btn.running {
		background: #e74c3c;
		box-shadow: 0 4px 12px rgba(231, 76, 60, 0.3);
	}

	.run-btn.running:hover {
		box-shadow: 0 6px 16px rgba(231, 76, 60, 0.4);
	}

	.play-icon, .stop-icon {
		font-size: 0.8rem;
	}

	/* Distribution - Horizontal Bars */
	.dist-rows {
		display: flex;
		flex-direction: column;
		gap: 5px;
	}

	.dist-row {
		display: flex;
		align-items: center;
		gap: 8px;
	}

	.dist-label {
		width: 16px;
		font-size: 0.85rem;
		font-weight: 900;
		color: var(--color-tone-1, #333);
		text-align: center;
	}

	.dist-bar-wrap {
		flex: 1;
		height: 22px;
		background: var(--color-tone-6, #eee);
		border-radius: 4px;
		overflow: hidden;
	}

	.dist-bar-h {
		height: 100%;
		background: var(--color-correct, #6aaa64);
		border-radius: 4px;
		display: flex;
		align-items: center;
		justify-content: flex-end;
		padding-right: 4px;
		min-width: 32px;
		transition: width 0.4s ease;
		box-sizing: border-box;
	}

	.dist-bar-h.fail {
		background: var(--color-absent, #787c7e);
	}

	.dist-count {
		font-size: 0.7rem;
		font-weight: 800;
		color: #fff;
	}

	/* Results */
	.results-section {
		flex: 1;
		display: flex;
		flex-direction: column;
		min-height: 100px;
	}

	.results-header {
		display: grid;
		grid-template-columns: 1.5fr 1fr 0.7fr 0.5fr;
		gap: 4px;
		padding: 8px 10px;
		background: var(--color-tone-6, #eee);
		border-radius: 8px;
		font-size: 0.55rem;
		font-weight: 800;
		color: var(--color-tone-2, #555);
		text-transform: uppercase;
		letter-spacing: 0.04em;
	}

	.results-list {
		flex: 1;
		overflow-y: auto;
		margin-top: 8px;
		max-height: 160px;
	}

	.result-row {
		display: grid;
		grid-template-columns: 1.5fr 1fr 0.7fr 0.5fr;
		gap: 4px;
		padding: 6px 10px;
		font-size: 0.68rem;
		font-weight: 600;
		color: var(--color-tone-1, #333);
		border-bottom: 1px solid var(--color-tone-6, #eee);
		transition: background 0.15s;
	}

	.result-row:hover {
		background: var(--color-tone-7, #f9f9f9);
	}

	.result-row.fail {
		color: #c0392b;
		background: rgba(231, 76, 60, 0.06);
	}

	.result-word {
		font-weight: 800;
		font-family: ui-monospace, monospace;
	}

	.no-results {
		text-align: center;
		padding: 24px;
		color: var(--color-tone-3, #888);
		font-style: italic;
		font-size: 0.9rem;
	}

	/* Scrollbar */
	.results-list::-webkit-scrollbar,
	.content::-webkit-scrollbar {
		width: 4px;
	}

	.results-list::-webkit-scrollbar-track,
	.content::-webkit-scrollbar-track {
		background: transparent;
	}

	.results-list::-webkit-scrollbar-thumb,
	.content::-webkit-scrollbar-thumb {
		background: var(--color-tone-4, #ccc);
		border-radius: 2px;
	}
</style>
