<script context="module" lang="ts">
	import {
		modeData,
		seededRandomInt,
		Stats,
		GameState,
		Settings,
		LetterStates,
		getWordNumber,
		words,
	} from "./utils";
	import Game from "./components/Game.svelte";
	import { letterStates, settings, mode } from "./stores";
	import { GameMode } from "./enums";
	import { Toaster } from "./components/widgets";
	import { setContext, tick } from "svelte";
	import GuessProgressChart from "./components/GuessProgressChart.svelte";
	import StatisticsPanel from "./components/StatisticsPanel.svelte";



	document.title = "Wordle+ | AI Lab Edition";
</script>

<script lang="ts">
	export let version: string;

	setContext("version", version);
	localStorage.setItem("version", version);

	let stats: Stats;
	let word: string;
	let state: GameState;
	let toaster: Toaster;

	settings.set(new Settings(localStorage.getItem("settings")));
	settings.subscribe((s) =>
		localStorage.setItem("settings", JSON.stringify(s)),
	);

	const hash = window.location.hash.replace("#", "").split("/");
	const key = hash[0] as keyof typeof GameMode;
	const modeVal: GameMode = key in GameMode ? GameMode[key] : GameMode.daily;

	// Historical mode support: #DAILY/123 (word number)
	if (hash.length > 1 && !isNaN(+hash[1])) {
		modeData.modes[modeVal].seed =
			(+hash[1] - 1) * modeData.modes[modeVal].unit +
			modeData.modes[modeVal].start;
		modeData.modes[modeVal].historical = true;
	}

	mode.set(modeVal);

	mode.subscribe((m) => {
		localStorage.setItem("mode", `${m}`);
		window.location.hash = GameMode[m];

		stats = new Stats(localStorage.getItem(`stats-${m}`) || m);
		word =
			words.words[
				seededRandomInt(0, words.words.length, modeData.modes[m].seed)
			];

		if (modeData.modes[m].historical) {
			state = new GameState(m, localStorage.getItem(`state-${m}-h`));
		} else {
			state = new GameState(m, localStorage.getItem(`state-${m}`));
		}

		letterStates.set(new LetterStates(state.board));
	});

	$: saveState(state);
	function saveState(state: GameState) {
		if (modeData.modes[$mode].historical) {
			localStorage.setItem(`state-${$mode}-h`, state.toString());
		} else {
			localStorage.setItem(`state-${$mode}`, state.toString());
		}
	}

	type PlayMode = "human" | "ai";
	let playMode: PlayMode = "human";

	let hintsEnabled = true;
	let gameComponent: Game;

	function clickCandidate(candidateWord: string) {
		if (playMode === "human" && gameComponent) {
			gameComponent.inputAndSubmit(candidateWord);
		}
	}

	type Algo =
		| "csp"
		| "hill_climb"
		| "stochastic_hc"
		| "sa"
		| "genetic"
		| "entropy";

	let aiAlgo: Algo = "csp";

	// AI Agent mode: random-only target word (always valid)
	let targetWord = "";

	const validWordSet = new Set(words.words.map((w) => w.toUpperCase()));

	function randomTargetWord() {
		const w =
			words.words[seededRandomInt(0, words.words.length, Date.now())];
		targetWord = w.toUpperCase();
	}

	function runWithTargetWord() {
		if (!targetWord) randomTargetWord();

		// Safety: ensure the word exists in our dictionary
		const t = targetWord.toUpperCase();
		if (!validWordSet.has(t)) {
			randomTargetWord();
		}

		word = targetWord.toLowerCase();
		state = new GameState($mode, null); // fresh board
		letterStates.set(new LetterStates(state.board));
	}

	// Placeholder UI data (plug FastAPI later)
	let topCandidates: Array<{ word: string; score: number }> = [
		{ word: "CRANE", score: 9.2 },
		{ word: "SLATE", score: 8.8 },
		{ word: "TRACE", score: 8.5 },
		{ word: "ARISE", score: 8.2 },
		{ word: "ROAST", score: 8.0 },
	];

	// Get remaining words from GuessProgressChart (avoid duplicate calculation)
	let remainingWords = words.words.length;
	$: entropyBits = remainingWords > 0 ? Math.log2(remainingWords) : 0;

	$: if (playMode === "human") aiAlgo = "best";

	// Reset game when switching between modes
	let prevPlayMode: PlayMode | null = null;
	$: {
		if (prevPlayMode !== null && prevPlayMode !== playMode) {
			// Only reset when mode actually changes (not on initial mount)
			state = new GameState($mode, null);
			letterStates.set(new LetterStates(state.board));
			targetWord = "";
		}
		prevPlayMode = playMode;
	}

	const maxScore = () => {
		let m = 1;
		for (const c of topCandidates) m = Math.max(m, c.score);
		return m;
	};
</script>

<Toaster bind:this={toaster} />

{#if toaster}
	<div class="shell">
		<!-- LEFT: CONTROL -->
		<aside class="panel left">
			<div class="card">
				<div class="card-label">🎮 PLAY MODE</div>

				<div class="segmented">
					<button
						class:active={playMode === "human"}
						on:click={() => (playMode = "human")}
						aria-pressed={playMode === "human"}
						type="button"
					>
						👤 Human
					</button>

					<button
						class:active={playMode === "ai"}
						on:click={() => (playMode = "ai")}
						aria-pressed={playMode === "ai"}
						type="button"
					>
						🤖 AI Agent
					</button>
				</div>

				{#if playMode === "human"}
					<div class="note">
						<div class="note-title">Assisted mode</div>
						<div class="note-text">
							Human plays. The assistant uses the <b
								>strongest agent</b
							> only. You can hide hints anytime using 👁️ on the right
							panel.
						</div>
					</div>
				{:else}
					<div class="field">
						<div class="field-label">Algorithm</div>
						<select bind:value={aiAlgo}>
							<option value="csp">CSP</option>
							<option value="hill_climb">Hill Climbing</option>
							<option value="stochastic_hc">Stochastic Hill Climbing</option>
							<option value="sa">Simulated Annealing</option>
							<option value="genetic">Genetic Algorithm</option>
							<option value="entropy">Entropy-based</option>
						</select>
					</div>

					<div class="field">
						<div class="field-label">Target word (random)</div>
						<div class="target-row">
							<input
								class="target-input"
								value={targetWord}
								placeholder="Click 🎲 to pick"
								readonly
								aria-readonly="true"
								on:keydown|stopPropagation
								on:keyup|stopPropagation
							/>
							<button
								class="icon-btn"
								type="button"
								on:click={randomTargetWord}
								title="Random valid word"
								aria-label="Random valid word"
							>
								🎲
							</button>
						</div>
					</div>

					<div class="btn-row">
						<button class="primary" type="button" on:click={runWithTargetWord}>
							<span class="btn-icon">▶</span>
							RUN
						</button>
						<button class="secondary" type="button" on:click={() => runWithTargetWord()}>
							<span class="btn-icon">↻</span>
							RESET
						</button>
					</div>


				{/if}
			</div>

			<GuessProgressChart game={state} dictionary={words.words} bind:currentRemaining={remainingWords} />
		</aside>

		<!-- CENTER: GAME -->
		<main class="center">
			<div class="game-stage">
				<div class="game-zoom">
					<Game bind:this={gameComponent} {stats} bind:word {toaster} bind:game={state} disabled={playMode === 'ai'} />
				</div>
			</div>
		</main>

		<!-- RIGHT PANEL: AI HINTS or STATISTICS -->
		{#if playMode === 'ai'}
			<aside class="panel right">
				<StatisticsPanel />
			</aside>
		{:else}
			<aside class="panel right">
			<div class="panel-head right-head">
				<div class="panel-title informing">
					<span class="icon">💡</span>
					<span>ANALYSIS</span>
				</div>

				<button
					class="eye-btn"
					class:off={!hintsEnabled}
					on:click={() => (hintsEnabled = !hintsEnabled)}
					aria-pressed={hintsEnabled}
					title={hintsEnabled ? "Disable hints" : "Enable hints"}
					type="button"
				>
					{hintsEnabled ? "👁️" : "🙈"}
				</button>
			</div>

			<div class="card">
				<div class="stats-grid" class:blur={!hintsEnabled}>
					<div class="stat">
						<div class="stat-label">Remaining</div>
						<div class="stat-value">
							{hintsEnabled
								? remainingWords.toLocaleString()
								: "?"}
						</div>
					</div>
					<div class="stat">
						<div class="stat-label">Entropy</div>
						<div class="stat-value">
							{hintsEnabled
								? `${entropyBits.toFixed(1)} bits`
								: "?"}
						</div>
					</div>
				</div>

				{#if hintsEnabled}
					<div class="section-title">Top candidates</div>

					<ul class="rank-list">
						{#each topCandidates.slice(0, 5) as c}
							<li 
								class="rank-item clickable" 
								on:click={() => clickCandidate(c.word)}
								on:keypress={(e) => e.key === 'Enter' && clickCandidate(c.word)}
								role="button"
								tabindex="0"
								title="Click to use this word"
							>
								<div class="rank-word">{c.word}</div>
								<div class="rank-bar">
									<div
										class="rank-fill"
										style={`width:${Math.max(6, Math.round((c.score / maxScore()) * 100))}%`}
									></div>
								</div>
								<div class="rank-score">
									{c.score.toFixed(1)}
								</div>
							</li>
						{/each}
					</ul>
				{/if}

				{#if !hintsEnabled}
					<div class="veil">Hints are taking a nap 😴</div>
				{/if}
			</div>
		</aside>
		{/if}
	</div>
{/if}

<style>
	:global(body) {
		margin: 0;
		padding: 0;
		background: var(--color-background, #ffffff);
		color: var(--color-tone-1, #111);
		font-family:
			ui-sans-serif,
			system-ui,
			-apple-system,
			Segoe UI,
			Roboto,
			Helvetica,
			Arial,
			"Apple Color Emoji",
			"Segoe UI Emoji";
	}

	.shell {
		height: 100vh;
		width: 100vw;
		box-sizing: border-box;
		display: grid;
		grid-template-columns: 340px minmax(640px, 1fr) 340px;
		gap: 14px;
		padding: 14px;
		background: linear-gradient(
			135deg,
			var(--color-tone-7, #f7f7f7),
			var(--color-background, #fff)
		);
	}

	.panel {
		height: calc(100vh - 28px);
		border-radius: 18px;
		border: 1px solid var(--color-tone-4, rgba(0, 0, 0, 0.12));
		background: var(--color-tone-7, #f7f7f7);
		box-shadow: 0 12px 30px rgba(0, 0, 0, 0.08);
		padding: 14px;
		overflow: auto;
		-webkit-overflow-scrolling: touch;
	}

	.center {
		height: calc(100vh - 28px);
		box-sizing: border-box;
		border-radius: 18px;
		border: 1px solid var(--color-tone-4, rgba(0, 0, 0, 0.12));
		background: var(--color-background, #fff);
		box-shadow: 0 12px 30px rgba(0, 0, 0, 0.08);
		overflow: hidden;
		display: flex;
		flex-direction: column;
	}

	.panel-head {
		display: flex;
		align-items: center;
		justify-content: space-between;
		padding-bottom: 10px;
		margin-bottom: 10px;
		border-bottom: 1px solid var(--color-tone-4, rgba(0, 0, 0, 0.12));
	}

	.panel-title {
		display: flex;
		align-items: center;
		gap: 10px;
		font-weight: 1000;
		letter-spacing: 0.14em;
		text-transform: uppercase;
		font-size: 0.8rem;
		opacity: 0.9;
	}

	.icon {
		font-size: 1.05rem;
	}

	.card {
		border-radius: 20px;
		border: none;
		background: linear-gradient(145deg, var(--color-background, #fff), var(--color-tone-7, #f8f8f8));
		box-shadow: 
			0 4px 6px rgba(0, 0, 0, 0.04),
			0 10px 20px rgba(0, 0, 0, 0.06),
			inset 0 1px 0 rgba(255, 255, 255, 0.8);
		padding: 18px;
		margin-bottom: 14px;
		position: relative;
		overflow: hidden;
	}

	.card::before {
		content: '';
		position: absolute;
		top: 0;
		left: 0;
		right: 0;
		height: 4px;
		background: linear-gradient(90deg, var(--color-correct, #6aaa64), #4ade80);
		border-radius: 20px 20px 0 0;
	}

	.card-label {
		font-size: 0.85rem;
		letter-spacing: 0.12em;
		text-transform: uppercase;
		font-weight: 900;
		color: var(--color-tone-1, #333);
		margin-bottom: 14px;
		margin-top: 4px;
		display: flex;
		align-items: center;
		gap: 8px;
	}

	.card-label::before {
		content: '';
		width: 4px;
		height: 18px;
		background: linear-gradient(180deg, var(--color-correct, #6aaa64), #4ade80);
		border-radius: 2px;
	}

	.segmented {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: 8px;
		padding: 6px;
		border-radius: 999px;
		background: var(--color-tone-6, #ececec);
		border: 1px solid var(--color-tone-4, rgba(0, 0, 0, 0.12));
	}

	.segmented button {
		border: 0;
		border-radius: 999px;
		padding: 12px 14px;
		font-weight: 900;
		cursor: pointer;
		background: transparent;
		color: inherit;
		opacity: 0.72;
		transition:
			background 0.14s ease,
			box-shadow 0.14s ease,
			opacity 0.14s ease,
			transform 0.12s ease;
	}

	.segmented button.active {
		background: var(--color-background, #fff);
		opacity: 1;
		box-shadow: 0 10px 18px rgba(0, 0, 0, 0.1);
	}

	.segmented button:hover {
		opacity: 0.9;
	}

	.segmented button:active {
		transform: translateY(1px);
	}

	.note {
		margin-top: 14px;
		border-radius: 16px;
		border: 1px solid rgba(106, 170, 100, 0.35);
		background: rgba(106, 170, 100, 0.1);
		padding: 14px;
	}

	.note-title {
		font-weight: 900;
		letter-spacing: 0.01em;
		margin-bottom: 8px;
		font-size: 1.2rem;
	}

	.note-text {
		opacity: 0.85;
		line-height: 1.35;
		font-size: 0.9rem;
	}

	.field {
		margin-top: 16px;
	}

	.field-label {
		font-size: 0.7rem;
		font-weight: 900;
		letter-spacing: 0.08em;
		text-transform: uppercase;
		color: var(--color-tone-2, #555);
		margin-bottom: 10px;
	}

	select {
		width: 100%;
		padding: 14px 16px;
		border-radius: 14px;
		border: 2px solid transparent;
		background: 
			linear-gradient(var(--color-background, #fff), var(--color-background, #fff)) padding-box,
			linear-gradient(135deg, rgba(106, 170, 100, 0.4), rgba(74, 222, 128, 0.4)) border-box;
		color: inherit;
		font-weight: 700;
		font-size: 0.9rem;
		outline: none;
		cursor: pointer;
		transition: all 0.25s ease;
		box-shadow: 0 4px 12px rgba(106, 170, 100, 0.1);
	}

	select:hover {
		background: 
			linear-gradient(var(--color-background, #fff), var(--color-background, #fff)) padding-box,
			linear-gradient(135deg, rgba(106, 170, 100, 0.6), rgba(74, 222, 128, 0.6)) border-box;
		box-shadow: 0 6px 16px rgba(106, 170, 100, 0.18);
	}

	select:focus {
		background: 
			linear-gradient(var(--color-background, #fff), var(--color-background, #fff)) padding-box,
			linear-gradient(135deg, var(--color-correct, #6aaa64), #4ade80) border-box;
		box-shadow: 0 8px 20px rgba(106, 170, 100, 0.22);
	}

	.target-row {
		display: grid;
		grid-template-columns: 1fr auto;
		gap: 10px;
		align-items: center;
	}

	.target-input {
		width: 100%;
		padding: 12px 14px;
		border-radius: 14px;
		border: 2px solid transparent;
		background: 
			linear-gradient(var(--color-tone-7, #f8f8f8), var(--color-tone-7, #f8f8f8)) padding-box,
			linear-gradient(135deg, rgba(106, 170, 100, 0.3), rgba(74, 222, 128, 0.3)) border-box;
		color: inherit;
		font-weight: 900;
		letter-spacing: 0.16em;
		text-transform: uppercase;
		outline: none;
		transition: all 0.25s ease;
	}

	.target-input:focus {
		background: 
			linear-gradient(var(--color-background, #fff), var(--color-background, #fff)) padding-box,
			linear-gradient(135deg, var(--color-correct, #6aaa64), #4ade80) border-box;
	}

	.icon-btn {
		width: 48px;
		height: 48px;
		border-radius: 14px;
		border: 2px solid transparent;
		background: 
			linear-gradient(var(--color-background, #fff), var(--color-background, #fff)) padding-box,
			linear-gradient(135deg, rgba(106, 170, 100, 0.4), rgba(74, 222, 128, 0.4)) border-box;
		cursor: pointer;
		display: inline-flex;
		align-items: center;
		justify-content: center;
		font-size: 1.2rem;
		box-shadow: 0 4px 12px rgba(106, 170, 100, 0.15);
		transition: all 0.25s ease;
	}

	.icon-btn:hover {
		background: 
			linear-gradient(var(--color-background, #fff), var(--color-background, #fff)) padding-box,
			linear-gradient(135deg, var(--color-correct, #6aaa64), #4ade80) border-box;
		box-shadow: 0 6px 18px rgba(106, 170, 100, 0.25);
		transform: translateY(-2px);
	}

	.icon-btn:active {
		transform: translateY(0);
	}

	.primary {
		width: 100%;
		border: 0;
		border-radius: 14px;
		padding: 16px 14px;
		font-size: 0.9rem;
		font-weight: 900;
		letter-spacing: 0.1em;
		text-transform: uppercase;
		cursor: pointer;
		color: #fff;
		background: linear-gradient(135deg, #22c55e, #16a34a);
		box-shadow: 0 6px 20px rgba(34, 197, 94, 0.35);
		transition: all 0.25s ease;
		position: relative;
		overflow: hidden;
		display: flex;
		align-items: center;
		justify-content: center;
		gap: 8px;
	}

	.primary::before {
		content: '';
		position: absolute;
		top: 0;
		left: -100%;
		width: 100%;
		height: 100%;
		background: linear-gradient(90deg, transparent, rgba(255,255,255,0.25), transparent);
		transition: left 0.5s ease;
	}

	.primary:hover::before {
		left: 100%;
	}

	.primary:hover {
		transform: translateY(-2px);
		box-shadow: 0 10px 28px rgba(34, 197, 94, 0.45);
	}

	.primary:active {
		transform: translateY(0);
	}

	.game-stage {
		height: 100%;
		width: 100%;
		display: flex;
		align-items: stretch;
		justify-content: center;
		padding: 12px;
		box-sizing: border-box;
		overflow: auto;
	}

	.game-zoom {
		--gameScale: 0.95;
		width: min(780px, 100%);
		margin: 0 auto;
		transform: scale(var(--gameScale));
		transform-origin: top center;
		padding-bottom: 40px;
	}

	.right-head {
		gap: 10px;
	}

	.eye-btn {
		border: 1px solid var(--color-tone-4, rgba(0, 0, 0, 0.12));
		background: var(--color-background, #fff);
		border-radius: 12px;
		padding: 8px 10px;
		cursor: pointer;
		box-shadow: 0 10px 18px rgba(0, 0, 0, 0.06);
	}

	.eye-btn.off {
		opacity: 0.75;
	}

	.stats-grid {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: 10px;
		margin-bottom: 12px;
	}

	.stat {
		border-radius: 14px;
		border: 1px solid var(--color-tone-4, rgba(0, 0, 0, 0.12));
		background: var(--color-background, #fff);
		padding: 10px 10px;
	}

	.stat-label {
		font-size: 0.7rem;
		opacity: 0.72;
		letter-spacing: 0.08em;
		text-transform: uppercase;
		font-weight: 900;
		margin-bottom: 6px;
	}

	.stat-value {
		font-size: 1.05rem;
		font-weight: 1000;
		letter-spacing: 0.02em;
	}

	.blur {
		filter: blur(3px);
		opacity: 0.65;
		user-select: none;
		pointer-events: none;
	}

	.section-title {
		margin: 12px 0 10px 0;
		font-size: 0.78rem;
		letter-spacing: 0.1em;
		text-transform: uppercase;
		font-weight: 1000;
		opacity: 0.7;
	}

	.rank-list {
		list-style: none;
		padding: 0;
		margin: 0;
		display: grid;
		gap: 10px;
	}

	.rank-item {
		display: grid;
		grid-template-columns: 84px 1fr 44px;
		gap: 10px;
		align-items: center;
		background: var(--color-background, #fff);
		border: 1px solid var(--color-tone-4, rgba(0, 0, 0, 0.12));
		border-radius: 14px;
		padding: 10px 10px;
		box-shadow: 0 10px 18px rgba(0, 0, 0, 0.05);
		transition: all 0.2s ease;
	}

	.rank-item.clickable {
		cursor: pointer;
	}

	.rank-item.clickable:hover {
		transform: translateY(-2px);
		box-shadow: 0 12px 24px rgba(34, 197, 94, 0.2);
		border-color: #22c55e;
		background: linear-gradient(135deg, #f0fdf4, #dcfce7);
	}

	.rank-item.clickable:active {
		transform: translateY(0);
	}

	.rank-word {
		font-weight: 1000;
		letter-spacing: 0.12em;
		text-transform: uppercase;
	}

	.rank-bar {
		height: 10px;
		background: var(--color-tone-7, #f1f1f1);
		border-radius: 999px;
		overflow: hidden;
		border: 1px solid var(--color-tone-4, rgba(0, 0, 0, 0.08));
	}

	.rank-fill {
		height: 100%;
		background: linear-gradient(90deg, #22c55e 0%, #16a34a 100%);
		border-radius: 999px;
	}

	.rank-score {
		text-align: right;
		font-weight: 1000;
		opacity: 0.72;
	}

	.veil {
		margin-top: 12px;
		padding: 12px 12px;
		border-radius: 14px;
		border: 1px dashed var(--color-tone-4, rgba(0, 0, 0, 0.18));
		font-weight: 900;
		text-align: center;
		opacity: 0.85;
	}

	.btn-row {
		margin-top: 12px;
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: 10px;
	}

	.primary,
	.secondary {
		height: 44px;
		display: flex;
		align-items: center;
		justify-content: center;
		padding: 0 14px;
		border-radius: 12px;
		font-weight: 800;
		font-size: 0.8rem;
		letter-spacing: 0.08em;
		text-transform: uppercase;
	}

	.secondary {
		border: none;
		background: linear-gradient(135deg, #f3f4f6, #e5e7eb);
		color: #374151;
		box-shadow: 0 4px 14px rgba(0, 0, 0, 0.08);
		cursor: pointer;
		transition: all 0.25s ease;
		position: relative;
		overflow: hidden;
	}

	.secondary::before {
		content: '';
		position: absolute;
		top: 0;
		left: -100%;
		width: 100%;
		height: 100%;
		background: linear-gradient(90deg, transparent, rgba(255,255,255,0.6), transparent);
		transition: left 0.5s ease;
	}

	.secondary:hover::before {
		left: 100%;
	}

	.secondary:hover {
		transform: translateY(-2px);
		box-shadow: 0 8px 20px rgba(0, 0, 0, 0.12);
		background: linear-gradient(135deg, #ffffff, #f3f4f6);
	}

	.secondary:active { 
		transform: translateY(0); 
	}

	.btn-icon {
		font-size: 0.75rem;
	}


	@media (max-width: 1360px) {
		.shell {
			grid-template-columns: 320px minmax(600px, 1fr) 320px;
		}
	}
</style>
