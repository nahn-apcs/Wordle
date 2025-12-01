<script lang="ts">
	import { onDestroy, onMount } from "svelte";
	import { Chart, registerables } from "chart.js";

	Chart.register(...registerables);

	export let game: any;
	export let dictionary: string[] = [];

	type CellState = "correct" | "present" | "absent" | "unknown";
	type RowFB = { guess: string; fb: CellState[] };

	type Point = {
		key: string;
		guess: string;
		remaining: number;
		remainingPct: number;
	};

	const MAX_BARS = 6;

	let points: Point[] = [];
	let lastSig = "";
	let timer: any = null;
	let initialCount = 0;
	let canvas: HTMLCanvasElement;
	let chart: Chart | null = null;

	// Export current remaining count for parent to use
	export let currentRemaining: number = 0;
	$: currentRemaining = points.length > 0 ? points[points.length - 1].remaining : initialCount;

	function emojiToState(emoji: string): CellState {
		if (emoji === "🟩") return "correct";
		if (emoji === "🟨") return "present";
		if (emoji === "⬛") return "absent";
		return "unknown";
	}

	function extractCompletedRows(board: any): RowFB[] {
		const rows: RowFB[] = [];
		if (!board || !board.words || !board.state) return rows;

		for (let i = 0; i < board.words.length; i++) {
			const word = board.words[i];
			const stateRow = board.state[i];

			if (!word || word.length !== 5) continue;
			if (!stateRow || stateRow.every((s: string) => s === "🔳")) continue;

			const fb: CellState[] = stateRow.map((emoji: string) => emojiToState(emoji));
			if (fb.some((x) => x === "unknown")) continue;

			rows.push({ guess: word.toUpperCase(), fb });
		}
		return rows;
	}

	function boardSignature(board: any): string {
		const rows = extractCompletedRows(board);
		return rows.map((r) => `${r.guess}:${r.fb.join("")}`).join("|");
	}

	function evalFeedback(candidate: string, guess: string): CellState[] {
		const c = candidate.toUpperCase().split("");
		const g = guess.toUpperCase().split("");
		const res: CellState[] = Array(5).fill("absent");
		const used = Array(5).fill(false);

		for (let i = 0; i < 5; i++) {
			if (g[i] === c[i]) {
				res[i] = "correct";
				used[i] = true;
			}
		}

		for (let i = 0; i < 5; i++) {
			if (res[i] === "correct") continue;
			let found = -1;
			for (let j = 0; j < 5; j++) {
				if (!used[j] && c[j] === g[i]) {
					found = j;
					break;
				}
			}
			if (found !== -1) {
				used[found] = true;
				res[i] = "present";
			}
		}
		return res;
	}

	function matchesAll(candidate: string, rows: RowFB[]): boolean {
		for (const r of rows) {
			const fb = evalFeedback(candidate, r.guess);
			for (let i = 0; i < 5; i++) {
				if (fb[i] !== r.fb[i]) return false;
			}
		}
		return true;
	}

	function recompute(board: any) {
		if (!dictionary?.length) {
			points = [];
			initialCount = 0;
			return;
		}

		const rows = extractCompletedRows(board);
		const dictUpper = dictionary.map((w) => w.toUpperCase());
		const initial = dictUpper.length;
		initialCount = initial;

		if (rows.length === 0) {
			points = [];
			updateChart();
			return;
		}

		let cur: string[] = dictUpper;
		const out: Point[] = [];

		for (let i = 0; i < rows.length; i++) {
			const sub = rows.slice(0, i + 1);
			cur = cur.filter((w) => matchesAll(w, sub));

			const remaining = cur.length;
			const remainingPct = Math.max(0, Math.min(100, (remaining / initial) * 100));

			out.push({
				key: `${rows[i].guess}-${i}`,
				guess: rows[i].guess,
				remaining,
				remainingPct,
			});
		}

		points = out.slice(Math.max(0, out.length - MAX_BARS));
		updateChart();
	}

	function updateChart() {
		if (!chart) return;

		const labels = points.map((p) => p.guess);
		const data = points.map((p) => p.remaining);

		chart.data.labels = labels;
		chart.data.datasets[0].data = data;
		chart.update();
	}

	function createChart() {
		if (!canvas) return;

		const ctx = canvas.getContext("2d");
		if (!ctx) return;

		// Gradient for area fill - green theme
		const areaGradient = ctx.createLinearGradient(0, 0, 0, 180);
		areaGradient.addColorStop(0, "rgba(34, 197, 94, 0.35)");
		areaGradient.addColorStop(0.5, "rgba(34, 197, 94, 0.12)");
		areaGradient.addColorStop(1, "rgba(34, 197, 94, 0.02)");

		chart = new Chart(ctx, {
			type: "line",
			data: {
				labels: [],
				datasets: [
					{
						label: "Remaining Words",
						data: [],
						borderColor: "#22c55e",
						backgroundColor: areaGradient,
						borderWidth: 2,
						pointBackgroundColor: "#fff",
						pointBorderColor: "#22c55e",
						pointBorderWidth: 2.5,
						pointRadius: 4,
						pointHoverRadius: 6,
						pointHoverBackgroundColor: "#22c55e",
						pointHoverBorderColor: "#fff",
						pointHoverBorderWidth: 2,
						tension: 0.35,
						fill: true,
						order: 1,
					},
				],
			},
			options: {
				responsive: true,
				maintainAspectRatio: false,
				layout: {
					padding: {
						top: 15,
						right: 15,
						bottom: 5,
						left: 5,
					},
				},
				clip: false,
				interaction: {
					intersect: false,
					mode: "index",
				},
				plugins: {
					legend: {
						display: false,
					},
					tooltip: {
						backgroundColor: "#1e293b",
						titleFont: {
							size: 14,
							weight: "bold",
						},
						bodyFont: {
							size: 13,
						},
						padding: 14,
						cornerRadius: 12,
						displayColors: false,
						callbacks: {
							title: (items) => {
								const label = items[0]?.label || "";
								return `📝 ${label}`;
							},
							label: (item) => {
								const val = item.raw as number;
								const pct = initialCount > 0 ? ((val / initialCount) * 100).toFixed(2) : 0;
								return [
									`Remaining: ${val.toLocaleString()} words`,
									`Coverage: ${pct}%`,
								];
							},
							afterLabel: (item) => {
								const idx = item.dataIndex;
								if (idx === 0) return `Eliminated: ${(initialCount - (item.raw as number)).toLocaleString()}`;
								const prev = points[idx - 1]?.remaining || initialCount;
								const curr = item.raw as number;
								const eliminated = prev - curr;
								return `Eliminated: ${eliminated.toLocaleString()} (${((eliminated / prev) * 100).toFixed(1)}%)`;
							},
						},
					},
				},
				scales: {
					x: {
						grid: {
							display: false,
						},
						ticks: {
							font: {
								size: 11,
								weight: "bold",
							},
							color: "#475569",
						},
						border: {
							display: false,
						},
					},
					y: {
						type: "logarithmic",
						min: 1,
						grid: {
							color: "rgba(0, 0, 0, 0.04)",
							lineWidth: 1,
						},
						ticks: {
							font: {
								size: 9,
								weight: "600",
							},
							color: "#64748b",
							maxTicksLimit: 6,
							callback: (val) => {
								if (typeof val === "number") {
									const allowed = [1, 10, 100, 1000, 10000, 100000];
									if (!allowed.includes(val)) return "";
									if (val >= 1000) return (val / 1000).toFixed(0) + "k";
									return val.toFixed(0);
								}
								return "";
							},
						},
						border: {
							display: false,
						},
					},
				},
				animation: {
					duration: 500,
					easing: "easeOutQuart",
				},
			},
		});
	}

	onMount(() => {
		createChart();

		if (game?.board) {
			recompute(game.board);
			lastSig = boardSignature(game.board);
		}

		timer = setInterval(() => {
			const b = game?.board;
			if (!b) return;

			const sig = boardSignature(b);
			if (sig !== lastSig) {
				lastSig = sig;
				recompute(b);
			}

			if (!sig && points.length > 0) {
				points = [];
				lastSig = "";
				updateChart();
			}
		}, 100);
	});

	onDestroy(() => {
		if (timer) clearInterval(timer);
		if (chart) chart.destroy();
	});
</script>

<div class="gp-card">
	<div class="card-label">📉 GUESS PROGRESS</div>

	<div class="stats-row">
		<div class="stat-box">
			<div class="stat-num">{initialCount.toLocaleString()}</div>
			<div class="stat-lbl">Start</div>
		</div>
		<div class="arrow">→</div>
		<div class="stat-box">
			<div class="stat-num">{points.length ? points[points.length - 1].remaining.toLocaleString() : "—"}</div>
			<div class="stat-lbl">Now</div>
		</div>
	</div>

	<div class="gp-content">
		<div class="chart-wrap" class:hidden={points.length === 0}>
			<canvas bind:this={canvas}></canvas>
		</div>

		{#if points.length === 0}
			<div class="empty-state">
				<div class="empty-icon">🎯</div>
				<div class="empty-msg">Make your first guess!</div>
				<div class="empty-hint">Chart updates in real-time</div>
			</div>
		{:else}
			<div class="legend-bar">
				<div class="legend-item">
					<div class="dot green"></div>
					<span>Remaining words (log scale)</span>
				</div>
			</div>
		{/if}
	</div>
</div>

<style>
	.gp-card {
		border-radius: 20px;
		background: linear-gradient(145deg, var(--color-background, #fff), var(--color-tone-7, #f8f8f8));
		border: none;
		box-shadow: 
			0 4px 6px rgba(0, 0, 0, 0.04),
			0 10px 20px rgba(0, 0, 0, 0.06),
			inset 0 1px 0 rgba(255, 255, 255, 0.8);
		padding: 18px;
		margin-top: 14px;
		position: relative;
		overflow: hidden;
	}

	.gp-card::before {
		content: '';
		position: absolute;
		top: 0;
		left: 0;
		right: 0;
		height: 4px;
		background: linear-gradient(90deg, #3b82f6, #8b5cf6);
		border-radius: 20px 20px 0 0;
	}

	.card-label {
		font-size: 0.85rem;
		font-weight: 900;
		letter-spacing: 0.12em;
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
		background: linear-gradient(180deg, #3b82f6, #8b5cf6);
		border-radius: 2px;
	}

	.stats-row {
		display: flex;
		align-items: center;
		gap: 10px;
		margin-bottom: 16px;
	}

	.stat-box {
		flex: 1;
		background: linear-gradient(135deg, rgba(59, 130, 246, 0.08), rgba(139, 92, 246, 0.08));
		border-radius: 16px;
		padding: 16px 14px;
		border: 1px solid rgba(59, 130, 246, 0.15);
		text-align: center;
		transition: all 0.2s ease;
	}

	.stat-box:hover {
		background: linear-gradient(135deg, rgba(59, 130, 246, 0.12), rgba(139, 92, 246, 0.12));
		transform: translateY(-2px);
		box-shadow: 0 4px 12px rgba(59, 130, 246, 0.15);
	}

	.stat-num {
		font-size: 1.6rem;
		font-weight: 900;
		background: linear-gradient(135deg, #3b82f6, #8b5cf6);
		-webkit-background-clip: text;
		-webkit-text-fill-color: transparent;
		background-clip: text;
		margin-bottom: 4px;
	}

	.stat-lbl {
		font-size: 0.65rem;
		font-weight: 800;
		text-transform: uppercase;
		letter-spacing: 0.08em;
		color: var(--color-tone-2, #666);
	}

	.arrow {
		font-size: 1.2rem;
		background: linear-gradient(135deg, #3b82f6, #8b5cf6);
		-webkit-background-clip: text;
		-webkit-text-fill-color: transparent;
		background-clip: text;
		font-weight: bold;
	}

	.gp-content {
		/* padding removed, already in card */
	}

	.empty-state {
		height: 140px;
		display: flex;
		flex-direction: column;
		align-items: center;
		justify-content: center;
		gap: 6px;
		background: var(--color-tone-7, #fff);
		border-radius: 12px;
		border: 1px dashed var(--color-tone-4, rgba(0, 0, 0, 0.15));
	}

	.empty-icon {
		font-size: 2rem;
		opacity: 0.4;
	}

	.empty-msg {
		font-size: 0.9rem;
		font-weight: 800;
		color: var(--color-tone-2, #555);
	}

	.empty-hint {
		font-size: 0.7rem;
		color: var(--color-tone-3, #888);
		font-weight: 600;
	}

	.chart-wrap {
		height: 180px;
		background: var(--color-tone-7, #fff);
		border-radius: 12px;
		border: 1px solid var(--color-tone-4, rgba(0, 0, 0, 0.1));
		padding: 12px;
		margin-bottom: 10px;
	}

	.chart-wrap.hidden {
		display: none;
	}

	.legend-bar {
		display: flex;
		align-items: center;
		justify-content: center;
		gap: 8px;
		padding: 8px;
		background: var(--color-tone-7, #fff);
		border-radius: 8px;
		border: 1px solid var(--color-tone-4, rgba(0, 0, 0, 0.1));
	}

	.legend-item {
		display: flex;
		align-items: center;
		gap: 6px;
		font-size: 0.65rem;
		font-weight: 700;
		color: var(--color-tone-2, #555);
	}

	.dot {
		width: 8px;
		height: 8px;
		border-radius: 50%;
	}

	.dot.green {
		background: var(--color-correct, #6aaa64);
	}
</style>
