<script lang="ts">
	import { failed, GameState } from "../../../utils";

	export let game: GameState;
	export let distribution: Guesses;

	$: max = Object.entries(distribution).reduce((p, c) => {
		if (!isNaN(Number(c[0]))) return Math.max(c[1], p);
		return p;
	}, 1);
</script>

<h3>guess distribution</h3>
<div class="container">
	{#each Object.entries(distribution) as guess, i (guess[0])}
		{@const g = Number(guess[0])}
		{@const countStr = String(guess[1])}
		{@const countMinWidth = Math.max(countStr.length * 12 + 20, 70)}
		{#if !isNaN(g)}
			<div class="graph">
				<span class="guess">{guess[0]}</span>
				<div
					class="bar"
					class:this={g === game.guesses && !game.active && !failed(game)}
					style="width: {Math.max((guess[1] / max) * 100, 10)}%; min-width: {countMinWidth}px;"
				>
					<span class="count">{countStr}</span>
				</div>
			</div>
		{/if}
	{/each}
</div>

<style>
	.container {
		width: 100%;
		padding: 0 8px;
		box-sizing: border-box;
		display: flex;
		flex-direction: column;
		gap: 4px;
	}
	.graph {
		height: 22px;
		display: flex;
		align-items: center;
		gap: 4px;
	}
	.guess {
		width: 14px;
		text-align: center;
		font-weight: bold;
		flex-shrink: 0;
	}
	.bar {
		background: var(--color-absent);
		border-radius: 4px;
		height: 100%;
		display: flex;
		justify-content: flex-end;
		align-items: center;
		padding: 0 4px;
		box-sizing: border-box;
		overflow: hidden;
	}
	.bar.this {
		background: var(--color-correct);
	}
	.count {
		font-weight: 700;
		font-size: 0.55rem;
		color: white;
		white-space: nowrap;
	}
</style>
