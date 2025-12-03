<script lang="ts">
	export let onClick = () => {};
</script>

<!-- svelte-ignore a11y-click-events-have-key-events -->
<svg
	xmlns="http://www.w3.org/2000/svg"
	viewBox="0 0 24 24"
	on:click={onClick}
>
	<slot />
</svg>

<style lang="scss">
	svg {
		cursor: pointer;
		width: 24px;
		margin: 0 4px;
		:global(path) {
			fill: var(--icon-color);
		}
	}
</style>
