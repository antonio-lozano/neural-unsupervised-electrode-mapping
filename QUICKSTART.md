# RF Demo Quickstart

- **Purpose**: load a curated LFP bundle, compute correlation distances, align MDS + tuned UMAP embeddings to cortex, and project them into visual-field space.
- **Environment (uv)**: uv sync then uv run python scripts/run_demo.py\n- **Key modules**: `rf_demo.data_loader` (NPZ bundle I/O), `rf_demo.pipeline` (distance + embedding + projection), `rf_demo.visualization` (figures).
- **Data extract**: `uv run python scripts/extract_demo_data.py --monkey monkey_L --freq-band LFP` (swap to `monkey_A` for the second subject) writes `data/demo_monkey_l_lfp_15s.npz` using the original loaders.
- **Pipeline run**: `uv run python scripts/run_demo.py` emits `results/<monkey>_<band>/` containing the distance heatmap, cortex alignment, visual-field projection, and `metrics.json` (append `--seed 31415` for reproducibility).
- **Notebook**: optional walkthroughs live in `notebooks/` and import the same package with `import rf_demo`.

Note: Pre-generated demo NPZ bundles are included under `data`, so users do not need to run the extraction step. The extraction script is only for maintainers who want to refresh the bundles from the full dataset.

