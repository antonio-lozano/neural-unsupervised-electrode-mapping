# RF Visual Field Demo

This folder provides a lightweight, reviewer-friendly demonstration of the receptive-field reconstruction pipeline. It
ships with pre-generated demo bundles (NPZ files) so users do not need to regenerate data. Reviewers can:

- load a small LFP sample alongside ground-truth cortical and visual field coordinates,
- compute correlation-based distances and run an MDS embedding,
- tune UMAP hyperparameters using the MDS result as a surrogate target,
- project recovered maps back onto cortex space and into the visual field, and
- visualise the outcome through reproducible scripts and notebooks.

## Repository layout

```
/demo
├── data/                  # minimal .npz bundles extracted from the full dataset
├── notebooks/             # optional exploratory walkthroughs (Jupyter)
├── scripts/               # command-line entry points
├── src/                   # reusable demo package
├── tests/                 # smoke tests for CI
├── README.md              # you are here
├── pyproject.toml         # project metadata
└── requirements.txt       # pinned runtime dependencies
```

## Getting started (uv)

```powershell
cd neural-unsupervised-electrode-mapping
# Install uv if needed: pipx install uv   (see https://docs.astral.sh/uv/)
uv sync
# No data extraction required: demo NPZ bundles are included in data/
uv run python scripts\run_demo.py --seed 31415
```

The command line entry downloads nothing and relies solely on the curated assets under `data/`.

> Prefer `uv`? Replace the virtualenv lines with `uv venv` and `uv pip install -r requirements.txt`.

The command line entry downloads nothing and relies solely on the curated assets under `data`.

## Regenerating the demo data (optional, for maintainers)

If you need to refresh the demo NPZ bundles from the full dataset, use:

```powershell
uv run uv run python scripts/extract_demo_data.py --monkey monkey_L --freq-band LFP
# swap to monkey_A and other bands if needed
```
This step requires the full dataset under `data/` and the original project loaders. End users do not need this.

## Relationship with the full codebase

- The demo imports helper routines copied from `code/review_performance_per_frequency_VF.py` for metrics and plotting.
- Loader logic is thin wrappers around `utils_extension.load_monkey_data_band_15seconds` to keep behaviour identical.
- Visualisation styles reuse the original Matplotlib setup to match figures in the manuscript.

This separation lets the core repository evolve without breaking the reviewer demo while maximising code reuse.


