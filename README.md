# RF Visual Field Demo

This repository provides a lightweight, reviewer-friendly demonstration of the receptive‑field reconstruction pipeline. It ships with pre‑generated demo bundles (NPZ files), so users do not need to regenerate data.

- Load a small LFP sample alongside ground‑truth cortical and visual‑field coordinates
- Compute correlation‑based distances and run an MDS embedding
- Tune UMAP hyperparameters using the MDS result as a surrogate target
- Project recovered maps back onto cortex space and into the visual field
- Visualise the outcome through reproducible scripts and notebooks

## Repository layout

```
data/                  # pre-generated .npz bundles included in the repo
notebooks/             # optional exploratory walkthroughs (Jupyter)
scripts/               # command-line entry points
src/                   # reusable demo package
tests/                 # smoke tests for CI
README.md              # you are here
pyproject.toml         # project metadata
requirements.txt       # pinned runtime dependencies
```

## Getting started (uv)

Windows (PowerShell):

```powershell
cd neural-unsupervised-electrode-mapping
# Install uv if needed: pipx install uv   (see https://docs.astral.sh/uv/)
uv sync
# No data extraction required: demo NPZ bundles are included in data/
uv run python -m ipykernel install --user --name neus-elec-map
uv run python scripts\run_demo.py --seed 31415
# Note: using --seed ensures reproducibility but can slow UMAP; omit for fastest run
```

Linux/macOS (bash):

```bash
cd neural-unsupervised-electrode-mapping
uv sync
# Use forward slashes on Unix shells
uv run python -m ipykernel install --user --name neus-elec-map
uv run python scripts/run_demo.py --seed 31415
# Omit --seed for the fastest run
```

The command‑line entry downloads nothing and relies solely on the curated assets under `data/`.

### Run the notebook

Two options:

- Recommended (no environment changes):
  - `uvx jupyter lab` (or `uvx jupyter notebook`)
  - Open `notebooks/rf_demo_notebook_demo.ipynb` and run all cells

- Alternatively (install into the project env):
  - `uv run python -m pip install notebook`
  - `uv run python -m notebook notebooks/rf_demo_notebook_demo.ipynb`

## Regenerating the demo data (optional, for maintainers)

If you need to refresh the demo NPZ bundles from the full dataset, use:

Windows (PowerShell):

```powershell
uv run python scripts\extract_demo_data.py --monkey monkey_L --freq-band LFP
# swap to monkey_A and other bands if needed
```

Linux/macOS (bash):

```bash
uv run python scripts/extract_demo_data.py --monkey monkey_L --freq-band LFP
# swap to monkey_A and other bands if needed
```

This step requires the full dataset under `data/` and the original project loaders. End users do not need this.

## Relationship with the full codebase

- The demo imports helper routines copied from the original project only where needed.
- Loader logic is a thin wrapper around the original utilities for maintainers (end users use bundled NPZs).
- Visualisation styles reuse the original Matplotlib setup to match figures in the manuscript.

This separation lets the core repository evolve without breaking the reviewer demo while maximising code reuse.
