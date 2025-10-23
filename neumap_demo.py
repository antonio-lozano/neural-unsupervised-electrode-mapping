# %%
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

# Ensure repo modules are importable when running this file directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO_SRC = REPO_ROOT / "demo" / "src"
for path in (REPO_ROOT, DEMO_SRC):
    normalized = str(path)
    if normalized not in sys.path:
        sys.path.append(normalized)


# %%
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import euclidean_distances

from demo.src.rf_demo.config import FREQ_BANDS, canonical_alias
from demo.src.rf_demo.data_loader import DemoBundle, WindowType, load_demo_bundle
from demo.src.rf_demo.pipeline import (
    align_to_cortex,
    compute_metrics,
    correlation_distance_matrix,
    fast_umap_grid,
    run_mds,
    run_umap_embedding,
)


def _load_repo_utils_module():
    module_name = "_repo_utils"
    if module_name in sys.modules:
        return sys.modules[module_name]
    utils_path = REPO_ROOT / "code" / "utils.py"
    spec = importlib.util.spec_from_file_location(module_name, utils_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load utils.py from {utils_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_repo_utils = _load_repo_utils_module()
calculate_corr_of_distances = _repo_utils.calculate_corr_of_distances
rescale_procrustes_map = _repo_utils.rescale_procrustes_map
scipy_Antonio_procrustes = _repo_utils.scipy_Antonio_procrustes


# %%
# Configuration for the demo run.
MONKEY_ALIAS = "monkey_L"
WINDOW: WindowType = "15s"
_PREFERRED_BANDS = ("LFP", "gamma", "highGamma")
TARGET_BANDS: Tuple[str, ...] = tuple(band for band in FREQ_BANDS if band in _PREFERRED_BANDS)
if not TARGET_BANDS:
    TARGET_BANDS = (FREQ_BANDS[0],)
SELECTION_MODE = "rmse"  # choices: "rmse", "iedc", "combo"
RANDOM_STATE: int | None = 42
MAX_GRID_COMBOS = 6  # keep the sweep lightweight for interactive use

monkey_canonical = canonical_alias(MONKEY_ALIAS)


# %%
# Load LFP + anatomical references for the requested frequency bands.
# Falls back to demo NPZ bundles if the full dataset is unavailable.
bundles: Dict[str, DemoBundle] = {}
for band in TARGET_BANDS:
    try:
        bundle = load_demo_bundle(monkey=monkey_canonical, freq_band=band, window=WINDOW)
    except FileNotFoundError as exc:
        print(f"Skipping {band}: {exc}")
        continue
    bundles[band] = bundle
    print(f"Loaded {band:<9} -> LFP shape {bundle.lfp.shape}")

if not bundles:
    raise RuntimeError("No frequency bands could be loaded. Check that demo data has been extracted.")

if "LFP" not in bundles:
    reference_band = next(iter(bundles))
    print(f"Reference band defaulting to {reference_band}")
else:
    reference_band = "LFP"

reference_bundle = bundles[reference_band]
monkey_alias = reference_bundle.monkey


# %%
# Build an MDS baseline from the reference band to serve as the surrogate target.
distance_matrix = correlation_distance_matrix(reference_bundle.lfp)
mds_embedding = run_mds(distance_matrix, random_state=RANDOM_STATE)
mds_aligned = align_to_cortex(reference_bundle.utah_mm, mds_embedding)
mds_metrics = compute_metrics(monkey_alias, reference_bundle.utah_mm, mds_aligned)
mds_distances = euclidean_distances(mds_embedding)

print("MDS metrics (cortex vs recovered)")
for name, value in mds_metrics.items():
    print(f"  {name:>14}: {value:0.4f}")


# %%
# Construct a lightweight UMAP grid (subset of the fast grid for brevity).
full_grid = fast_umap_grid(reference_bundle.lfp.shape[0])
pruned_grid: List[Dict[str, Any]] = []
for params in full_grid:
    if len(pruned_grid) >= MAX_GRID_COMBOS:
        break
    if params["n_neighbors"] >= reference_bundle.lfp.shape[0]:
        continue
    pruned_grid.append(params)

if not pruned_grid:
    raise RuntimeError("UMAP grid is empty; adjust MAX_GRID_COMBOS or the neighbor bounds.")

print("Evaluating UMAP grid:")
for item in pruned_grid:
    print(f"  n_neighbors={item['n_neighbors']:>3}, min_dist={item['min_dist']}")


# %%
# Sweep the grid across the chosen frequency bands.
alpha = 3.0  # weighting for combo mode
candidate_scores: List[Dict[str, Any]] = []
best_candidate: Dict[str, Any] | None = None

for freq_band, bundle in bundles.items():
    for params in pruned_grid:
        n_neighbors = min(int(params["n_neighbors"]), bundle.lfp.shape[0] - 1)
        n_neighbors = max(n_neighbors, 2)
        embedding = run_umap_embedding(bundle.lfp, n_neighbors, params["min_dist"], RANDOM_STATE)

        umap_distances = euclidean_distances(embedding)
        surrogate_corr = float(calculate_corr_of_distances(mds_embedding, embedding))
        surrogate_rmse = float(
            np.sqrt(mean_squared_error(mds_distances.ravel(), umap_distances.ravel()))
        )

        if SELECTION_MODE == "rmse":
            score = surrogate_rmse
        elif SELECTION_MODE == "iedc":
            score = 1.0 - surrogate_corr
        elif SELECTION_MODE == "combo":
            score = surrogate_rmse + alpha * (1.0 - surrogate_corr)
        else:
            raise ValueError(f"Unknown SELECTION_MODE: {SELECTION_MODE}")

        candidate = {
            "freq_band": freq_band,
            "n_neighbors": n_neighbors,
            "min_dist": params["min_dist"],
            "surrogate_corr": surrogate_corr,
            "surrogate_rmse": surrogate_rmse,
            "score": score,
            "embedding": embedding,
            "bundle": bundle,
        }
        candidate_scores.append(candidate)

        if best_candidate is None or score < best_candidate["score"]:
            best_candidate = candidate

if best_candidate is None:
    raise RuntimeError("UMAP sweep did not produce any embeddings.")

print(
    f"Best candidate -> freq: {best_candidate['freq_band']}, "
    f"n_neighbors: {best_candidate['n_neighbors']}, min_dist: {best_candidate['min_dist']}"
)
print(
    f"  surrogate_corr={best_candidate['surrogate_corr']:.4f}, "
    f"surrogate_rmse={best_candidate['surrogate_rmse']:.4f}, score={best_candidate['score']:.4f}"
)


# %%
# Transfer the winning embedding to cortex space and compute evaluation metrics.
best_embedding = best_candidate["embedding"]
try:
    _, _, _, R_m, s_m, norm1_m, norm2_m, mean1_m, mean2_m = scipy_Antonio_procrustes(
        mds_aligned, best_embedding
    )
    umap_aligned_mds = rescale_procrustes_map(best_embedding, R_m, s_m, norm1_m, norm2_m, mean1_m, mean2_m)
except Exception as exc:
    print(f"MDS alignment failed ({exc}); falling back to direct cortex alignment.")
    umap_aligned_mds = align_to_cortex(reference_bundle.utah_mm, best_embedding)

umap_metrics = compute_metrics(monkey_alias, reference_bundle.utah_mm, umap_aligned_mds)
print("UMAP metrics via MDS transfer")
for name, value in umap_metrics.items():
    print(f"  {name:>14}: {value:0.4f}")

try:
    _, _, _, R_gt, s_gt, norm1_gt, norm2_gt, mean1_gt, mean2_gt = scipy_Antonio_procrustes(
        reference_bundle.utah_mm, best_embedding
    )
    umap_aligned_gt = rescale_procrustes_map(
        best_embedding, R_gt, s_gt, norm1_gt, norm2_gt, mean1_gt, mean2_gt
    )
    umap_gt_metrics = compute_metrics(monkey_alias, reference_bundle.utah_mm, umap_aligned_gt)
    print("UMAP metrics via direct cortex alignment")
    for name, value in umap_gt_metrics.items():
        print(f"  {name:>14}: {value:0.4f}")
except Exception as exc:
    print(f"Direct cortex alignment failed: {exc}")
    umap_gt_metrics = None


# %%
# Inspect the top-scoring configurations for quick comparison.
top_candidates = sorted(candidate_scores, key=lambda item: item["score"])[:5]
print("Top parameter sets (sorted by score):")
for idx, candidate in enumerate(top_candidates, start=1):
    print(
        f"  {idx:>2}. freq={candidate['freq_band']:<9} n_neighbors={candidate['n_neighbors']:>3} "
        f"min_dist={candidate['min_dist']:.2f} score={candidate['score']:.4f} "
        f"corr={candidate['surrogate_corr']:.4f} rmse={candidate['surrogate_rmse']:.4f}"
    )
