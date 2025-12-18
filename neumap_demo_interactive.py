# %% [markdown]
# Interactive Neumap Demo (Cortex-first alignment)
#
# This interactive script mirrors the pipeline in `neumap_demo_v2.py` but is organized
# into VS Code Python cells so you can run and inspect each stage independently.
#
# Key principles:
# - Align in cortical space (Procrustes to ground truth Utah), then project to visual field.
# - Visual field plots are direct projections (no visual-field alignment to RFs).
# - Visual field axes are fixed to [-10, 10] degrees.
#
# Tip: Use the VS Code "Run Cell" buttons (or Shift+Enter) to run cell by cell.

# %%
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import euclidean_distances
from umap import UMAP

# Ensure repo root is importable
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

# Local utilities
from config_loader import load_config, get_data_paths  # type: ignore
from cortex_model_utils import cortex_to_visual_mapping  # type: ignore
from utils import (  # type: ignore
    calculate_corr_of_distances,
    get_monkey_pixPerMM_utahMax,
    rescale_procrustes_map,
    scipy_Antonio_procrustes,
)
from utils_extension import (  # type: ignore
    load_monkey_data_band,
    load_monkey_data_band_15seconds,
)

# %% [markdown]
# Configuration and constants

# %%
WindowType = Literal["15s", "full"]

FREQ_BANDS: Tuple[str, ...] = ("LFP", "low", "alpha", "beta", "gamma", "highGamma")
LITERATURE_WEDGE_PARAMS: Dict[str, float] = {"a": 0.61, "b": 106.0, "k": 13.6, "alpha": 0.86}
OPTIMIZED_WEDGE_PARAMS: Dict[str, Dict[str, float]] = {
    "monkey_L": {"a": 0.5251, "b": 80.0, "k": 13.64, "alpha": 0.774},
    "monkey_A": {"a": 0.6857, "b": 106.0, "k": 16.69, "alpha": 0.948},
}
PREFERRED_BANDS: Tuple[str, ...] = ("LFP", "gamma", "highGamma", "beta")
STANDARIZE_LFP = False
EXCLUDE_V4 = False
PIXELS_PER_DEG = 25.78  # Conversion factor for RFs from pixels to degrees

@dataclass(frozen=True)
class BandBundle:
    freq_band: str
    lfp: np.ndarray
    utah_mm: np.ndarray
    colors: np.ndarray
    rfs: np.ndarray

# Load configuration
try:
    CONFIG = load_config()
    DATA_PATHS = get_data_paths(CONFIG)
except Exception as e:
    print(f"Warning: Could not load config.yaml ({e}). Using default paths.")
    CONFIG = None
    DATA_PATHS = None

# %% [markdown]
# Helper functions

# %%

def resolve_monkey_name(alias: str) -> Tuple[str, str]:
    normalized = alias.strip()
    lowered = normalized.lower()
    if lowered in {"monkey_l", "monkey_L"}:
        return "monkey_L", "monkey_L"
    if lowered in {"monkey_a", "monkey_A", "monkey_A"}:
        return "monkey_A", "monkey_A"
    raise KeyError(f"Unknown monkey identifier '{alias}'. Expected one of monkey_L, monkey_A, monkey_L, monkey_A.")


def correlation_distance_matrix(lfp: np.ndarray) -> np.ndarray:
    corr = np.corrcoef(lfp)
    corr = np.clip(corr, -0.999, 0.999)
    dist = 1.0 - corr
    np.fill_diagonal(dist, 0.0)
    dist = np.nan_to_num(dist, nan=0.0, posinf=0.0, neginf=0.0)
    return dist.astype(np.float32)


def run_mds(distance_matrix: np.ndarray, random_state: int | None) -> np.ndarray:
    kwargs: Dict[str, Any] = {
        "n_components": 2,
        "dissimilarity": "precomputed",
        "n_init": 4,
        "max_iter": 400,
    }
    if random_state is not None:
        kwargs["random_state"] = random_state
    reducer = MDS(**kwargs)
    return reducer.fit_transform(distance_matrix)


def run_umap_embedding(lfp: np.ndarray, n_neighbors: int, min_dist: float, random_state: int | None) -> np.ndarray:
    pca = PCA(n_components=2, random_state=random_state)
    pca_init = pca.fit_transform(lfp)
    kwargs: Dict[str, Any] = {
        "n_components": 2,
        "metric": "euclidean",
        "n_neighbors": int(n_neighbors),
        "min_dist": float(min_dist),
        "init": pca_init,
    }
    # Don't set random_state to enable parallelism (faster)
    reducer = UMAP(**kwargs)
    return reducer.fit_transform(lfp)


def align_to_cortex(utah_mm: np.ndarray, embedding: np.ndarray) -> np.ndarray:
    _, _, _, R, s, norm1, norm2, mean1, mean2 = scipy_Antonio_procrustes(utah_mm, embedding)
    return rescale_procrustes_map(embedding, R, s, norm1, norm2, mean1, mean2)


def _wedge_params_for_monkey(monkey_alias: str) -> Dict[str, float]:
    return OPTIMIZED_WEDGE_PARAMS.get(monkey_alias, LITERATURE_WEDGE_PARAMS)


def project_to_visual_field(points_mm: np.ndarray, monkey_alias: str) -> np.ndarray:
    """Project cortical positions to visual field using wedge dipole model.
    Returns positions in degrees (N x 2).
    """
    params = _wedge_params_for_monkey(monkey_alias)
    vx, vy = cortex_to_visual_mapping(
        points_mm[:, 0],
        points_mm[:, 1],
        params["a"],
        params["b"],
        params["alpha"],
        params["k"],
    )
    return np.vstack([vx, vy]).T


def compute_metrics(monkey_alias: str, utah_mm: np.ndarray, recovered_mm: np.ndarray) -> Dict[str, float]:
    cortical_corr = float(calculate_corr_of_distances(utah_mm, recovered_mm))
    cortical_rmse = float(np.sqrt(mean_squared_error(utah_mm, recovered_mm)))
    utah_visual = project_to_visual_field(utah_mm, monkey_alias)
    recovered_visual = project_to_visual_field(recovered_mm, monkey_alias)
    visual_corr = float(calculate_corr_of_distances(utah_visual, recovered_visual))
    visual_rmse = float(np.sqrt(mean_squared_error(utah_visual, recovered_visual)))
    return {
        "cortical_corr": cortical_corr,
        "cortical_rmse": cortical_rmse,
        "visual_corr": visual_corr,
        "visual_rmse": visual_rmse,
    }


def _data_paths() -> Tuple[str, str, str, str]:
    if DATA_PATHS is not None:
        base_path = DATA_PATHS['eyes_closed_data']
        utah_path = DATA_PATHS['utah_coordinates']
        channel_map = DATA_PATHS['channel_area_mapping']
        delete_path = DATA_PATHS['deleted_electrodes']
    else:
        base_path = str(REPO_ROOT / "data" / "EYES_CLOSED")
        utah_path = str(REPO_ROOT / "data" / "coordinates_of_electrodes_on_cortex_using_photos_of_arrays")
        channel_map = str(REPO_ROOT / "data" / "channel_area_mapping" / "channel_area_mapping.mat")
        delete_path = str(REPO_ROOT / "data" / "deletedElectrodesDictionary")
    return base_path, utah_path, channel_map, delete_path


# %% [markdown]
# Select subject, window, and frequency band

# %%
monkey_alias_input = "monkey_L"   # e.g., "monkey_L", "monkey_L", "monkey_A", "monkey_A"
window: WindowType = "15s"        # "15s" or "full"
freq_band = "LFP"                 # one of FREQ_BANDS; try "gamma" or "highGamma" as well
random_state: int | None = 42

monkey_alias, monkey_actual = resolve_monkey_name(monkey_alias_input)
pixels_per_mm, utah_max_px = get_monkey_pixPerMM_utahMax(monkey_actual)
if pixels_per_mm is None or utah_max_px is None:
    raise ValueError(f"Missing scaling constants for monkey '{monkey_actual}'.")
pixels_per_mm = float(pixels_per_mm)
utah_max_px = float(utah_max_px)
print({
    "monkey_alias": monkey_alias,
    "monkey_actual": monkey_actual,
    "window": window,
    "freq_band": freq_band,
})

# %% [markdown]
# Load data for the chosen band

# %%
base_path, utah_path, channel_map, delete_path = _data_paths()
loader = load_monkey_data_band_15seconds if window == "15s" else load_monkey_data_band
results = loader(
    monkey_actual,
    STANDARIZE_LFP,
    EXCLUDE_V4,
    base_path,
    utah_path,
    channel_map,
    delete_path,
    freq_band=freq_band,
    load_MUA=False,
    LFP_float16=False,
)

lfp = np.asarray(results[0], dtype=np.float32)
colors = np.asarray(results[5], dtype=np.float32)
rfs_raw = np.asarray(results[13], dtype=np.float32)
utah_norm = np.asarray(results[15], dtype=np.float32)

# Scale Utah coordinates to real cortical size in mm
scale = utah_max_px / pixels_per_mm
utah_mm = utah_norm * scale

# Scale RFs from pixels to degrees
rfs = rfs_raw / PIXELS_PER_DEG

print({
    "lfp": lfp.shape,
    "utah_mm": utah_mm.shape,
    "rfs": rfs.shape,
})

# %% [markdown]
# MDS: compute distances, run, align to cortex, and plot in cortical space

# %%
distance_matrix = correlation_distance_matrix(lfp)
mds_embedding = run_mds(distance_matrix, random_state)
mds_aligned = align_to_cortex(utah_mm, mds_embedding)
mds_metrics = compute_metrics(monkey_alias, utah_mm, mds_aligned)

print("MDS metrics:", mds_metrics)

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(utah_mm[:, 0], utah_mm[:, 1], c=colors, s=30, alpha=0.7, edgecolors='black', linewidth=0.3, label='GT')
ax.scatter(mds_aligned[:, 0], mds_aligned[:, 1], c=colors, s=30, alpha=0.7, edgecolors='black', linewidth=0.3, marker='x', label='MDS')
ax.set_aspect('equal')
ax.set_title('Cortical space: GT vs MDS (aligned)')
ax.set_xlabel('X (mm)')
ax.set_ylabel('Y (mm)')
ax.legend()
plt.show()

# %% [markdown]
# Visual field: direct projection of GT and MDS; compare to measured RFs

# %%
utah_visual = project_to_visual_field(utah_mm, monkey_alias)
mds_visual = project_to_visual_field(mds_aligned, monkey_alias)

visual_xlim = (-10.0, 10.0)
visual_ylim = (-10.0, 10.0)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
# RFs
ax = axes[0]
ax.scatter(rfs[:, 0], rfs[:, 1], c=colors, s=20, alpha=0.7, edgecolors='black', linewidth=0.3)
ax.set_title('Measured RFs')
ax.set_xlim(visual_xlim); ax.set_ylim(visual_ylim); ax.set_aspect('equal')
# MDS → VF
ax = axes[1]
ax.scatter(mds_visual[:, 0], mds_visual[:, 1], c=colors, s=20, alpha=0.7, edgecolors='black', linewidth=0.3)
ax.set_title('MDS → Visual field (direct)')
ax.set_xlim(visual_xlim); ax.set_ylim(visual_ylim); ax.set_aspect('equal')
# GT → VF
ax = axes[2]
ax.scatter(utah_visual[:, 0], utah_visual[:, 1], c=colors, s=20, alpha=0.7, edgecolors='black', linewidth=0.3)
ax.set_title('GT cortex → Visual field (direct)')
ax.set_xlim(visual_xlim); ax.set_ylim(visual_ylim); ax.set_aspect('equal')
plt.show()

# %% [markdown]
# UMAP: define a tiny grid for speed, run on chosen band, select best vs MDS surrogate

# %%

def fast_umap_grid(num_electrodes: int) -> List[Dict[str, float]]:
    """Small, valid UMAP hyperparameter grid for interactive runs."""
    if num_electrodes <= 3:
        n_neighbors_values = [2]
    elif num_electrodes < 150:
        n_neighbors_values = [min(num_electrodes - 1, max(2, num_electrodes // 2))]
    else:
        n_neighbors_values = [150]
        if num_electrodes > 550:
            n_neighbors_values.append(550)
    min_dist_values = [0.1, 0.5, 0.9]
    return [
        {"n_neighbors": float(n), "min_dist": float(d)}
        for n in n_neighbors_values
        for d in min_dist_values
    ]

# Build grid and compute MDS distances for surrogate score
umap_grid = fast_umap_grid(lfp.shape[0])
mds_distances = euclidean_distances(mds_embedding)

print("UMAP grid:", umap_grid)

# %% [markdown]
# Run UMAP grid and pick the best embedding by surrogate (RMSE vs MDS distances)

# %%
alpha = 3.0
selection_mode = "rmse"  # "rmse" | "iedc" | "combo"

candidate_scores: List[Dict[str, Any]] = []
best_candidate: Dict[str, Any] | None = None

for params in umap_grid:
    n_neighbors = min(int(params["n_neighbors"]), lfp.shape[0] - 1)
    n_neighbors = max(n_neighbors, 2)
    embedding = run_umap_embedding(lfp, n_neighbors, params["min_dist"], random_state)

    umap_distances = euclidean_distances(embedding)
    surrogate_corr = float(calculate_corr_of_distances(mds_embedding, embedding))
    surrogate_rmse = float(np.sqrt(mean_squared_error(mds_distances.ravel(), umap_distances.ravel())))

    if selection_mode == "rmse":
        score = surrogate_rmse
    elif selection_mode == "iedc":
        score = 1.0 - surrogate_corr
    elif selection_mode == "combo":
        score = surrogate_rmse + alpha * (1.0 - surrogate_corr)
    else:
        raise ValueError(f"Unknown SELECTION_MODE: {selection_mode}")

    candidate = {
        "n_neighbors": n_neighbors,
        "min_dist": params["min_dist"],
        "surrogate_corr": surrogate_corr,
        "surrogate_rmse": surrogate_rmse,
        "score": score,
        "embedding": embedding,
    }
    candidate_scores.append(candidate)
    if best_candidate is None or score < best_candidate["score"]:
        best_candidate = candidate

best_candidate

# %% [markdown]
# Align best UMAP to cortex, print metrics, and plot in cortical space

# %%
if best_candidate is None:
    raise RuntimeError("UMAP sweep did not produce any embeddings.")

best_embedding = best_candidate["embedding"]
_, _, _, R_gt, s_gt, norm1_gt, norm2_gt, mean1_gt, mean2_gt = scipy_Antonio_procrustes(utah_mm, best_embedding)
umap_aligned = rescale_procrustes_map(best_embedding, R_gt, s_gt, norm1_gt, norm2_gt, mean1_gt, mean2_gt)

umap_metrics = compute_metrics(monkey_alias, utah_mm, umap_aligned)
print("UMAP metrics:", umap_metrics)

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(utah_mm[:, 0], utah_mm[:, 1], c=colors, s=30, alpha=0.7, edgecolors='black', linewidth=0.3, label='GT')
ax.scatter(umap_aligned[:, 0], umap_aligned[:, 1], c=colors, s=30, alpha=0.7, edgecolors='black', linewidth=0.3, marker='x', label='UMAP')
ax.set_aspect('equal')
ax.set_title('Cortical space: GT vs UMAP (aligned)')
ax.set_xlabel('X (mm)')
ax.set_ylabel('Y (mm)')
ax.legend()
plt.show()

# %% [markdown]
# Visual field: direct projection of GT and best UMAP; compare to measured RFs

# %%
umap_visual = project_to_visual_field(umap_aligned, monkey_alias)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
# RFs
ax = axes[0]
ax.scatter(rfs[:, 0], rfs[:, 1], c=colors, s=20, alpha=0.7, edgecolors='black', linewidth=0.3)
ax.set_title('Measured RFs')
ax.set_xlim(visual_xlim); ax.set_ylim(visual_ylim); ax.set_aspect('equal')
# UMAP → VF
ax = axes[1]
ax.scatter(umap_visual[:, 0], umap_visual[:, 1], c=colors, s=20, alpha=0.7, edgecolors='black', linewidth=0.3)
ax.set_title('UMAP → Visual field (direct)')
ax.set_xlim(visual_xlim); ax.set_ylim(visual_ylim); ax.set_aspect('equal')
# GT → VF
ax = axes[2]
ax.scatter(utah_visual[:, 0], utah_visual[:, 1], c=colors, s=20, alpha=0.7, edgecolors='black', linewidth=0.3)
ax.set_title('GT cortex → Visual field (direct)')
ax.set_xlim(visual_xlim); ax.set_ylim(visual_ylim); ax.set_aspect('equal')
plt.show()
