from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, cast

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import euclidean_distances
from umap.umap_ import UMAP  # avoid parametric UMAP (TensorFlow) import

from .config import DEFAULT_FREQ_BAND, DEFAULT_MONKEY, DEFAULT_RANDOM_SEED, canonical_alias, FREQ_BANDS
from .data_loader import DemoBundle, WindowType, load_demo_bundle
from .utils import (
    calculate_corr_of_distances,
    scipy_Antonio_procrustes,
    rescale_procrustes_map,
)
from .cortex_model_utils import cortex_to_visual_mapping

LITERATURE_WEDGE_PARAMS = {"a": 0.61, "b": 106.0, "k": 13.6, "alpha": 0.86}
OPTIMIZED_WEDGE_PARAMS = {
    "monkey_L": {"a": 0.5251, "b": 80.0, "k": 13.64, "alpha": 0.774},
    "monkey_A": {"a": 0.6857, "b": 106.0, "k": 16.69, "alpha": 0.948},
}


@dataclass(frozen=True)
class EmbeddingResult:
    name: str
    embedding: np.ndarray
    aligned_mm: np.ndarray
    metrics: Dict[str, float]
    params: Dict[str, float]
    surrogate_metrics: Dict[str, float] | None = None
    # aligned to ground-truth cortex (using scipy_Antonio_procrustes with utah as anchor)
    aligned_mm_gt: np.ndarray | None = None
    # metrics computed for GT-aligned embedding
    gt_metrics: Dict[str, float] | None = None


@dataclass(frozen=True)
class PipelineOutput:
    bundle: DemoBundle
    distance_matrix: np.ndarray
    mds: EmbeddingResult
    umap_best: EmbeddingResult
    umap_candidates: List[Tuple[Dict[str, float], Dict[str, float]]]


def correlation_distance_matrix(lfp: np.ndarray) -> np.ndarray:
    corr = np.corrcoef(lfp)
    corr = np.clip(corr, -0.999, 0.999)
    dist = 1.0 - corr
    np.fill_diagonal(dist, 0.0)
    dist = np.nan_to_num(dist, nan=0.0, posinf=0.0, neginf=0.0)
    return dist.astype(np.float32)


def run_mds(distance_matrix: np.ndarray, random_state: int | None) -> np.ndarray:
    kwargs = {
        "n_components": 2,
        "dissimilarity": "precomputed",
        "n_init": 8,
        "max_iter": 400,
    }
    if random_state is not None:
        kwargs["random_state"] = random_state
    reducer = MDS(**kwargs)
    return reducer.fit_transform(distance_matrix)


def run_umap_embedding(lfp: np.ndarray, n_neighbors: int, min_dist: float, random_state: int | None) -> np.ndarray:
    pca = PCA(n_components=2) if random_state is None else PCA(n_components=2, random_state=random_state)
    pca_init = cast(np.ndarray, pca.fit_transform(lfp))
    kwargs: Dict[str, Any] = dict(
        n_components=2,
        metric="euclidean",
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        init=cast(Any, pca_init),
    )
    if random_state is not None:
        kwargs["random_state"] = random_state
    reducer = UMAP(**kwargs)
    return cast(np.ndarray, reducer.fit_transform(lfp))


def align_to_cortex(utah_mm: np.ndarray, embedding: np.ndarray) -> np.ndarray:
    _, _, _, R, s, norm1, norm2, mean1, mean2 = scipy_Antonio_procrustes(utah_mm, embedding)
    return rescale_procrustes_map(embedding, R, s, norm1, norm2, mean1, mean2)

def _wedge_params_for_monkey(monkey_alias: str) -> Dict[str, float]:
    return OPTIMIZED_WEDGE_PARAMS.get(monkey_alias, LITERATURE_WEDGE_PARAMS)


def project_to_visual_field(points_mm: np.ndarray, monkey: str) -> np.ndarray:
    params = _wedge_params_for_monkey(monkey)
    vx, vy = cortex_to_visual_mapping(
        points_mm[:, 0],
        points_mm[:, 1],
        params["a"],
        params["b"],
        params["alpha"],
        params["k"],
    )
    return np.column_stack([vx, vy])


def compute_metrics(monkey: str, utah_mm: np.ndarray, recovered_mm: np.ndarray) -> Dict[str, float]:
    cortical_corr = float(calculate_corr_of_distances(utah_mm, recovered_mm))
    cortical_rmse = float(np.sqrt(mean_squared_error(utah_mm, recovered_mm)))

    utah_visual = project_to_visual_field(utah_mm, monkey)
    recovered_visual = project_to_visual_field(recovered_mm, monkey)

    visual_corr = float(calculate_corr_of_distances(utah_visual, recovered_visual))
    visual_rmse = float(np.sqrt(mean_squared_error(utah_visual, recovered_visual)))

    return {
        "cortical_corr": cortical_corr,
        "cortical_rmse": cortical_rmse,
        "visual_corr": visual_corr,
        "visual_rmse": visual_rmse,
    }


def default_umap_grid(num_electrodes: int) -> List[Dict[str, Any]]:
    n_neighbors_values = list(range(50, num_electrodes + 1, 50))
    min_dist_values = [round(x * 0.1, 1) for x in range(1, 10)]  # 0.1 to 0.9
    return [
        {"n_neighbors": n, "min_dist": d}
        for n in n_neighbors_values
        for d in min_dist_values
    ]


def fast_umap_grid(num_electrodes: int) -> List[Dict[str, Any]]:
    n_neighbors_values = list(range(150, num_electrodes + 1, 150))
    min_dist_values = [0.1, 0.5, 0.9]
    return [
        {"n_neighbors": n, "min_dist": d}
        for n in n_neighbors_values
        for d in min_dist_values
    ]


def run_demo_pipeline(
    monkey: str = DEFAULT_MONKEY,
    window: WindowType = "15s",
    random_state: int | None = DEFAULT_RANDOM_SEED,
    umap_grid: List[Dict[str, Any]] | None = None,
    selection_mode: str = 'rmse'
) -> PipelineOutput:
    """
    Run the demo pipeline for a given monkey and window.
    Args:
        monkey: Monkey name
        window: Time window
        random_state: Random seed
        umap_grid: List of UMAP parameter dicts (n_neighbors, min_dist). If None, uses fast_umap_grid.
        selection_mode: Metric for UMAP selection ('rmse', 'iedc', 'combo')
    Returns:
        PipelineOutput
    """
    monkey_alias = canonical_alias(monkey)
    bundle = load_demo_bundle(monkey=monkey_alias, freq_band="LFP", window=window)
    monkey_alias = bundle.monkey

    dist_matrix = correlation_distance_matrix(bundle.lfp)
    mds_embedding = run_mds(dist_matrix, random_state)
    mds_aligned = align_to_cortex(bundle.utah_mm, mds_embedding)
    mds_metrics = compute_metrics(monkey_alias, bundle.utah_mm, mds_aligned)

    mds_params: Dict[str, float] = {}
    if random_state is not None:
        mds_params["random_state"] = float(random_state)
    mds_result = EmbeddingResult(
        name="MDS",
        embedding=mds_embedding,
        aligned_mm=mds_aligned,
        metrics=mds_metrics,
        params=mds_params,
    )

    if umap_grid is None:
        umap_grid = fast_umap_grid(bundle.lfp.shape[0])

    # Load LFP data for each frequency band for parameter search
    freq_lfp_data = {}
    for fb in FREQ_BANDS:
        try:
            freq_bundle = load_demo_bundle(monkey=monkey_alias, freq_band=fb, window=window)
            freq_lfp_data[fb] = freq_bundle.lfp
        except FileNotFoundError:
            print(f"Warning: No data for {fb}, skipping")
            continue

    if not freq_lfp_data:
        raise RuntimeError("No frequency band data available for UMAP parameter search")

    # Parameter search per frequency band
    best_freq = None
    best_params: Dict[str, Any] | None = None
    best_metrics: Dict[str, float] | None = None
    best_embedding: np.ndarray | None = None
    best_score = np.inf
    candidate_scores: List[Tuple[Dict[str, Any], Dict[str, float]]] = []

    for freq_band, lfp_data in freq_lfp_data.items():
        print(f"Searching UMAP params for {freq_band}...")
        print(f"  Testing UMAP grid:")
        for params in umap_grid:
            print(f"    n_neighbors={params['n_neighbors']}, min_dist={params['min_dist']}")
        print(f"  Testing UMAP grid:")
        for params in umap_grid:
            print(f"    n_neighbors={params['n_neighbors']}, min_dist={params['min_dist']}")
        freq_best_score = np.inf
        freq_best_params = None
        freq_best_metrics = None
        freq_best_embedding = None
        alpha = 3.0

        for params in umap_grid:
            n_neighbors = int(params["n_neighbors"])
            min_dist = params["min_dist"]

            
            embedding = run_umap_embedding(lfp_data, n_neighbors, min_dist, random_state)
            surrogate_corr = float(calculate_corr_of_distances(mds_embedding, embedding))
            mds_distances = euclidean_distances(mds_embedding)
            umap_distances = euclidean_distances(embedding)
            surrogate_rmse = float(
                np.sqrt(mean_squared_error(mds_distances.ravel(), umap_distances.ravel()))
            )
            param_dict = {"n_neighbors": float(n_neighbors), "min_dist": min_dist, "freq_band": freq_band}
            metrics = {"iedc": surrogate_corr, "rmse": surrogate_rmse}
            candidate_scores.append((param_dict, metrics))
            # Selection logic
            if selection_mode == 'rmse':
                score = surrogate_rmse
            elif selection_mode == 'iedc':
                score = 1 - surrogate_corr
            elif selection_mode == 'combo':
                score = surrogate_rmse + alpha * (1 - surrogate_corr)
            else:
                raise ValueError(f"Unknown selection_mode: {selection_mode}")
            if score < freq_best_score:
                freq_best_score = score
                freq_best_params = param_dict
                freq_best_metrics = metrics
                freq_best_embedding = embedding

        print(f"Best for {freq_band}: score {freq_best_score:.3f} (mode: {selection_mode})")
        # Check if this frequency is better
        if freq_best_score < best_score:
            best_score = freq_best_score
            best_freq = freq_band
            best_params = freq_best_params
            best_metrics = freq_best_metrics
            best_embedding = freq_best_embedding

    print(f"Selected {best_freq} with score {best_score:.3f} (mode: {selection_mode})")

    if best_freq is None or best_params is None or best_metrics is None or best_embedding is None:
        raise RuntimeError("UMAP parameter search did not find any valid embeddings")

    # Use the best embedding directly (no need to re-run UMAP)
    # Align UMAP embedding to cortex using the MDS surrogate alignment (leave existing behavior)
    # First, align UMAP to MDS-aligned map (mds_aligned is already in cortex mm space)
    try:
        _, _, _, R_m, s_m, norm1_m, norm2_m, mean1_m, mean2_m = scipy_Antonio_procrustes(mds_aligned, best_embedding)
        umap_aligned_mds = rescale_procrustes_map(best_embedding, R_m, s_m, norm1_m, norm2_m, mean1_m, mean2_m)
        umap_metrics = compute_metrics(monkey_alias, bundle.utah_mm, umap_aligned_mds)
    except Exception:
        # fallback to aligning directly to cortex if MDS-based alignment fails
        umap_aligned_mds = align_to_cortex(bundle.utah_mm, best_embedding)
        umap_metrics = compute_metrics(monkey_alias, bundle.utah_mm, umap_aligned_mds)

    # Now compute an additional alignment directly to the ground-truth cortical map (utah_mm)
    try:
        _, _, _, R_gt, s_gt, norm1_gt, norm2_gt, mean1_gt, mean2_gt = scipy_Antonio_procrustes(bundle.utah_mm, best_embedding)
        umap_aligned_gt = rescale_procrustes_map(best_embedding, R_gt, s_gt, norm1_gt, norm2_gt, mean1_gt, mean2_gt)
        umap_gt_metrics = compute_metrics(monkey_alias, bundle.utah_mm, umap_aligned_gt)
    except Exception:
        umap_aligned_gt = None
        umap_gt_metrics = None

    umap_result = EmbeddingResult(
        name="UMAP",
        embedding=best_embedding,
        aligned_mm=umap_aligned_mds,
        metrics=umap_metrics,
        params=best_params,
        surrogate_metrics=best_metrics,
        aligned_mm_gt=umap_aligned_gt,
        gt_metrics=umap_gt_metrics,
    )

    return PipelineOutput(
        bundle=bundle,
        distance_matrix=dist_matrix,
        mds=mds_result,
        umap_best=umap_result,
        umap_candidates=candidate_scores,
    )
