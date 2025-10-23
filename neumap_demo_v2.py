from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import euclidean_distances
from umap import UMAP

REPO_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = REPO_ROOT / "code"
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))
if str(CODE_ROOT) not in sys.path:
    sys.path.append(str(CODE_ROOT))

from code.cortex_model_utils import cortex_to_visual_mapping  # noqa: E402
from code.utils import (  # noqa: E402
    calculate_corr_of_distances,
    get_monkey_pixPerMM_utahMax,
    rescale_procrustes_map,
    scipy_Antonio_procrustes,
)
from code.utils_extension import (  # noqa: E402
    load_monkey_data_band,
    load_monkey_data_band_15seconds,
)

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


@dataclass(frozen=True)
class BandBundle:
    freq_band: str
    lfp: np.ndarray
    utah_mm: np.ndarray
    colors: np.ndarray


def resolve_monkey_name(alias: str) -> Tuple[str, str]:
    normalized = alias.strip()
    lowered = normalized.lower()
    if lowered in {"monkey_l", "lick"}:
        return "monkey_L", "LICK"
    if lowered in {"monkey_a", "ashton", "aston"}:
        return "monkey_A", "ASHTON"
    raise KeyError(f"Unknown monkey identifier '{alias}'. Expected one of monkey_L, monkey_A, LICK, ASHTON.")


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
        "n_init": 8,
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
    if random_state is not None:
        kwargs["random_state"] = random_state
    reducer = UMAP(**kwargs)
    return reducer.fit_transform(lfp)


def align_to_cortex(utah_mm: np.ndarray, embedding: np.ndarray) -> np.ndarray:
    _, _, _, R, s, norm1, norm2, mean1, mean2 = scipy_Antonio_procrustes(utah_mm, embedding)
    return rescale_procrustes_map(embedding, R, s, norm1, norm2, mean1, mean2)


def _wedge_params_for_monkey(monkey_alias: str) -> Dict[str, float]:
    return OPTIMIZED_WEDGE_PARAMS.get(monkey_alias, LITERATURE_WEDGE_PARAMS)


def project_to_visual_field(points_mm: np.ndarray, monkey_alias: str) -> np.ndarray:
    params = _wedge_params_for_monkey(monkey_alias)
    vx, vy = cortex_to_visual_mapping(
        points_mm[:, 0],
        points_mm[:, 1],
        params["a"],
        params["b"],
        params["alpha"],
        params["k"],
    )
    return np.column_stack([vx, vy])


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


def fast_umap_grid(num_electrodes: int) -> List[Dict[str, float]]:
    n_neighbors_values = list(range(150, num_electrodes + 1, 150))
    if not n_neighbors_values:
        fallback = max(2, num_electrodes // 3)
        n_neighbors_values = [min(max(fallback, 2), max(num_electrodes - 1, 2))]
    min_dist_values = [0.1, 0.5, 0.9]
    return [
        {"n_neighbors": float(n), "min_dist": float(d)}
        for n in n_neighbors_values
        for d in min_dist_values
    ]


def _data_paths() -> Tuple[str, str, str, str]:
    base_path = str(REPO_ROOT / "data" / "EYES_CLOSED")
    utah_path = str(REPO_ROOT / "data" / "coordinates_of_electrodes_on_cortex_using_photos_of_arrays")
    channel_map = str(REPO_ROOT / "data" / "channel_area_mapping" / "channel_area_mapping.mat")
    delete_path = str(REPO_ROOT / "data" / "deletedElectrodesDictionary")
    return base_path, utah_path, channel_map, delete_path


def _load_band_bundle(
    monkey_actual: str,
    freq_band: str,
    window: WindowType,
    pixels_per_mm: float,
    utah_max_px: float,
) -> BandBundle:
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
    utah_norm = np.asarray(results[15], dtype=np.float32)

    scale = utah_max_px / pixels_per_mm
    utah_mm = utah_norm * scale

    if lfp.shape[0] != utah_mm.shape[0]:
        raise RuntimeError(
            f"Channel mismatch for {freq_band}: LFP has {lfp.shape[0]}, but Utah map has {utah_mm.shape[0]}"
        )

    return BandBundle(freq_band=freq_band, lfp=lfp, utah_mm=utah_mm, colors=colors)


def _prune_umap_grid(full_grid: List[Dict[str, float]], max_combos: int, num_electrodes: int) -> List[Dict[str, float]]:
    pruned: List[Dict[str, float]] = []
    for params in full_grid:
        if len(pruned) >= max_combos:
            break
        if int(params["n_neighbors"]) >= num_electrodes:
            continue
        pruned.append(params)
    return pruned


def main() -> None:
    monkey_alias_input = "monkey_L"
    window: WindowType = "15s"
    selection_mode = "rmse"
    random_state: int | None = 42
    max_grid_combos = 6

    monkey_alias, monkey_actual = resolve_monkey_name(monkey_alias_input)
    pixels_per_mm, utah_max_px = get_monkey_pixPerMM_utahMax(monkey_actual)
    if pixels_per_mm is None or utah_max_px is None:
        raise ValueError(f"Missing scaling constants for monkey '{monkey_actual}'.")
    pixels_per_mm = float(pixels_per_mm)
    utah_max_px = float(utah_max_px)

    target_bands = tuple(b for b in FREQ_BANDS if b in PREFERRED_BANDS)
    if not target_bands:
        target_bands = (FREQ_BANDS[0],)

    bundles: Dict[str, BandBundle] = {}
    for freq in target_bands:
        try:
            bundle = _load_band_bundle(monkey_actual, freq, window, pixels_per_mm, utah_max_px)
        except FileNotFoundError as exc:
            print(f"Skipping {freq}: {exc}")
            continue
        except RuntimeError as exc:
            print(f"Skipping {freq}: {exc}")
            continue
        bundles[freq] = bundle
        print(f"Loaded {freq:<9} -> LFP shape {bundle.lfp.shape}")

    if not bundles:
        raise RuntimeError("No frequency bands could be loaded. Check that the dataset has been extracted.")

    reference_band = "LFP" if "LFP" in bundles else next(iter(bundles))
    if reference_band != "LFP":
        print(f"Reference band defaulting to {reference_band}")

    reference_bundle = bundles[reference_band]

    distance_matrix = correlation_distance_matrix(reference_bundle.lfp)
    mds_embedding = run_mds(distance_matrix, random_state)
    mds_aligned = align_to_cortex(reference_bundle.utah_mm, mds_embedding)
    mds_metrics = compute_metrics(monkey_alias, reference_bundle.utah_mm, mds_aligned)
    mds_distances = euclidean_distances(mds_embedding)

    print("MDS metrics (cortex vs recovered)")
    for name, value in mds_metrics.items():
        print(f"  {name:>14}: {value:0.4f}")

    full_grid = fast_umap_grid(reference_bundle.lfp.shape[0])
    pruned_grid = _prune_umap_grid(full_grid, max_grid_combos, reference_bundle.lfp.shape[0])
    if not pruned_grid:
        raise RuntimeError("UMAP grid is empty; adjust MAX_GRID_COMBOS or the neighbor bounds.")

    print("Evaluating UMAP grid:")
    for params in pruned_grid:
        print(f"  n_neighbors={int(params['n_neighbors']):>3}, min_dist={params['min_dist']}")

    alpha = 3.0
    candidate_scores: List[Dict[str, Any]] = []
    best_candidate: Dict[str, Any] | None = None

    for freq_band, bundle in bundles.items():
        for params in pruned_grid:
            n_neighbors = min(int(params["n_neighbors"]), bundle.lfp.shape[0] - 1)
            n_neighbors = max(n_neighbors, 2)
            embedding = run_umap_embedding(bundle.lfp, n_neighbors, params["min_dist"], random_state)

            umap_distances = euclidean_distances(embedding)
            surrogate_corr = float(calculate_corr_of_distances(mds_embedding, embedding))
            surrogate_rmse = float(
                np.sqrt(mean_squared_error(mds_distances.ravel(), umap_distances.ravel()))
            )

            if selection_mode == "rmse":
                score = surrogate_rmse
            elif selection_mode == "iedc":
                score = 1.0 - surrogate_corr
            elif selection_mode == "combo":
                score = surrogate_rmse + alpha * (1.0 - surrogate_corr)
            else:
                raise ValueError(f"Unknown SELECTION_MODE: {selection_mode}")

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

    best_embedding = best_candidate["embedding"]
    try:
        _, _, _, R_m, s_m, norm1_m, norm2_m, mean1_m, mean2_m = scipy_Antonio_procrustes(
            mds_aligned, best_embedding
        )
        umap_aligned_mds = rescale_procrustes_map(best_embedding, R_m, s_m, norm1_m, norm2_m, mean1_m, mean2_m)
    except Exception as exc:  # pragma: no cover - diagnostic fallback
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
    except Exception as exc:  # pragma: no cover - diagnostic fallback
        print(f"Direct cortex alignment failed: {exc}")

    top_candidates = sorted(candidate_scores, key=lambda item: item["score"])[:5]
    print("Top parameter sets (sorted by score):")
    for idx, candidate in enumerate(top_candidates, start=1):
        print(
            f"  {idx:>2}. freq={candidate['freq_band']:<9} n_neighbors={candidate['n_neighbors']:>3} "
            f"min_dist={candidate['min_dist']:.2f} score={candidate['score']:.4f} "
            f"corr={candidate['surrogate_corr']:.4f} rmse={candidate['surrogate_rmse']:.4f}"
        )


if __name__ == "__main__":
    main()
