# -*- coding: utf-8 -*-
"""
Performance vs Time analysis with visual field projection.

This script modernizes the classic performance-vs-time pipeline by:
- Using the new consolidated data loaders (``load_monkey_data_band``)
- Computing dimensionality reduction (PCA + UMAP + MDS) across multiple time windows
- Evaluating reconstruction quality in cortical space (IEDC + RMSE)
- Projecting recovered maps to visual field space and repeating the evaluation
- Saving figures and serialized results under ``results/review_performance_vs_time_VF``

The output data structure mirrors ``paper_figuresV10_performanceVsTime_August_2024.py``
with additional entries for the visual field metrics.

SAMPLING STRATEGIES:
    USE_CONTINUOUS_CHUNKS = False (default)
        - Random subsampling: uses single seeded RNG for reproducibility
        - Independent samples across repetitions but deterministic results
        - Files saved as: *_random.npy and *_random.png

    USE_CONTINUOUS_CHUNKS = True
        - Continuous chunks: samples consecutive time points from random positions
        - Tests temporal continuity effects
        - Uses seeded RNG for reproducibility
        - Files saved as: *_continuous.npy and *_continuous.png

EXAMPLE GENERATION:
    GENERATE_EXAMPLES = False (default)
        - Set to True to save example maps for each time window
        - Generates NUM_EXAMPLES_PER_TIME (default: 3) examples per time window
        - Each example shows: aligned recovered map + data heatmap (up to 500ms)
        - Saved in: results/review_performance_vs_time_VF/examples_{random|continuous}/

WEDGE PARAMETER MODES:
    USE_LITERATURE_VALUES = False (default)
        - Uses optimized wedge dipole parameters for each monkey
        - Better accuracy, recommended for most analyses

    USE_LITERATURE_VALUES = True
        - Uses original literature values: a=0.61, b=106, k=13.6, alpha=0.86
        - Consistent baseline, useful for comparisons

EXAMPLES:

    # Default: Random sampling with optimized wedge parameters
    uv run python code/review_performance_vs_time_VF.py

    # Force run analysis (useful if you changed DO_ANALYSIS_DEFAULT = False)
    uv run python code/review_performance_vs_time_VF.py --do-analysis

    # Skip analysis and just generate plots from existing results
    uv run python code/review_performance_vs_time_VF.py --no-analysis
    
    # To test continuous chunks: Set USE_CONTINUOUS_CHUNKS = True in the script
    # Results won't overwrite random sampling results (different filenames)

    python code/review_performance_per_frequency_VF.py --do-analysis --generate-example-maps --num-examples 5

    # Use cached results with examples (if previously generated)
    uv run python python code/review_performance_per_frequency_VF.py --no-analysis --generate-example-maps

"""

#%%
from __future__ import annotations

import os
import sys
import json
import math
from pathlib import Path
from collections import OrderedDict
from datetime import datetime
from typing import Dict, Tuple
import argparse

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from numpy.random import default_rng
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.metrics import mean_squared_error
import umap

# -----------------------------------------------------------------------------
# Repository-aware imports
# -----------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.append(str(CODE_DIR))

from utils import (  # type: ignore
    calculate_corr_of_distances,
    corr_Ds,
    get_monkey_pixPerMM_utahMax,
    rescale_procrustes_map,
    scipy_Antonio_procrustes,
)
from utils_extension import load_monkey_data_band  # type: ignore
from cortex_model_utils import cortex_to_visual_mapping  # type: ignore

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

RESULTS_DIR = ROOT / "results" / "review_performance_vs_time_VF"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

STANDARIZE_LFP = False
EXCLUDE_V4 = True
FREQ_BAND = "gamma"
METHOD_NAMES = ["PCA", "UMAP", "MDS"]
METHOD_NAMES = ["PCA", "UMAP", "MDS"]
#DATASET_LABELS = ["gamma", "MUA", "MUALFP_timeSeries"]
DATASET_LABELS = ["gamma", "MUA"]

# Time windows: only multiples of 5 seconds (0.25s minimum for stability, then 5, 10, 15)
TIME_WINDOWS_SECONDS = np.array([0.25, 5.0, 10.0, 15.0])
TIME_WINDOWS_MS = TIME_WINDOWS_SECONDS * 1000.0
TIME_WINDOWS_SECONDS = TIME_WINDOWS_MS / 1000.0


TIME_WINDOWS_MS = np.arange(250, 2500 * 6 + 1, 250 * 3)

# Configuration flags
TIME_WINDOWS_SECONDS = TIME_WINDOWS_MS / 1000.0


SAMPLING_RATE = 500
SAMPLE_COUNTS = (TIME_WINDOWS_SECONDS * SAMPLING_RATE).astype(int)
NUM_REPS = 30  # can be increased to 30 to match historical analyses

# Sampling strategy flag
# False: Random subsampling - creates fresh RNG each repetition (matches old script exactly)
#        More variance but fully independent samples
# True:  Continuous chunks - samples consecutive time points from random starting positions
#        Less variance, tests temporal continuity
USE_CONTINUOUS_CHUNKS = False

# Example generation flag
GENERATE_EXAMPLES = True  # Set to True to save example maps for each time window
NUM_EXAMPLES_PER_TIME = 3  # Number of example maps to save per time window

UMAP_PARAMS_BY_MONKEY: Dict[str, Tuple[int, float]] = {
    "monkey_L": (110, 0.5),
    "monkey_A": (120, 0.9),
}

# Wedge dipole parameters for cortex-to-visual projection
# Literature values (will be optimized or set to predefined values)
LITERATURE_WEDGE_PARAMS = {"a": 0.61, "b": 106, "k": 13.6, "alpha": 0.86}

# Predefined optimized wedge parameters for each monkey (from review_visual_field_projection)
OPTIMIZED_WEDGE_PARAMS = {
    'monkey_L': {
        'a': 0.5251,
        'b': 80.0,
        'k': 13.64,
        'alpha': 0.774
    },
    'monkey_A': {
        'a': 0.6857,
        'b': 106.0,
        'k': 16.69,
        'alpha': 0.948
    }
}

# Flag to control wedge parameter selection
USE_LITERATURE_VALUES = False  # Set to True to use literature values, False for optimized values

# Set WEDGE_PARAMS based on the flag
if USE_LITERATURE_VALUES:
    WEDGE_PARAMS = LITERATURE_WEDGE_PARAMS.copy()
else:
    # Use optimized parameters (will be monkey-specific later in the script)
    WEDGE_PARAMS = LITERATURE_WEDGE_PARAMS.copy()  # fallback default



SAVE_RESULTS = True

# Set this flag to control analysis behavior (can be overridden by --do-analysis CLI argument)
DO_ANALYSIS_DEFAULT = True

# -----------------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Performance vs Time analysis with visual field projection")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--do-analysis",
        action="store_true",
        help="Run the analysis. Overrides the DO_ANALYSIS_DEFAULT setting."
    )
    group.add_argument(
        "--no-analysis",
        action="store_true",
        help="Skip analysis and load existing results. Overrides the DO_ANALYSIS_DEFAULT setting."
    )
    args = parser.parse_args()

    # Determine final do_analysis value
    if args.do_analysis:
        do_analysis = True
    elif args.no_analysis:
        do_analysis = False
    else:
        do_analysis = DO_ANALYSIS_DEFAULT

    args.do_analysis = do_analysis
    return args

# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------

def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _save_fig(fig: plt.Figure, output: Path, dpi: int = 300) -> None:
    """Save a matplotlib figure and log confirmation with timestamp.

    Centralizes saving so we can easily add timestamped variants later.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    # On Windows, LastWriteTime reflects modification; creation time may stay old if overwritten.
    try:
        ts = datetime.fromtimestamp(output.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        print(f"Saved figure: {output.name} (modified {ts})")
    except Exception as e:
        print(f"Saved figure: {output.name} (timestamp unavailable: {e})")


def _results_path(monkey: str) -> Path:
    sampling_suffix = "continuous" if USE_CONTINUOUS_CHUNKS else "random"
    return RESULTS_DIR / f"performanceTimeResults_{monkey}_{FREQ_BAND}_{sampling_suffix}.npy"


def load_latest_results(monkey: str) -> OrderedDict | None:
    file_path = _results_path(monkey)
    if file_path.exists():
        return np.load(file_path, allow_pickle=True).item()
    return None


def project_to_visual_field(points_mm: np.ndarray) -> np.ndarray:
    """Project cortical coordinates (mm) to visual field space (deg)."""
    vx, vy = cortex_to_visual_mapping(
        points_mm[:, 0],
        points_mm[:, 1],
        WEDGE_PARAMS["a"],
        WEDGE_PARAMS["b"],
        WEDGE_PARAMS["alpha"],
        WEDGE_PARAMS["k"],
    )
    return np.column_stack([vx, vy])


# -----------------------------------------------------------------------------
# Core analysis
# -----------------------------------------------------------------------------

def load_band_data(monkey: str) -> Tuple[np.ndarray, ...]:
    """Wrapper around ``load_monkey_data_band`` with repo-relative paths."""
    base_path = ROOT / "data" / "EYES_CLOSED"
    utah_path = ROOT / "data" / "coordinates_of_electrodes_on_cortex_using_photos_of_arrays"
    channel_map_path = ROOT / "data" / "channel_area_mapping" / "channel_area_mapping.mat"
    delete_elecs_path = ROOT / "data" / "deletedElectrodesDictionary"

    return load_monkey_data_band(
        monkey,
        STANDARIZE_LFP,
        EXCLUDE_V4,
        str(base_path),
        str(utah_path),
        str(channel_map_path),
        str(delete_elecs_path),
        freq_band=FREQ_BAND,
        load_MUA=True,
        LFP_float16=False,
    )


def compute_visual_metrics(
    utah_mm: np.ndarray,
    recovered_mm: np.ndarray,
) -> Tuple[float, float]:
    utah_visual = project_to_visual_field(utah_mm)
    recovered_visual = project_to_visual_field(recovered_mm)
    corr_visual = calculate_corr_of_distances(utah_visual, recovered_visual)
    rms_visual = np.sqrt(mean_squared_error(utah_visual, recovered_visual))
    return corr_visual, rms_visual


def _save_example_map(
    monkey: str,
    dataset: str,
    method: str,
    time_window: float,
    rep: int,
    sampled_cols: np.ndarray,
    data_matrix: np.ndarray,
    recovered_mm: np.ndarray,
    utah_mm: np.ndarray,
    colors_w_alpha: np.ndarray,
    corr_val: float,
    rms_val: float,
) -> None:
    """Save example map showing aligned recovered positions and data heatmap."""
    sampling_suffix = "continuous" if USE_CONTINUOUS_CHUNKS else "random"
    
    # Create examples directory
    examples_dir = RESULTS_DIR / f"examples_{sampling_suffix}"
    examples_dir.mkdir(exist_ok=True)
    
    # Create figure with 2 subplots: aligned map + data heatmap
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150)
    
    # Left: Aligned recovered map
    ax = axes[0]
    ax.scatter(recovered_mm[:, 0], recovered_mm[:, 1], c=colors_w_alpha, s=30, alpha=0.8, edgecolors='black', linewidth=0.5)
    ax.scatter(utah_mm[:, 0], utah_mm[:, 1], c='black', s=10, alpha=0.3, marker='x', label='Ground truth')
    ax.set_title(f'Aligned Map\nIEDC={corr_val:.3f}, RMSE={rms_val:.2f}mm', fontsize=12, fontweight='bold')
    ax.set_xlabel('X (mm)', fontsize=10)
    ax.set_ylabel('Y (mm)', fontsize=10)
    ax.set_aspect('equal')
    ax.legend(loc='best', fontsize=8)
    ax.set_facecolor('white')
    ax.spines[['top', 'right']].set_visible(False)
    
    # Right: Data heatmap (subset up to 500ms = 250 samples at 500Hz)
    ax = axes[1]
    max_time_samples = min(250, len(sampled_cols))  # 500ms at 500Hz
    subset_cols = sampled_cols[:max_time_samples]
    data_subset = data_matrix[:, subset_cols]
    
    im = ax.imshow(data_subset, aspect='auto', cmap='seismic', interpolation='nearest')
    ax.set_title(f'Data Subset ({len(subset_cols)} samples, {len(subset_cols)/SAMPLING_RATE*1000:.0f}ms)', 
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Time samples', fontsize=10)
    ax.set_ylabel('Channels', fontsize=10)
    plt.colorbar(im, ax=ax, label='Amplitude')
    
    fig.suptitle(f'{monkey} | {dataset} | {method} | {time_window}s | Rep {rep+1}/{NUM_EXAMPLES_PER_TIME}', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save
    filename = f"{monkey}_{dataset}_{method}_{time_window}s_rep{rep+1:02d}.png"
    output = examples_dir / filename
    fig.savefig(output, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    if rep == 0:
        print(f"        Saved example: {filename}")


def compute_performance_for_monkey(monkey: str) -> OrderedDict:
    print(f"\n===== Processing {monkey} =====")
    
    # Print sampling strategy
    sampling_mode = "CONTINUOUS CHUNKS" if USE_CONTINUOUS_CHUNKS else "RANDOM SAMPLING"
    print(f"Sampling strategy: {sampling_mode}")

    # Select wedge parameters based on flag and monkey
    global WEDGE_PARAMS
    if USE_LITERATURE_VALUES:
        WEDGE_PARAMS = LITERATURE_WEDGE_PARAMS.copy()
        print(f"Using literature wedge parameters: a={WEDGE_PARAMS['a']:.4f}, b={WEDGE_PARAMS['b']:.1f}, k={WEDGE_PARAMS['k']:.2f}, alpha={WEDGE_PARAMS['alpha']:.3f}")
    else:
        # Use optimized parameters for this monkey
        if monkey in OPTIMIZED_WEDGE_PARAMS:
            WEDGE_PARAMS = OPTIMIZED_WEDGE_PARAMS[monkey].copy()
            print(f"Using optimized wedge parameters for {monkey}: a={WEDGE_PARAMS['a']:.4f}, b={WEDGE_PARAMS['b']:.1f}, k={WEDGE_PARAMS['k']:.2f}, alpha={WEDGE_PARAMS['alpha']:.3f}")
        else:
            WEDGE_PARAMS = LITERATURE_WEDGE_PARAMS.copy()
            print(f"Warning: No optimized parameters found for monkey {monkey}. Using literature values: a={WEDGE_PARAMS['a']:.4f}, b={WEDGE_PARAMS['b']:.1f}, k={WEDGE_PARAMS['k']:.2f}, alpha={WEDGE_PARAMS['alpha']:.3f}")

    data_tuple = load_band_data(monkey)
    (
        LFP,
        MUABINNED,
        anchorLFP,
        anchorUtah,
        anchorUtahDistances,
        colors,
        array_number_v1,
        array_idx,
        colors_w_alpha,
        channel_area_map,
        array_nums,
        areas,
        channel_nums,
        rfs,
        rfs_distances,
        utah,
        utah_distances,
        bad_dict,
        subject_cond,
        nonvalidEList,
        idx_nonvalid,
        nonvalidE,
        utah_max,
        selectedChannelsIndices,
    ) = data_tuple

    pixels_per_mm, utah_max_scale = get_monkey_pixPerMM_utahMax(monkey)
    if pixels_per_mm is None or utah_max_scale is None:
        raise RuntimeError(f"Missing scaling information for monkey {monkey}")

    utah_mm = utah * utah_max_scale / pixels_per_mm

    # Convert RFs to degrees
    PIXELS_PER_DEG = 25.78
    rfs_deg = rfs / PIXELS_PER_DEG

    datasets = [
        LFP,
        MUABINNED if MUABINNED is not None else None,
        None,
    ]
    if datasets[1] is None:
        DATASET_LABELS_USE = DATASET_LABELS[:1]
        datasets = datasets[:1]
    else:
        DATASET_LABELS_USE = DATASET_LABELS.copy()
        # Only create MUALFP if LFP and MUABINNED have compatible shapes
        if LFP.shape[0] == MUABINNED.shape[0]:
            datasets[2] = np.hstack((LFP, MUABINNED))
        else:
            print(f"Warning: LFP ({LFP.shape[0]} electrodes) and MUABINNED ({MUABINNED.shape[0]} electrodes) have incompatible shapes. Skipping MUALFP combination.")
            DATASET_LABELS_USE = DATASET_LABELS[:2]
            datasets = datasets[:2]

    n_data = len(datasets)
    n_methods = len(METHOD_NAMES)
    n_windows = len(SAMPLE_COUNTS)

    cortical_corr = np.zeros((n_data, n_methods, n_windows, NUM_REPS))
    cortical_rms = np.zeros_like(cortical_corr)
    visual_corr = np.zeros_like(cortical_corr)
    visual_rms = np.zeros_like(cortical_corr)
    window_tracker = np.broadcast_to(SAMPLE_COUNTS, (n_data, n_methods, n_windows))

    # Identify unique arrays for within-array calculations
    # selectedChannelsIndices contains indices grouped by array
    n_arrays = len(selectedChannelsIndices)
    
    # Initialize within-array result arrays
    within_array_cortical_iedc = np.zeros((n_data, n_methods, n_windows, n_arrays, NUM_REPS))
    within_array_cortical_rms = np.zeros_like(within_array_cortical_iedc)
    within_array_visual_iedc = np.zeros_like(within_array_cortical_iedc)
    within_array_visual_rms = np.zeros_like(within_array_cortical_iedc)

    # Print sampling strategy
    sampling_strategy = "continuous chunks" if USE_CONTINUOUS_CHUNKS else "random subsampling"
    print(f"  Sampling strategy: {sampling_strategy}")
    
    # Initialize RNG with seed for reproducibility (both sampling modes)
    if USE_CONTINUOUS_CHUNKS:
        rng = default_rng(20251007)  # Seeded RNG for continuous chunks
    else:
        rng = default_rng(20251007)  # Seeded RNG for random sampling
    
    umap_neighbors, umap_min_dist = UMAP_PARAMS_BY_MONKEY.get(monkey, (110, 0.5))

    best_H = {}
    best_iedc = {}

    for data_idx, (label, data_matrix) in enumerate(zip(DATASET_LABELS_USE, datasets)):
        if data_matrix is None:
            continue
        print(f"  Dataset: {label} | shape={data_matrix.shape}")
        max_cols = data_matrix.shape[1]

        for method_idx, method_name in enumerate(METHOD_NAMES):
            print(f"    Method: {method_name}")

            for window_idx, sample_count in enumerate(SAMPLE_COUNTS):
                if sample_count > max_cols:
                    print(f"      Skipping time window {TIME_WINDOWS_SECONDS[window_idx]}s: dataset only has {max_cols} columns")
                    continue
                
                time_window = TIME_WINDOWS_SECONDS[window_idx]
                key = (label, method_name, time_window)
                best_iedc[key] = -np.inf
                best_H[key] = None
                
                for rep in range(NUM_REPS):
                    # Choose sampling strategy
                    if USE_CONTINUOUS_CHUNKS:
                        # Sample a continuous chunk starting at a random position
                        start_idx = rng.integers(0, max_cols - sample_count + 1)
                        sampled_cols = np.arange(start_idx, start_idx + sample_count)
                    else:
                        # Random sampling using seeded RNG for reproducibility
                        sampled_cols = rng.choice(max_cols, size=sample_count, replace=False)
                    
                    X = data_matrix[:, sampled_cols]
                    if rep == 0:
                        print(f"      Time {time_window}s: sampling {sample_count} cols from {max_cols} available ({sample_count/max_cols*100:.1f}%)")

                    if method_name == "PCA":
                        reducer = PCA(n_components=2)
                        embedding = reducer.fit_transform(X)
                    elif method_name == "UMAP":
                        pca_init = PCA(n_components=2).fit_transform(X)
                        reducer = umap.UMAP(
                            n_neighbors=umap_neighbors,
                            min_dist=umap_min_dist,
                            metric="euclidean",
                            init=pca_init,
                            n_components=2,
                        )
                        embedding = reducer.fit_transform(X)
                    elif method_name == "MDS":
                        # Compute correlation-based dissimilarity matrix
                        corr_matrix = corr_Ds(X)[0]
                        dissimilarity_matrix = np.power(1 - corr_matrix, 1).astype(np.float32)
                        reducer = MDS(n_components=2, dissimilarity="precomputed", random_state=42, n_init=1)
                        embedding = reducer.fit_transform(dissimilarity_matrix)
                    else:
                        raise ValueError(f"Unsupported method {method_name}")

                    # Use full map alignment (all electrodes) instead of anchor points
                    _, _, _, R, s, norm1, norm2, mean1, mean2 = scipy_Antonio_procrustes(
                        utah_mm, embedding
                    )
                    recovered_mm = rescale_procrustes_map(
                        embedding, R, s, norm1, norm2, mean1, mean2
                    )

                    corr_val = calculate_corr_of_distances(utah_mm, recovered_mm)
                    rms_val = np.sqrt(mean_squared_error(utah_mm, recovered_mm))
                    corr_visual, rms_visual = compute_visual_metrics(utah_mm, recovered_mm)

                    cortical_corr[data_idx, method_idx, window_idx, rep] = corr_val
                    cortical_rms[data_idx, method_idx, window_idx, rep] = rms_val
                    visual_corr[data_idx, method_idx, window_idx, rep] = corr_visual
                    visual_rms[data_idx, method_idx, window_idx, rep] = rms_visual

                    # Calculate within-array performance metrics
                    for array_idx in range(n_arrays):
                        # Get electrodes for this array
                        single_arr_indices = selectedChannelsIndices[array_idx]
                        utah_single = utah_mm[single_arr_indices]
                        recovered_single = recovered_mm[single_arr_indices]
                        
                        # Skip arrays with too few electrodes
                        if len(utah_single) < 3:
                            continue
                        
                        # Center arrays to 0 without rescaling
                        utah_single_centered = utah_single - utah_single.mean(axis=0)
                        recovered_single_centered = recovered_single - recovered_single.mean(axis=0)
                        
                        # Cortical metrics
                        corr_cortical_array = calculate_corr_of_distances(utah_single_centered, recovered_single_centered)
                        rms_cortical_array = np.sqrt(mean_squared_error(utah_single_centered, recovered_single_centered))
                        within_array_cortical_iedc[data_idx, method_idx, window_idx, array_idx, rep] = corr_cortical_array
                        within_array_cortical_rms[data_idx, method_idx, window_idx, array_idx, rep] = rms_cortical_array
                        
                        # Visual metrics (project each array separately)
                        utah_visual_single = project_to_visual_field(utah_single)
                        recovered_visual_single = project_to_visual_field(recovered_single)
                        utah_visual_centered = utah_visual_single - utah_visual_single.mean(axis=0)
                        recovered_visual_centered = recovered_visual_single - recovered_visual_single.mean(axis=0)
                        corr_visual_array = calculate_corr_of_distances(utah_visual_centered, recovered_visual_centered)
                        rms_visual_array = np.sqrt(mean_squared_error(utah_visual_centered, recovered_visual_centered))
                        within_array_visual_iedc[data_idx, method_idx, window_idx, array_idx, rep] = corr_visual_array
                        within_array_visual_rms[data_idx, method_idx, window_idx, array_idx, rep] = rms_visual_array

                    if corr_val > best_iedc[key]:
                        best_iedc[key] = corr_val
                        best_H[key] = recovered_mm.copy()
                    
                    # Generate example maps if enabled
                    if GENERATE_EXAMPLES and rep < NUM_EXAMPLES_PER_TIME:
                        _save_example_map(
                            monkey=monkey,
                            dataset=label,
                            method=method_name,
                            time_window=time_window,
                            rep=rep,
                            sampled_cols=sampled_cols,
                            data_matrix=data_matrix,
                            recovered_mm=recovered_mm,
                            utah_mm=utah_mm,
                            colors_w_alpha=colors_w_alpha,
                            corr_val=corr_val,
                            rms_val=rms_val,
                        )

    performance_results = OrderedDict(
        [
            ("DATA_corr_results", cortical_corr),
            ("DATA_eu_results", cortical_rms),
            ("DATA_corr_visual_results", visual_corr),
            ("DATA_eu_visual_results", visual_rms),
            ("DATA_window_results", window_tracker),
            ("Dimensions Order", "Data, Method, Time windows, Repetitions"),
            ("NUM_REPS", NUM_REPS),
            ("DATANAMES", DATASET_LABELS_USE),
            ("NUM_POINTS_LIST", SAMPLE_COUNTS.tolist()),
            ("methodNames", METHOD_NAMES),
            ("time_windows_seconds", TIME_WINDOWS_SECONDS.tolist()),
            ("sampling_strategy", "continuous_chunks" if USE_CONTINUOUS_CHUNKS else "random"),
            ("visual_field_params", WEDGE_PARAMS),
            ("rf_positions", rfs_deg),
            ("utah_positions", utah_mm),
            ("colors_w_alpha", colors_w_alpha),
            ("best_H", best_H),
            ("best_iedc", best_iedc),
            ("within_array_cortical_iedc", within_array_cortical_iedc),
            ("within_array_cortical_rms", within_array_cortical_rms),
            ("within_array_visual_iedc", within_array_visual_iedc),
            ("within_array_visual_rms", within_array_visual_rms),
        ]
    )
    return performance_results


# -----------------------------------------------------------------------------
# Plotting helpers
# -----------------------------------------------------------------------------

def _plot_time_performance_combined(
    monkey: str,
    results: OrderedDict,
    metric_key: str,
    ylabel: str,
    filename_suffix: str,
) -> None:
    """Plot performance across time windows with all methods in one subplot per dataset."""
    data = results[metric_key]  # Shape: (n_datasets, n_methods, n_windows, reps)
    datanames = results["DATANAMES"]
    methods = results["methodNames"]
    time_windows = np.array(results["time_windows_seconds"])
    
    # Reorder methods: UMAP, PCA, MDS
    method_order = ['UMAP', 'PCA', 'MDS']
    methods_sorted = [m for m in method_order if m in methods]
    
    # Define colors for methods
    method_colors = {
        'PCA': '#ff7f0e',      # Orange
        'UMAP': '#4B0082',     # Dark purple
        'MDS': '#40E0D0'       # Turquoise
    }
    
    # Create subplots: one per dataset (gamma and MUA)
    n_datasets = len(datanames)
    fig, axes = plt.subplots(1, n_datasets, figsize=(7 * n_datasets, 6), dpi=200)
    if n_datasets == 1:
        axes = [axes]
    
    for d_idx, dataset_name in enumerate(datanames):
        ax = axes[d_idx]
        
        n_windows = len(time_windows)
        n_methods = len(methods_sorted)
        
        # Prepare data for plotting
        positions = []
        box_data_all = []
        colors_all = []
        method_positions = {method: [] for method in methods_sorted}
        
        # Overlapping boxplots - all methods at same x-position per time window
        width = 0.12
        for tw_idx, tw in enumerate(time_windows):
            pos = tw  # Use actual time value as position
            for sorted_idx, method in enumerate(methods_sorted):
                method_idx = methods.index(method)
                positions.append(pos)
                method_positions[method].append(pos)
                box_data_all.append(data[d_idx, method_idx, tw_idx, :])
                colors_all.append(method_colors.get(method, 'black'))
        
        # Create boxplot
        bp = ax.boxplot(box_data_all, positions=positions, widths=width,
                       patch_artist=True, showfliers=False, zorder=4)
        
        # Color the boxes
        for patch, color in zip(bp['boxes'], colors_all):
            patch.set_facecolor(color)
            patch.set_alpha(1.0)
        for median in bp['medians']:
            median.set_color('black')
            median.set_linewidth(2)
        
        # Add black outline to boxes
        for box in bp['boxes']:
            box.set_edgecolor('black')
            box.set_linewidth(1.5)
        
        # Add connecting lines between medians for each method
        for method in methods_sorted:
            method_idx = methods.index(method)
            median_values = []
            x_positions = method_positions[method]
            
            # Calculate median for each time window for this method
            for tw_idx in range(n_windows):
                median_val = np.median(data[d_idx, method_idx, tw_idx, :])
                median_values.append(median_val)
            
            # Draw thicker colored line below
            ax.plot(x_positions, median_values, color=method_colors.get(method, 'black'), 
                   linewidth=3.5, alpha=0.8, zorder=2, linestyle='-')
            
            # Draw thinner black line on top
            ax.plot(x_positions, median_values, color='black', linewidth=1.5, 
                   alpha=0.8, zorder=3, linestyle='-')
        
        # Set monkey label
        monkey_label = 'Monkey L' if monkey == 'monkey_L' else 'Monkey A'
        
        # Formatting
        ax.set_title(f"{monkey_label} - {dataset_name}", fontsize=16, fontweight='bold', pad=15)
        ax.set_xlabel("Time Window (s)", fontsize=14, fontweight='semibold')
        ax.set_ylabel(ylabel, fontsize=14, fontweight='semibold')
        # X-axis: show clean multiples of 5 based on data range
        max_time = time_windows.max()
        tick_values = [0, 5, 10, 15]
        tick_values = [t for t in tick_values if t <= max_time]
        ax.set_xticks(tick_values)
        ax.set_xticklabels([str(t) for t in tick_values])
        ax.tick_params(axis='both', labelsize=13)
        ax.set_facecolor('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        ax.grid(False)
        # Set y-axis limits based on metric type
        if "IEDC" in ylabel:
            ax.set_ylim(0.0, 1.0)
        elif "deg" in ylabel:
            ax.set_ylim(0.0, 2.0)
        elif "mm" in ylabel:
            ax.set_ylim(0, 7)
        # Add legend to first subplot
        if d_idx == 0:
            from matplotlib.lines import Line2D
            legend_elements = [Line2D([0], [0], color=method_colors.get(method, 'black'), 
                                     lw=4, label=method) for method in methods_sorted]
            legend = ax.legend(handles=legend_elements, loc='upper right', fontsize=13, 
                             frameon=False)
            for text, method in zip(legend.get_texts(), methods_sorted):
                text.set_color(method_colors.get(method, 'black'))
    
    fig.tight_layout()
    sampling_suffix = "continuous" if USE_CONTINUOUS_CHUNKS else "random"
    output = RESULTS_DIR / f"{monkey}_{filename_suffix}_combined_{sampling_suffix}.png"
    fig.savefig(output, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved combined plot: {output.name}")


def _plot_metric_grid(
    monkey: str,
    results: OrderedDict,
    metric_key: str,
    ylabel: str,
    filename_suffix: str,
    cmap: str = "viridis",
) -> None:
    data = results[metric_key]
    datanames = results["DATANAMES"]
    methods = results["methodNames"]
    time_windows = np.array(results["time_windows_seconds"])
    mean_data = data.mean(axis=-1)  # average across repetitions

    fig, axes = plt.subplots(len(datanames), len(methods), figsize=(16, 9), dpi=200)
    if axes.ndim == 1:
        axes = axes[:, None]

    # Define colors for methods - orange for PCA, dark purple for UMAP, turquoise for MDS
    method_colors = {
        'PCA': '#ff7f0e',      # Orange
        'UMAP': '#4B0082',     # Dark purple
        'MDS': '#40E0D0'       # Turquoise
    }

    for d_idx, label in enumerate(datanames):
        for m_idx, method in enumerate(methods):
            ax = axes[d_idx, m_idx]
            values = mean_data[d_idx, m_idx]
            reps = data[d_idx, m_idx]  # shape (time_windows, reps)
            color = method_colors.get(method, 'black')
            
            # Add boxplots with higher z-order (fully opaque)
            for i, tw in enumerate(time_windows):
                bp = ax.boxplot(reps[i, :], positions=[tw], widths=0.15, patch_artist=True, showfliers=False, zorder=4)
                for patch in bp['boxes']:
                    patch.set_facecolor(color)
                    patch.set_alpha(1.0)
                    patch.set_edgecolor('black')
                    patch.set_linewidth(1.5)
                for median in bp['medians']:
                    median.set_color('black')
                    median.set_linewidth(2)
            
            # Plot colored line (thicker, below)
            ax.plot(time_windows, values, linewidth=3.5, color=color, alpha=0.8, zorder=2)
            
            # Plot black line on top (thinner)
            ax.plot(time_windows, values, linewidth=1.5, color='black', alpha=0.8, zorder=3)
            
            ax.set_title(f"{label} | {method}", fontsize=14, fontweight='semibold')
            ax.set_xlabel("Time window (s)", fontsize=12, fontweight='semibold')
            ax.set_ylabel(ylabel, fontsize=12, fontweight='semibold')
            ax.tick_params(axis='both', labelsize=10)
            ax.set_facecolor('white')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(1.5)
            ax.spines['bottom'].set_linewidth(1.5)
            ax.grid(False)
            
            # Set y-axis limits based on metric type
            if "IEDC" in ylabel:
                ax.set_ylim(0.25, 1)
            elif "RMSE (deg)" in ylabel:
                ax.set_ylim(0.0, 2)
            elif "RMSE (mm)" in ylabel:
                ax.set_ylim(0, 6)

    fig.suptitle(f"{ylabel} - {monkey}", fontsize=16, fontweight='bold', y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    output = RESULTS_DIR / f"{monkey}_{filename_suffix}.png"
    fig.savefig(output, dpi=300, bbox_inches='tight')
    plt.close(fig)


def generate_summary_plots(monkey: str, results: OrderedDict) -> None:
    print(f"Creating summary plots for {monkey}")
    
    # Combined plots (all methods in one subplot per dataset)
    _plot_time_performance_combined(monkey, results, "DATA_corr_results", "IEDC (cortical)", "cortical_iedc")
    _plot_time_performance_combined(monkey, results, "DATA_eu_results", "RMSE (mm)", "cortical_rmse")
    _plot_time_performance_combined(monkey, results, "DATA_corr_visual_results", "IEDC (visual)", "visual_iedc")
    _plot_time_performance_combined(monkey, results, "DATA_eu_visual_results", "RMSE (deg)", "visual_rmse")
    
    # Original grid plots (separate subplots per method - keeping for reference if needed)
    # _plot_metric_grid(monkey, results, "DATA_corr_results", "IEDC (cortical)", "cortical_corr")
    # _plot_metric_grid(monkey, results, "DATA_eu_results", "RMSE (mm)", "cortical_rmse")
    # _plot_metric_grid(monkey, results, "DATA_corr_visual_results", "IEDC (visual)", "visual_corr")
    # _plot_metric_grid(monkey, results, "DATA_eu_visual_results", "RMSE (deg)", "visual_rmse")
    
    _plot_within_array_metrics(monkey, results)
    _plot_best_maps(monkey, results)


def _plot_within_array_metrics(monkey: str, results: OrderedDict) -> None:
    """Plot within-array performance metrics for individual electrode arrays."""
    print(f"  Plotting within-array metrics for {monkey}")

    # Combined plots
    _plot_within_array_time_performance_combined(
        monkey, results, "within_array_cortical_iedc", "Within-Array IEDC (cortical)", "within_array_cortical_iedc"
    )
    _plot_within_array_time_performance_combined(
        monkey, results, "within_array_cortical_rms", "Within-Array RMSE (mm)", "within_array_cortical_rms"
    )
    _plot_within_array_time_performance_combined(
        monkey, results, "within_array_visual_iedc", "Within-Array IEDC (visual)", "within_array_visual_iedc"
    )
    _plot_within_array_time_performance_combined(
        monkey, results, "within_array_visual_rms", "Within-Array RMSE (deg)", "within_array_visual_rms"
    )

    # Original grid plots (keeping for reference if needed)
    # _plot_within_array_metric_grid(
    #     monkey, results, "within_array_cortical_iedc", "Within-Array IEDC (cortical)", "within_array_cortical_iedc"
    # )
    # _plot_within_array_metric_grid(
    #     monkey, results, "within_array_cortical_rms", "Within-Array RMSE (mm)", "within_array_cortical_rms"
    # )
    # _plot_within_array_metric_grid(
    #     monkey, results, "within_array_visual_iedc", "Within-Array IEDC (visual)", "within_array_visual_iedc"
    # )
    # _plot_within_array_metric_grid(
    #     monkey, results, "within_array_visual_rms", "Within-Array RMSE (deg)", "within_array_visual_rms"
    # )


def _plot_within_array_time_performance_combined(
    monkey: str,
    results: OrderedDict,
    metric_key: str,
    ylabel: str,
    filename_suffix: str,
) -> None:
    """Plot within-array performance across time windows with all methods in one subplot per dataset."""
    data = results[metric_key]  # Shape: (n_datasets, n_methods, n_windows, n_arrays, reps)
    datanames = results["DATANAMES"]
    methods = results["methodNames"]
    time_windows = np.array(results["time_windows_seconds"])
    
    # Average across arrays and repetitions
    plot_data = data.mean(axis=(-1, -2))  # Shape: (n_datasets, n_methods, n_windows)
    
    # Reorder methods: UMAP, PCA, MDS
    method_order = ['UMAP', 'PCA', 'MDS']
    methods_sorted = [m for m in method_order if m in methods]
    
    # Define colors for methods
    method_colors = {
        'PCA': '#ff7f0e',      # Orange
        'UMAP': '#4B0082',     # Dark purple
        'MDS': '#40E0D0'       # Turquoise
    }
    
    # Create subplots: one per dataset
    n_datasets = len(datanames)
    fig, axes = plt.subplots(1, n_datasets, figsize=(7 * n_datasets, 6), dpi=200)
    if n_datasets == 1:
        axes = [axes]
    
    for d_idx, dataset_name in enumerate(datanames):
        ax = axes[d_idx]
        
        n_windows = len(time_windows)
        
        # Prepare data for plotting
        positions = []
        box_data_all = []
        colors_all = []
        method_positions = {method: [] for method in methods_sorted}
        
        # Overlapping boxplots
        width = 0.12
        for tw_idx, tw in enumerate(time_windows):
            pos = tw
            for sorted_idx, method in enumerate(methods_sorted):
                method_idx = methods.index(method)
                positions.append(pos)
                method_positions[method].append(pos)
                # Get all reps for this time window, averaging across arrays
                reps_data = data[d_idx, method_idx, tw_idx, :, :].mean(axis=0)  # Average across arrays
                box_data_all.append(reps_data)
                colors_all.append(method_colors.get(method, 'black'))
        
        # Create boxplot
        bp = ax.boxplot(box_data_all, positions=positions, widths=width,
                       patch_artist=True, showfliers=False, zorder=4)
        
        # Color the boxes
        for patch, color in zip(bp['boxes'], colors_all):
            patch.set_facecolor(color)
            patch.set_alpha(1.0)
        for median in bp['medians']:
            median.set_color('black')
            median.set_linewidth(2)
        
        # Add black outline to boxes
        for box in bp['boxes']:
            box.set_edgecolor('black')
            box.set_linewidth(1.5)
        
        # Add connecting lines between medians for each method
        for method in methods_sorted:
            method_idx = methods.index(method)
            median_values = []
            x_positions = method_positions[method]
            
            for tw_idx in range(n_windows):
                # Calculate median from the averaged-across-arrays data
                reps_data = data[d_idx, method_idx, tw_idx, :, :].mean(axis=0)
                median_val = np.median(reps_data)
                median_values.append(median_val)
            
            # Draw thicker colored line below
            ax.plot(x_positions, median_values, color=method_colors.get(method, 'black'), 
                   linewidth=3.5, alpha=0.8, zorder=2, linestyle='-')
            
            # Draw thinner black line on top
            ax.plot(x_positions, median_values, color='black', linewidth=1.5, 
                   alpha=0.8, zorder=3, linestyle='-')
        
        # Set monkey label
        monkey_label = 'Monkey L' if monkey == 'monkey_L' else 'Monkey A'
        
        # Formatting
        ax.set_title(f"{monkey_label} - {dataset_name}", fontsize=16, fontweight='bold', pad=15)
        ax.set_xlabel("Time Window (s)", fontsize=14, fontweight='semibold')
        ax.set_ylabel(ylabel, fontsize=14, fontweight='semibold')
        # X-axis: show clean multiples of 5 based on data range
        max_time = time_windows.max()
        tick_values = [0, 5, 10, 15]
        tick_values = [t for t in tick_values if t <= max_time]
        ax.set_xticks(tick_values)
        ax.set_xticklabels([str(t) for t in tick_values])
        ax.tick_params(axis='both', labelsize=13)
        ax.set_facecolor('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        ax.grid(False)
        # Set y-axis limits based on metric type
        if "IEDC" in ylabel:
            ax.set_ylim(0.0, 1.0)
        elif "deg" in ylabel:
            ax.set_ylim(0.0, 2.0)
        elif "mm" in ylabel:
            ax.set_ylim(0, 3.5)  # Within-array typically lower
        
        # Add legend to first subplot
        if d_idx == 0:
            from matplotlib.lines import Line2D
            legend_elements = [Line2D([0], [0], color=method_colors.get(method, 'black'), 
                                     lw=4, label=method) for method in methods_sorted]
            legend = ax.legend(handles=legend_elements, loc='upper right', fontsize=13, 
                             frameon=False)
            for text, method in zip(legend.get_texts(), methods_sorted):
                text.set_color(method_colors.get(method, 'black'))
    
    fig.tight_layout()
    sampling_suffix = "continuous" if USE_CONTINUOUS_CHUNKS else "random"
    output = RESULTS_DIR / f"{monkey}_{filename_suffix}_combined_{sampling_suffix}.png"
    fig.savefig(output, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved combined within-array plot: {output.name}")


def _plot_within_array_metric_grid(
    monkey: str,
    results: OrderedDict,
    metric_key: str,
    ylabel: str,
    filename_suffix: str,
) -> None:
    """Plot within-array metrics across datasets, methods, and time windows."""
    data = results[metric_key]  # Shape: (n_data, n_methods, n_windows, n_arrays, n_reps)
    datanames = results["DATANAMES"]
    methods = results["methodNames"]
    time_windows = np.array(results["time_windows_seconds"])

    # Average across repetitions and arrays to get overall within-array performance
    mean_data = data.mean(axis=(-1, -2))  # Average across arrays and repetitions

    fig, axes = plt.subplots(len(datanames), len(methods), figsize=(16, 9), dpi=200)
    if axes.ndim == 1:
        axes = axes[:, None]

    # Define colors for methods - orange for PCA, dark purple for UMAP, turquoise for MDS
    method_colors = {
        'PCA': '#ff7f0e',      # Orange
        'UMAP': '#4B0082',     # Dark purple
        'MDS': '#40E0D0'       # Turquoise
    }

    for d_idx, label in enumerate(datanames):
        for m_idx, method in enumerate(methods):
            ax = axes[d_idx, m_idx]
            values = mean_data[d_idx, m_idx]
            reps = data[d_idx, m_idx].mean(axis=-2)  # Average across arrays, keep time windows and reps
            color = method_colors.get(method, 'black')

            # Add boxplots with higher z-order (fully opaque)
            for i, tw in enumerate(time_windows):
                bp = ax.boxplot(reps[i, :], positions=[tw], widths=0.15, patch_artist=True, showfliers=False, zorder=4)
                for patch in bp['boxes']:
                    patch.set_facecolor(color)
                    patch.set_alpha(1.0)
                    patch.set_edgecolor('black')
                    patch.set_linewidth(1.5)
                for median in bp['medians']:
                    median.set_color('black')
                    median.set_linewidth(2)
            
            # Plot colored line (thicker, below)
            ax.plot(time_windows, values, linewidth=3.5, color=color, alpha=0.8, zorder=2)
            
            # Plot black line on top (thinner)
            ax.plot(time_windows, values, linewidth=1.5, color='black', alpha=0.8, zorder=3)
            
            ax.set_title(f"{label} | {method}", fontsize=14, fontweight='semibold')
            ax.set_xlabel("Time window (s)", fontsize=12, fontweight='semibold')
            ax.set_ylabel(ylabel, fontsize=12, fontweight='semibold')
            ax.tick_params(axis='both', labelsize=10)
            ax.set_facecolor('white')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(1.5)
            ax.spines['bottom'].set_linewidth(1.5)
            ax.grid(False)
            
            # Set y-axis limits based on metric type
            if "IEDC" in ylabel:
                ax.set_ylim(0.25, 1)
            elif "RMSE (deg)" in ylabel:
                ax.set_ylim(0.0, 2)
            elif "RMSE (mm)" in ylabel:
                ax.set_ylim(0, 6)

    fig.suptitle(f"{ylabel} - {monkey}", fontsize=16, fontweight='bold', y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    output = RESULTS_DIR / f"{monkey}_{filename_suffix}.png"
    fig.savefig(output, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved {filename_suffix} plot to {output}")


def _plot_best_maps(monkey: str, results: OrderedDict) -> None:
    # Find the best condition across all
    best_score = -np.inf
    best_key = None
    for key, score in results['best_iedc'].items():
        if score > best_score:
            best_score = score
            best_key = key

    if best_key is None:
        print("No best key found")
        return

    dataset, method, time_window = best_key
    H = results['best_H'][best_key]
    rf_positions = results['rf_positions']  # Assuming already in degrees, or convert if needed
    utah_positions = results['utah_positions']
    colors_w_alpha = results['colors_w_alpha']

    # Get RMSE for this condition
    # Find the indices
    data_idx = results['DATANAMES'].index(dataset)
    method_idx = results['methodNames'].index(method)
    window_idx = results['time_windows_seconds'].index(time_window)
    
    rms_cortical = results['DATA_eu_results'][data_idx, method_idx, window_idx].mean()
    iedc_visual = results['DATA_corr_visual_results'][data_idx, method_idx, window_idx].mean()
    rms_visual = results['DATA_eu_visual_results'][data_idx, method_idx, window_idx].mean()

    # Project H to visual field
    projected = project_to_visual_field(H)

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # 1. Utah array ground truth
    axes[0, 0].scatter(utah_positions[:, 0], utah_positions[:, 1], c=colors_w_alpha, s=10, alpha=0.7, edgecolor='black', linewidth=0.5)
    axes[0, 0].set_title('Utah Array Ground Truth\n(cortical positions)', fontsize=12, fontweight='semibold')
    axes[0, 0].set_xlabel('X (mm)', fontsize=10)
    axes[0, 0].set_ylabel('Y (mm)', fontsize=10)
    axes[0, 0].set_aspect('equal')
    axes[0, 0].set_facecolor('white')
    axes[0, 0].spines[['top', 'right']].set_visible(False)
    axes[0, 0].grid(False)

    # 2. Aligned recovered H in cortex
    axes[0, 1].scatter(H[:, 0], H[:, 1], c=colors_w_alpha, s=10, alpha=0.7, edgecolor='black', linewidth=0.5)
    axes[0, 1].set_title(f'Aligned Recovered H in Cortex\nIEDC: {best_score:.3f}, RMSE: {rms_cortical:.3f} mm', fontsize=12, fontweight='semibold')
    axes[0, 1].set_xlabel('X (mm)', fontsize=10)
    axes[0, 1].set_ylabel('Y (mm)', fontsize=10)
    axes[0, 1].set_aspect('equal')
    axes[0, 1].set_facecolor('white')
    axes[0, 1].spines[['top', 'right']].set_visible(False)
    axes[0, 1].grid(False)

    # 3. RF ground truth
    # Add concentric circles and crosshairs
    max_radius = np.max(np.sqrt(rf_positions[:, 0]**2 + rf_positions[:, 1]**2)) + 1
    for r in np.arange(1, max_radius, 1):
        circle = Circle((0, 0), r, fill=False, color='gray', linewidth=0.5, alpha=0.5)
        axes[1, 0].add_patch(circle)
    axes[1, 0].axhline(0, color='gray', linewidth=0.5, alpha=0.5)
    axes[1, 0].axvline(0, color='gray', linewidth=0.5, alpha=0.5)
    
    axes[1, 0].scatter(rf_positions[:, 0], rf_positions[:, 1], c=colors_w_alpha, s=20, edgecolor='black', linewidth=0.5)
    axes[1, 0].set_title('RF Ground Truth\n(visual field positions)', fontsize=12, fontweight='semibold')
    axes[1, 0].set_xlabel('X (deg)', fontsize=10)
    axes[1, 0].set_ylabel('Y (deg)', fontsize=10)
    axes[1, 0].set_xlim(-1, 6.5)
    axes[1, 0].set_ylim(-6.5, 1)
    axes[1, 0].set_aspect('equal')
    axes[1, 0].set_facecolor('white')
    axes[1, 0].spines[['top', 'right']].set_visible(False)
    axes[1, 0].grid(False)

    # Project Utah ground truth to visual field for comparison
    utah_projected = project_to_visual_field(utah_positions)

    # 4. Projected visual field map with ground truth overlay
    # Add concentric circles and crosshairs
    max_radius_proj = max(
        np.max(np.sqrt(projected[:, 0]**2 + projected[:, 1]**2)) + 1,
        np.max(np.sqrt(utah_projected[:, 0]**2 + utah_projected[:, 1]**2)) + 1
    )
    for r in np.arange(1, max_radius_proj, 1):
        circle = Circle((0, 0), r, fill=False, color='gray', linewidth=0.5, alpha=0.5)
        axes[1, 1].add_patch(circle)
    axes[1, 1].axhline(0, color='gray', linewidth=0.5, alpha=0.5)
    axes[1, 1].axvline(0, color='gray', linewidth=0.5, alpha=0.5)
    
    # Plot projected recovered map (solid circles)
    axes[1, 1].scatter(projected[:, 0], projected[:, 1], c=colors_w_alpha, s=20, alpha=0.8, marker='o', edgecolor='black', linewidth=0.5, label='Recovered projected')
    
    axes[1, 1].legend(loc='upper right', fontsize=8)
    axes[1, 1].set_title(f'Projected Visual Field Map\nIEDC: {iedc_visual:.3f}, RMSE: {rms_visual:.3f} deg', fontsize=12, fontweight='semibold')
    axes[1, 1].set_xlabel('X (deg)', fontsize=10)
    axes[1, 1].set_ylabel('Y (deg)', fontsize=10)
    axes[1, 1].set_xlim(-1, 6.5)
    axes[1, 1].set_ylim(-6.5, 1)
    axes[1, 1].set_aspect('equal')
    axes[1, 1].set_facecolor('white')
    axes[1, 1].spines[['top', 'right']].set_visible(False)
    axes[1, 1].grid(False)

    fig.suptitle(f'Best Recovered Maps - {monkey}\n{dataset} | {method} | {time_window}s time window', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    sampling_suffix = "continuous" if USE_CONTINUOUS_CHUNKS else "random"
    output = RESULTS_DIR / f"{monkey}_best_maps_{sampling_suffix}.png"
    fig.savefig(output, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved best maps plot to {output}")


# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------

def create_stitched_time_comparison_figures(results_monkey_L: OrderedDict, results_monkey_A: OrderedDict) -> None:
    """Create stitched comparison figures combining Monkey L and Monkey A results across time.
    
    Creates separate figures for gamma and MUA, with 4 metric types each:
    - IEDC Cortical
    - RMSE Cortical  
    - IEDC Visual Field
    - RMSE Visual Field
    
    Each figure has layout: Row 1 = Monkey L (global, within-array), Row 2 = Monkey A (global, within-array)
    Each subplot shows all three methods (UMAP, PCA, MDS) combined.
    """
    from matplotlib.lines import Line2D
    
    # Define metric configurations
    metrics = [
        {
            'name': 'IEDC_Cortical',
            'monkey_L_global_data': results_monkey_L["DATA_corr_results"],
            'monkey_L_within_data': results_monkey_L["within_array_cortical_iedc"],
            'monkey_A_global_data': results_monkey_A["DATA_corr_results"],
            'monkey_A_within_data': results_monkey_A["within_array_cortical_iedc"],
            'ylabel': 'IEDC',
            'ylim_global': (0.0, 1.0),
            'ylim_within': (0.0, 1.0),
            'title': 'IEDC - Cortical Mapping Performance'
        },
        {
            'name': 'RMSE_Cortical',
            'monkey_L_global_data': results_monkey_L["DATA_eu_results"],
            'monkey_L_within_data': results_monkey_L["within_array_cortical_rms"],
            'monkey_A_global_data': results_monkey_A["DATA_eu_results"],
            'monkey_A_within_data': results_monkey_A["within_array_cortical_rms"],
            'ylabel': 'RMSE (mm)',
            'ylim_global': (0, 7),
            'ylim_within': (0, 3.5),
            'title': 'RMSE - Cortical Mapping Performance'
        },
        {
            'name': 'IEDC_Visual',
            'monkey_L_global_data': results_monkey_L["DATA_corr_visual_results"],
            'monkey_L_within_data': results_monkey_L["within_array_visual_iedc"],
            'monkey_A_global_data': results_monkey_A["DATA_corr_visual_results"],
            'monkey_A_within_data': results_monkey_A["within_array_visual_iedc"],
            'ylabel': 'IEDC',
            'ylim_global': (0.0, 1.0),
            'ylim_within': (0.0, 1.0),
            'title': 'IEDC - Visual Field Mapping Performance'
        },
        {
            'name': 'RMSE_Visual',
            'monkey_L_global_data': results_monkey_L["DATA_eu_visual_results"],
            'monkey_L_within_data': results_monkey_L["within_array_visual_rms"],
            'monkey_A_global_data': results_monkey_A["DATA_eu_visual_results"],
            'monkey_A_within_data': results_monkey_A["within_array_visual_rms"],
            'ylabel': 'RMSE (deg)',
            'ylim_global': (0, 2.0),
            'ylim_within': (0, 2.0),
            'title': 'RMSE - Visual Field Mapping Performance'
        }
    ]
    
    time_windows = np.array(results_monkey_L["time_windows_seconds"])
    methods = results_monkey_L["methodNames"]
    datanames = results_monkey_L["DATANAMES"]  # gamma and MUA
    
    # Reorder methods: UMAP, PCA, MDS
    method_order = ['UMAP', 'PCA', 'MDS']
    methods_sorted = [m for m in method_order if m in methods]
    
    # Define colors for methods
    method_colors = {
        'PCA': '#ff7f0e',      # Orange
        'UMAP': '#4B0082',     # Dark purple
        'MDS': '#40E0D0'       # Turquoise
    }
    
    # Create separate figures for each dataset (gamma and MUA)
    for d_idx, dataset_name in enumerate(datanames):
        # Create each stitched figure for this dataset
        for metric_config in metrics:
            # 2 rows (monkey_L, monkey_A) x 2 cols (global, within-array)
            fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=200)
            fig.suptitle(f"{metric_config['title']} - {dataset_name}", fontsize=18, fontweight='bold', y=0.995)
            
            # Define subplot configurations
            subplot_configs = [
                (0, 0, metric_config['monkey_L_global_data'], 'Monkey L', 'Global Performance'),
                (0, 1, metric_config['monkey_L_within_data'], 'Monkey L', 'Within-Array Performance'),
                (1, 0, metric_config['monkey_A_global_data'], 'Monkey A', 'Global Performance'),
                (1, 1, metric_config['monkey_A_within_data'], 'Monkey A', 'Within-Array Performance'),
            ]
            
            for row, col, data, monkey_label, subplot_title in subplot_configs:
                ax = axes[row, col]
                
                # Handle within-array data
                if data.ndim == 5:  # within-array: (n_datasets, n_methods, n_windows, n_arrays, reps)
                    plot_data = data.mean(axis=-2)  # Average across arrays: (n_datasets, n_methods, n_windows, reps)
                else:  # global: (n_datasets, n_methods, n_windows, reps)
                    plot_data = data
                
                # Extract data for this specific dataset only
                dataset_data = plot_data[d_idx]  # Shape: (n_methods, n_windows, reps)
                
                n_windows = len(time_windows)
                n_methods = len(methods_sorted)
                positions = []
                box_data_all = []
                colors_all = []
                method_positions = {method: [] for method in methods_sorted}
                
                # Overlapping boxplots - all methods at same x-position per time window (like frequency script)
                width = 0.25  # Match width from frequency script
                for tw_idx, tw in enumerate(time_windows):
                    pos = tw  # Use actual time value as position
                    for sorted_idx, method in enumerate(methods_sorted):
                        method_idx = methods.index(method)
                        positions.append(pos)
                        method_positions[method].append(pos)
                        box_data_all.append(dataset_data[method_idx, tw_idx, :])
                        colors_all.append(method_colors.get(method, 'black'))
                
                # Create boxplot
                bp = ax.boxplot(box_data_all, positions=positions, widths=width*0.8,
                               patch_artist=True, showfliers=False, zorder=4)
                
                # Color the boxes
                for patch, color in zip(bp['boxes'], colors_all):
                    patch.set_facecolor(color)
                    patch.set_alpha(1.0)
                for median in bp['medians']:
                    median.set_color('black')
                    median.set_linewidth(2)
                
                # Add black outline to boxes
                for box in bp['boxes']:
                    box.set_edgecolor('black')
                    box.set_linewidth(1.5)
                
                # Overlay individual data points on top of boxes
                # for tw_idx, tw in enumerate(time_windows):
                #     pos = tw
                #     for sorted_idx, method in enumerate(methods_sorted):
                #         method_idx = methods.index(method)
                #         # Get individual repetition values
                #         y_values = dataset_data[method_idx, tw_idx, :]
                #         # Add small jitter to x-position for visibility
                #         x_jitter = np.random.RandomState(42 + tw_idx * 10 + sorted_idx).uniform(-0.08, 0.08, size=len(y_values))
                #         x_values = np.full(len(y_values), pos) + x_jitter
                #         # Plot individual points
                #         ax.scatter(x_values, y_values, color=method_colors.get(method, 'black'), 
                #                  s=35, alpha=0.6, edgecolors='black', linewidths=0.5, zorder=5)
                
                # Add connecting lines between medians
                for method in methods_sorted:
                    method_idx = methods.index(method)
                    median_values = []
                    x_positions = method_positions[method]
                    
                    for tw_idx in range(n_windows):
                        median_val = np.median(dataset_data[method_idx, tw_idx, :])
                        median_values.append(median_val)
                    
                    # Draw thicker colored line below
                    ax.plot(x_positions, median_values, color=method_colors.get(method, 'black'), 
                           linewidth=3.5, alpha=0.8, zorder=2, linestyle='-')
                    
                    # Draw thinner black line on top
                    ax.plot(x_positions, median_values, color='black', linewidth=1.5, 
                           alpha=0.8, zorder=3, linestyle='-')
                
                # Formatting
                # X-axis: show clean multiples of 5 based on data range
                max_time = time_windows.max()
                tick_values = [0, 5, 10, 15]
                tick_values = [t for t in tick_values if t <= max_time]
                ax.set_xticks(tick_values)
                ax.set_xticklabels([str(t) for t in tick_values], fontsize=11)
                ax.set_title(f"{monkey_label} - {subplot_title}", fontsize=14, fontweight='bold', pad=10)
                ax.set_xlabel("Time Window (s)", fontsize=12, fontweight='semibold')
                ax.set_ylabel(metric_config['ylabel'], fontsize=12, fontweight='semibold')
                ax.tick_params(axis='both', labelsize=11)
                ax.set_facecolor('white')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_linewidth(1.5)
                ax.spines['bottom'].set_linewidth(1.5)
                ax.grid(False)
                
                # Set y-limits
                if col == 0:  # Global
                    ax.set_ylim(metric_config['ylim_global'])
                else:  # Within-array
                    ax.set_ylim(metric_config['ylim_within'])
                
                # Add legend to top-right subplot
                if row == 0 and col == 1:
                    legend_elements = [Line2D([0], [0], color=method_colors.get(method, 'black'), 
                                             lw=4, label=method) for method in methods_sorted]
                    legend = ax.legend(handles=legend_elements, loc='upper right', fontsize=11, 
                                     frameon=False)
                    for text, method in zip(legend.get_texts(), methods_sorted):
                        text.set_color(method_colors.get(method, 'black'))
            
            fig.tight_layout(rect=[0, 0, 1, 0.99])
            sampling_suffix = "continuous" if USE_CONTINUOUS_CHUNKS else "random"
            
            # Save PNG version
            output_png = RESULTS_DIR / f"stitched_time_comparison_{metric_config['name']}_{dataset_name}_{sampling_suffix}.png"
            fig.savefig(output_png, dpi=300, bbox_inches='tight')
            print(f"  Saved stitched time comparison PNG: {output_png.name}")
            
            # Save A4 PDF version (publication ready)
            output_pdf = RESULTS_DIR / f"stitched_time_comparison_{metric_config['name']}_{dataset_name}_{sampling_suffix}.pdf"
            # A4 size: 8.27 x 11.69 inches
            fig.set_size_inches(8.27, 11.69)
            fig.savefig(output_pdf, dpi=300, bbox_inches='tight', format='pdf')
            print(f"  Saved stitched time comparison PDF: {output_pdf.name}")
            
            plt.close(fig)



def create_combined_performance_figure() -> None:
    """Create a combined figure showing all performance metrics for both monkeys."""
    import matplotlib.image as mpimg
    
    monkeys = ["monkey_L", "monkey_A"]
    metrics = [
        ("cortical_corr", "IEDC (cortical)"),
        ("cortical_rmse", "RMSE (cortical)"), 
        ("visual_corr", "IEDC (visual)"),
        ("visual_rmse", "RMSE (visual)"),
        ("within_array_cortical_iedc", "Within-Array IEDC (cortical)"),
        ("within_array_cortical_rms", "Within-Array RMSE (cortical)"),
        ("within_array_visual_iedc", "Within-Array IEDC (visual)"),
        ("within_array_visual_rms", "Within-Array RMSE (visual)")
    ]
    
    # Create 8x2 figure (8 metrics x 2 monkeys)
    fig, axes = plt.subplots(8, 2, figsize=(24, 64), dpi=150)
    
    for row, (metric_suffix, metric_title) in enumerate(metrics):
        for col, monkey in enumerate(monkeys):
            ax = axes[row, col]
            
            # Load the individual plot image
            img_path = RESULTS_DIR / f"{monkey}_{metric_suffix}.png"
            if img_path.exists():
                img = mpimg.imread(img_path)
                ax.imshow(img)
                ax.set_title(f"{monkey} - {metric_title}", fontsize=16, fontweight='bold', pad=20)
            else:
                ax.text(0.5, 0.5, f"Plot not found:\n{img_path.name}", 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
            
            # Remove axis ticks and labels
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
    
    # Add row labels
    for row, (_, metric_title) in enumerate(metrics):
        # Add metric label on the left side
        fig.text(0.02, 0.88 - row * 0.1, metric_title, 
                ha='center', va='center', rotation=90, fontsize=14, fontweight='bold')
    
    # Add column labels  
    for col, monkey in enumerate(monkeys):
        fig.text(0.27 + col * 0.46, 0.95, monkey, 
                ha='center', va='center', fontsize=20, fontweight='bold')
    
    fig.suptitle("Performance vs Time Analysis - All Metrics & Monkeys\n(Including Within-Array Performance)", 
                fontsize=24, fontweight='bold', y=0.98)
    
    # Save the combined figure
    output_path = RESULTS_DIR / "combined_performance_all_metrics.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print(f"Saved combined performance figure to: {output_path}")


def create_within_array_combined_figure() -> None:
    """Create a combined figure showing only within-array performance metrics for both monkeys."""
    import matplotlib.image as mpimg
    
    monkeys = ["monkey_L", "monkey_A"]
    metrics = [
        ("within_array_cortical_iedc", "Within-Array IEDC (cortical)"),
        ("within_array_cortical_rms", "Within-Array RMSE (cortical)"),
        ("within_array_visual_iedc", "Within-Array IEDC (visual)"),
        ("within_array_visual_rms", "Within-Array RMSE (visual)")
    ]
    
    # Create 4x2 figure (4 metrics x 2 monkeys)
    fig, axes = plt.subplots(4, 2, figsize=(24, 32), dpi=150)
    
    for row, (metric_suffix, metric_title) in enumerate(metrics):
        for col, monkey in enumerate(monkeys):
            ax = axes[row, col]
            
            # Load the individual plot image
            img_path = RESULTS_DIR / f"{monkey}_{metric_suffix}.png"
            if img_path.exists():
                img = mpimg.imread(img_path)
                ax.imshow(img)
                ax.set_title(f"{monkey} - {metric_title}", fontsize=16, fontweight='bold', pad=20)
            else:
                ax.text(0.5, 0.5, f"Plot not found:\n{img_path.name}", 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
            
            # Remove axis ticks and labels
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
    
    # Add row labels
    for row, (_, metric_title) in enumerate(metrics):
        # Add metric label on the left side
        fig.text(0.02, 0.85 - row * 0.2, metric_title, 
                ha='center', va='center', rotation=90, fontsize=18, fontweight='bold')
    
    # Add column labels  
    for col, monkey in enumerate(monkeys):
        fig.text(0.27 + col * 0.46, 0.95, monkey, 
                ha='center', va='center', fontsize=20, fontweight='bold')
    
    fig.suptitle("Within-Array Performance Analysis - All Metrics & Monkeys", 
                fontsize=24, fontweight='bold', y=0.98)
    
    # Save the combined figure
    output_path = RESULTS_DIR / "combined_within_array_performance.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print(f"Saved within-array combined performance figure to: {output_path}")

def create_gamma_mua_iedc_figure(monkey_results: dict) -> None:
    """Create a 2x2 figure of cortical IEDC vs time for gamma and MUA datasets.

    Layout:
        Row 0: monkey_L (gamma | MUA)
        Row 1: monkey_A (gamma | MUA)

    Each panel overlays PCA (orange), UMAP (purple), and MDS (turquoise) with rep boxplots + mean lines.
    """
    if not monkey_results:
        print("No monkey results provided to create_gamma_mua_iedc_figure; skipping.")
        return

    required_datasets = ["gamma", "MUA"]
    try:
        import matplotlib.pyplot as plt  # already imported globally, safeguard
    except Exception:
        print("Matplotlib not available; skipping gamma/MUA IEDC figure.")
        return

    color_pca = "#ff7f0e"       # orange
    color_umap = "#4B0082"      # dark purple
    color_mds = "#40E0D0"       # turquoise

    monkeys_order = ["monkey_L", "monkey_A"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=180)
    fig.suptitle(
        "Cortical IEDC vs Time (Gamma & MUA)\nPCA (orange) vs UMAP (purple) vs MDS (turquoise)",
        fontsize=14,
        fontweight="bold",
        y=0.97,
    )

    for row, monkey in enumerate(monkeys_order):
        if monkey not in monkey_results:
            # blank both columns
            for col in range(2):
                ax = axes[row, col]
                ax.text(0.5, 0.5, f"No results for {monkey}", ha="center", va="center", transform=ax.transAxes)
                ax.axis("off")
            continue

        results = monkey_results[monkey]
        datanames = results["DATANAMES"]
        methods = results["methodNames"]
        time_windows = np.array(results["time_windows_seconds"])  # shape (n_windows,)
        data_corr = results["DATA_corr_results"]  # (n_data, n_methods, n_windows, reps)

        # method indices
        try:
            idx_pca = methods.index("PCA")
            idx_umap = methods.index("UMAP")
            idx_mds = methods.index("MDS")
        except ValueError:
            for col in range(2):
                ax = axes[row, col]
                ax.text(0.5, 0.5, "PCA/UMAP/MDS missing", ha="center", va="center", transform=ax.transAxes)
                ax.axis("off")
            continue

        for col, ds in enumerate(required_datasets):
            ax = axes[row, col]
            if ds not in datanames:
                ax.text(0.5, 0.5, f"{ds} missing", ha="center", va="center", transform=ax.transAxes)
                ax.axis("off")
                continue
            d_idx = datanames.index(ds)
            pca_vals = data_corr[d_idx, idx_pca]  # (n_windows, reps)
            umap_vals = data_corr[d_idx, idx_umap]
            mds_vals = data_corr[d_idx, idx_mds]
            pca_mean = pca_vals.mean(axis=-1)
            umap_mean = umap_vals.mean(axis=-1)
            mds_mean = mds_vals.mean(axis=-1)

            # Add boxplots first (higher zorder, fully opaque)
            for tw_i, tw in enumerate(time_windows):
                # PCA boxplot
                bp_pca = ax.boxplot(pca_vals[tw_i, :], positions=[tw - 0.08], widths=0.06, patch_artist=True, 
                                   showfliers=False, zorder=4)
                for patch in bp_pca['boxes']:
                    patch.set_facecolor(color_pca)
                    patch.set_alpha(1.0)
                    patch.set_edgecolor('black')
                    patch.set_linewidth(1.5)
                for median in bp_pca['medians']:
                    median.set_color('black')
                    median.set_linewidth(2)
                
                # UMAP boxplot
                bp_umap = ax.boxplot(umap_vals[tw_i, :], positions=[tw], widths=0.06, patch_artist=True, 
                                    showfliers=False, zorder=4)
                for patch in bp_umap['boxes']:
                    patch.set_facecolor(color_umap)
                    patch.set_alpha(1.0)
                    patch.set_edgecolor('black')
                    patch.set_linewidth(1.5)
                for median in bp_umap['medians']:
                    median.set_color('black')
                    median.set_linewidth(2)
                
                # MDS boxplot
                bp_mds = ax.boxplot(mds_vals[tw_i, :], positions=[tw + 0.08], widths=0.06, patch_artist=True, 
                                   showfliers=False, zorder=4)
                for patch in bp_mds['boxes']:
                    patch.set_facecolor(color_mds)
                    patch.set_alpha(1.0)
                    patch.set_edgecolor('black')
                    patch.set_linewidth(1.5)
                for median in bp_mds['medians']:
                    median.set_color('black')
                    median.set_linewidth(2)

            # Plot colored mean lines (thicker, below)
            ax.plot(time_windows, pca_mean, linewidth=3.5, color=color_pca, alpha=0.8, zorder=2, label="PCA")
            ax.plot(time_windows, umap_mean, linewidth=3.5, color=color_umap, alpha=0.8, zorder=2, label="UMAP")
            ax.plot(time_windows, mds_mean, linewidth=3.5, color=color_mds, alpha=0.8, zorder=2, label="MDS")
            
            # Plot black lines on top (thinner)
            ax.plot(time_windows, pca_mean, linewidth=1.5, color='black', alpha=0.8, zorder=3)
            ax.plot(time_windows, umap_mean, linewidth=1.5, color='black', alpha=0.8, zorder=3)
            ax.plot(time_windows, mds_mean, linewidth=1.5, color='black', alpha=0.8, zorder=3)

            ax.set_title(f"{monkey} | {ds}", fontweight="semibold", fontsize=11)
            ax.set_xlabel("Time (s)", fontsize=10)
            if col == 0:
                ax.set_ylabel("IEDC (cortical)", fontsize=10)
            ax.set_ylim(0.25, 1.0)
            ax.set_xticks(time_windows)
            ax.tick_params(labelsize=9)
            ax.set_facecolor("white")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines['left'].set_linewidth(1.5)
            ax.spines['bottom'].set_linewidth(1.5)
            ax.grid(False)
            ax.legend(frameon=False, fontsize=8, loc="lower right")

    plt.tight_layout(rect=(0, 0, 1, 0.95))
    out_path = RESULTS_DIR / "gamma_mua_cortical_iedc_summary.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved gamma/MUA cortical IEDC figure to: {out_path}")

def main() -> None:
    args = parse_args()
    do_analysis = args.do_analysis

    monkeys = ["monkey_L", "monkey_A"]
    monkey_results: dict = {}
    for monkey in monkeys:
        results = None
        if not do_analysis:
            results = load_latest_results(monkey)
            if results is None:
                print(f"No cached results found for {monkey}; running analysis anyway.")
                do_analysis = True

        if do_analysis:
            results = compute_performance_for_monkey(monkey)
            if SAVE_RESULTS:
                np.save(_results_path(monkey), results)
                with open(_results_path(monkey).with_suffix(".json"), "w", encoding="utf-8") as fh:
                    json.dump({"created": datetime.now().isoformat()}, fh, indent=2)

        if results is not None:
            generate_summary_plots(monkey, results)
            monkey_results[monkey] = results
        else:
            print(f"Failed to obtain results for {monkey}")
    
    # Create stitched comparison figures (matching frequency script style)
    if "monkey_L" in monkey_results and "monkey_A" in monkey_results:
        print("\nCreating stitched comparison figures...")
        create_stitched_time_comparison_figures(monkey_results["monkey_L"], monkey_results["monkey_A"])
    
    # Create combined figure with all metrics for both monkeys
    # create_combined_performance_figure()
    
    # Create separate combined figure for within-array metrics
    # create_within_array_combined_figure()
    
    # Create gamma/MUA specific figure
    create_gamma_mua_iedc_figure(monkey_results)
    
    print("All analyses complete.")


if __name__ == "__main__":
    main()

# %%
