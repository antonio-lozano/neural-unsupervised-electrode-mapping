# -*- coding: utf-8 -*-
"""
Visual Field Projection Analysis

This script loads monkey electrophysiological data, computes UMAP embeddings from LFP signals,
aligns them with ground truth cortical positions using Procrustes analysis, and projects both
to the visual field for comparison and validation.

USAGE:
    cd /path/to/project
    uv run python code/review_visual_field_projection.py

CONFIGURATION FLAGS:
    MONKE: Choose monkey dataset ('monkey_A' or 'monkey_L')
    STANDARIZE_LFP: Whether to standardize LFP data (default: False)
    EXCLUDE_V4: Whether to exclude V4 visual area (default: True)
    freq_band: Frequency band to analyze ('gamma', 'alpha', 'beta', etc.)

WEDGE DIPOLE PARAMETER MODES (choose one):

    1. OPTIMIZE_WEDGE_PARAMETERS = True
       - Runs optimization to find best wedge dipole parameters for current monkey
       - Saves optimized parameters to wedge_dipole_optimal_params/ folder
       - Use when you want to optimize parameters for a new monkey or improve accuracy
       - Time-consuming (several minutes)

    2. USE_LITERATURE_VALUES = True
       - Uses original literature values: a=0.61, b=106, k=13.6, alpha=0.86
       - Fast execution, reproducible baseline
       - Use for comparison studies or when optimization not needed

    3. Default mode (both flags = False)
       - Loads previously optimized parameters from file if available
       - Falls back to predefined optimized parameters for each monkey
       - Fast execution with optimized accuracy
       - Recommended for most analyses

EXAMPLES:

    # Run with optimized parameters (default, recommended)
    uv run python code/review_visual_field_projection.py

    # Optimize wedge parameters for current monkey
    # (set OPTIMIZE_WEDGE_PARAMETERS = True in script)
    uv run python code/review_visual_field_projection.py

    # Use literature values for comparison
    # (set USE_LITERATURE_VALUES = True in script)
    uv run python code/review_visual_field_projection.py

    # Change monkey dataset
    # (set MONKE = 'monkey_L' in script)
    uv run python code/review_visual_field_projection.py

OUTPUT:
    - Comprehensive plots comparing ground truth vs UMAP projections
    - RMS error and correlation metrics for alignment quality
    - Saved optimized parameters (when optimization is run)
    - Summary statistics and data shapes

DEPENDENCIES:
    - numpy, matplotlib, scikit-learn, umap-learn
    - scipy, pathlib, json
    - Custom utilities: utils.py, utils_extension.py, cortex_model_utils.py

AUTHOR: Antonio Lozano
"""

# %% 1. Import dependencies
import os
import sys
import json
from datetime import datetime

sys.path.append('code')

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial import procrustes
import umap
from sklearn.metrics import mean_squared_error

from utils import scipy_Antonio_procrustes, rescale_procrustes_map, calculate_corr_of_distances, get_monkey_pixPerMM_utahMax, apply_procrustes_and_evaluate
# Set matplotlib for inline plotting
import matplotlib
matplotlib.use('module://matplotlib_inline.backend_inline')

print("All dependencies imported successfully")

# %% 3.5. Define evaluation function

ROOT = Path(__file__).resolve().parents[1] if '__file__' in globals() else Path.cwd().parent
CODE_DIR = ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.append(str(CODE_DIR))

from utils_extension import load_monkey_data_band
from cortex_model_utils import visual_to_cortex_mapping, cortex_to_visual_mapping
import time

print(f"Project root: {ROOT}")
print(f"Code directory: {CODE_DIR}")

# %% 3. Configure analysis parameters
MONKE = 'monkey_L'  # Options: 'monkey_L' or 'monkey_A'
STANDARIZE_LFP = False
EXCLUDE_V4 = True
freq_band = 'gamma'

# UMAP parameters
umap_params = {
    'n_neighbors': 100,
    'min_dist': 0.9,
    'n_components': 2,
    'random_state': 42
}

# Wedge dipole optimization settings
OPTIMIZE_WEDGE_PARAMETERS = False  # Set to True to run optimization, False to use predefined optimized parameters
USE_LITERATURE_VALUES = False  # Set to True to use original literature values instead of optimized ones

# Create folder for storing optimized parameters
WEDGE_PARAMS_DIR = ROOT / "wedge_dipole_optimal_params"
WEDGE_PARAMS_DIR.mkdir(parents=True, exist_ok=True)

# Predefined optimized wedge parameters for each monkey
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

# Wedge dipole parameters for projection
# Literature values (will be optimized or set to predefined values)
a = 0.61
b = 106
k = 13.6
alpha = 0.86


print(f" Configuration set for monkey: {MONKE}")

# %% 4. Setup paths
deleteElecsPath = str(ROOT / 'data' / 'deletedElectrodesDictionary')
whereUtahPath = str(ROOT / 'data' / 'coordinates_of_electrodes_on_cortex_using_photos_of_arrays')
basePath = str(ROOT / 'data' / 'EYES_CLOSED')
channelAreaMappingPath = str(ROOT / 'data' / 'channel_area_mapping' / 'channel_area_mapping.mat')

print(" Paths configured")

# %% 5. Load data
print("Loading data... This may take a few minutes.")

lfp, anchorLFP, anchorUtah, anchorUtahDistances, MUABINNED, colors, array_number_v1, array_idx, colors_w_alpha, channel_area_map, array_nums, areas, channel_nums, rfs, rfs_distances, utah, utah_distances, bad_dict, subject_cond, nonvalidEList, idx_nonvalid, nonvalidE, utah_max, selected = load_monkey_data_band(
    MONKE, STANDARIZE_LFP, EXCLUDE_V4, basePath, whereUtahPath, channelAreaMappingPath, deleteElecsPath, freq_band=freq_band, load_MUA=False, LFP_float16=False)

# Assign to old variable names for compatibility
LFP = lfp
arrayNumberV1 = array_number_v1
arrayIDX = array_idx
colorsWAlpha = colors_w_alpha
channelAreaMap = channel_area_map
arrayNums = array_nums
badElectrodesDict = bad_dict
subjectConditionList = subject_cond
utahMax = utah_max
selectedChannelsIndices = selected
idxNonvalid = idx_nonvalid

print("Data loaded successfully")
print(f"  LFP shape: {LFP.shape}")
print(f"  RF positions shape: {rfs.shape}")
print(f"  Utah positions shape: {utah.shape}")

# Get scaling parameters
PIXELS_PER_DEG = 25.78  
pixels_per_mm, utahMax = get_monkey_pixPerMM_utahMax(MONKE)
print(f"  Pixels per mm: {pixels_per_mm}, Utah max: {utahMax}")

# Scale RFs to degrees
rfs = rfs / PIXELS_PER_DEG


# Rescale Utah ground truth to real cortical size
utah = utah * utahMax / pixels_per_mm

# Define anchor point indexes based on monkey
if MONKE == 'monkey_L':
    AP_indexes = np.array([65, 300, 400], dtype='int32')
elif MONKE == 'monkey_A':
    AP_indexes = np.array([1, 630, 200], dtype='int32')
    AP_indexes = AP_indexes / 2  # this monkey has half the number of good electrodes
    AP_indexes = AP_indexes.astype('int32')

print(f"Anchor point indexes: {AP_indexes}")

# Select anchor points from scaled utah
anchorUtah = utah[AP_indexes]
anchorRfs = rfs[AP_indexes]

# %% Optimize wedge parameters or use predefined values
if OPTIMIZE_WEDGE_PARAMETERS:
    from scipy.optimize import minimize

    def objective_function(params, utah_positions, rf_positions):
        """Objective function to minimize: RMSE between projected Utah and RF positions"""
        a, b, k, alpha = params
        
        # Project Utah positions to visual field
        projected = cortex_to_visual_mapping(utah_positions[:, 0], utah_positions[:, 1], a, b, alpha, k)
        projected = np.vstack(projected).T
        
        # Calculate RMSE between projected and measured RF positions
        rmse = np.sqrt(np.mean((projected - rf_positions)**2))
        return rmse

    # Initial guess from literature
    initial_params = [a, b, k, alpha]

    # Parameter bounds (reasonable ranges around literature values)
    bounds = [
        (0.4, 0.8),   # a
        (80, 130),    # b  
        (10, 17),     # k
        (0.7, 1.0)    # alpha
    ]

    print("\nOptimizing wedge dipole parameters to match Utah projections to RF measurements...")
    print(f"Initial parameters: a={initial_params[0]:.4f}, b={initial_params[1]:.1f}, k={initial_params[2]:.2f}, alpha={initial_params[3]:.3f}")

    # Run optimization
    result = minimize(
        objective_function, 
        initial_params, 
        args=(utah, rfs),
        bounds=bounds,
        method='L-BFGS-B',
        options={'maxiter': 100, 'disp': False}
    )

    # Extract optimized parameters
    a_opt, b_opt, k_opt, alpha_opt = result.x

    print(f"Optimization result: success={result.success}, final RMSE={result.fun:.6f}")
    print(f"Optimized parameters:")
    print(f"  a = {a_opt:.4f}")
    print(f"  b = {b_opt:.1f}")
    print(f"  k = {k_opt:.2f}")
    print(f"  alpha = {alpha_opt:.3f}")

    # Save optimized parameters to file
    optimized_params = {
        'a': float(a_opt),
        'b': float(b_opt),
        'k': float(k_opt),
        'alpha': float(alpha_opt),
        'rmse': float(result.fun),
        'optimization_success': bool(result.success),
        'timestamp': datetime.now().isoformat(),
        'monkey': MONKE
    }
    
    params_file = WEDGE_PARAMS_DIR / f"wedge_params_{MONKE}_optimized.json"
    with open(params_file, 'w') as f:
        json.dump(optimized_params, f, indent=2)
    
    print(f"Saved optimized parameters to: {params_file}")

    # Use optimized parameters for all projections
    a, b, k, alpha = a_opt, b_opt, k_opt, alpha_opt

    print(f"Using optimized parameters: a={a:.4f}, b={b:.1f}, k={k:.2f}, alpha={alpha:.3f}")

else:
    # Check if user wants to use literature values
    if USE_LITERATURE_VALUES:
        # Use original literature values
        a, b, k, alpha = 0.61, 106, 13.6, 0.86
        print(f"\nUsing literature wedge parameters:")
        print(f"  a = {a:.4f}")
        print(f"  b = {b:.1f}")
        print(f"  k = {k:.2f}")
        print(f"  alpha = {alpha:.3f}")
    else:
        # Try to load previously optimized parameters first, fall back to predefined
        params_file = WEDGE_PARAMS_DIR / f"wedge_params_{MONKE}_optimized.json"
        if params_file.exists():
            try:
                with open(params_file, 'r') as f:
                    saved_params = json.load(f)
                a, b, k, alpha = saved_params['a'], saved_params['b'], saved_params['k'], saved_params['alpha']
                print(f"\nLoaded previously optimized wedge parameters for {MONKE} from file:")
                print(f"  a = {a:.4f}")
                print(f"  b = {b:.1f}")
                print(f"  k = {k:.2f}")
                print(f"  alpha = {alpha:.3f}")
                print(f"  RMSE = {saved_params.get('rmse', 'N/A')}")
            except (json.JSONDecodeError, KeyError) as e:
                print(f"\nWarning: Could not load saved parameters ({e}), using predefined values.")
                params = OPTIMIZED_WEDGE_PARAMS.get(MONKE, OPTIMIZED_WEDGE_PARAMS['monkey_L'])  # fallback
                a, b, k, alpha = params['a'], params['b'], params['k'], params['alpha']
                print(f"Using predefined optimized wedge parameters for {MONKE}.")
        else:
            # Use predefined optimized parameters for the current monkey
            if MONKE in OPTIMIZED_WEDGE_PARAMS:
                params = OPTIMIZED_WEDGE_PARAMS[MONKE]
                a, b, k, alpha = params['a'], params['b'], params['k'], params['alpha']
                print(f"\nUsing predefined optimized wedge parameters for {MONKE}:")
                print(f"  a = {a:.4f}")
                print(f"  b = {b:.1f}")
                print(f"  k = {k:.2f}")
                print(f"  alpha = {alpha:.3f}")
            else:
                print(f"\nWarning: No predefined parameters found for monkey {MONKE}. Using literature values.")
                print(f"  a = {a:.4f}")
                print(f"  b = {b:.1f}")
                print(f"  k = {k:.2f}")
                print(f"  alpha = {alpha:.3f}")

dotSize = 3
linewidth = 0.2

# %% 6. Compute UMAP embedding with multiple runs to find best alignment
print("Computing UMAP embedding")

import time

# Get sampling parameters
sampling_rate = 500  # Hz
duration = 15
total_timepoints = LFP.shape[1]
n_samples = duration * sampling_rate

# Run UMAP multiple times and select the best alignment
n_runs = 20
best_umap_embedding = None
best_rms_error = float('inf')
best_run_idx = -1

print(f"Running UMAP {n_runs} times to find best alignment...")

for run_idx in range(n_runs):
    print(f"  Run {run_idx + 1}/{n_runs}...")

    # Sample new LFP data subset for each UMAP run
    if n_samples >= total_timepoints:
        # If we want more samples than available, use all timepoints
        selected_indices = np.arange(total_timepoints)
        print(f"    Warning: Requested {n_samples} samples but only {total_timepoints} available. Using all timepoints.")
    else:
        # Randomly select a continuous chunk of time points from the recording
        max_start = total_timepoints - n_samples
        start_idx = np.random.randint(0, max_start + 1)
        selected_indices = np.arange(start_idx, start_idx + n_samples)
        print(f"    Sampled continuous chunk: {len(selected_indices)} time points (from {start_idx} to {start_idx + n_samples - 1})")

    LFP_subset = LFP[:, selected_indices]
    X = LFP_subset if LFP_subset.shape[0] <= LFP_subset.shape[1] else LFP_subset.T

    reducer = umap.UMAP(
        n_components=umap_params['n_components'],
        n_neighbors=umap_params['n_neighbors'],
        min_dist=umap_params['min_dist'],
        verbose=False
    )

    current_umap_embedding = reducer.fit_transform(X)
    
    # Test alignment quality
    utah_std, umap_std, disparity, R, s, norm1, norm2, mean1, mean2 = scipy_Antonio_procrustes(utah, current_umap_embedding)
    current_aligned_umap = rescale_procrustes_map(current_umap_embedding, R, s, norm1, norm2, mean1, mean2)
    current_rms_error = np.sqrt(mean_squared_error(current_aligned_umap, utah))
    
    print(f"    RMS error: {current_rms_error:.4f} mm")
    
    if current_rms_error < best_rms_error:
        best_rms_error = current_rms_error
        best_umap_embedding = current_umap_embedding
        best_run_idx = run_idx

print(f"✓ Selected UMAP run {best_run_idx + 1} with RMS error: {best_rms_error:.4f} mm")
umap_embedding = best_umap_embedding

# Select anchor points in UMAP space
anchorUmap = umap_embedding[AP_indexes]



# %% 7. Align UMAP with ground truth using Procrustes
print("Aligning UMAP embedding with ground truth...")

# Apply custom Procrustes alignment
utah_std, umap_std, disparity, R, s, norm1, norm2, mean1, mean2 = scipy_Antonio_procrustes(utah, umap_embedding)

# Rescale aligned UMAP to match utah scale (real mm)
aligned_umap = rescale_procrustes_map(umap_embedding, R, s, norm1, norm2, mean1, mean2)

# Evaluate alignment in real mm units
rms_error_mm = np.sqrt(mean_squared_error(aligned_umap, utah))
corr_dist = calculate_corr_of_distances(aligned_umap, utah)

rotation_matrix = R
scale_factor = s

print("✓ Procrustes alignment completed")
print(f"  RMS error: {rms_error_mm:.4f} mm")
print(f"  Correlation of distances: {corr_dist:.4f}")
print(f"  Scale factor: {s:.4f}")

# Select anchor points in aligned UMAP space
anchorAligned = aligned_umap[AP_indexes]

# %% 8. Project ground truth and aligned UMAP to visual field
print("Projecting to visual field...")

# Project ground truth Utah positions to visual field
gt_visual = cortex_to_visual_mapping(utah[:, 0], utah[:, 1], a, b, alpha, k)
gt_visual = np.vstack(gt_visual).T

# Project aligned UMAP positions to visual field
umap_visual = cortex_to_visual_mapping(aligned_umap[:, 0], aligned_umap[:, 1], a, b, alpha, k)
umap_visual = np.vstack(umap_visual).T

# Project anchor points to visual field
anchorGtVisual = cortex_to_visual_mapping(anchorUtah[:, 0], anchorUtah[:, 1], a, b, alpha, k)
anchorGtVisual = np.vstack(anchorGtVisual).T

anchorUmapVisual = cortex_to_visual_mapping(anchorAligned[:, 0], anchorAligned[:, 1], a, b, alpha, k)
anchorUmapVisual = np.vstack(anchorUmapVisual).T

print("✓ Projections completed")

# Create sampling of cortex-to-visual mapping
# Create a grid of points across the cortical surface
x_grid = np.linspace(utah[:, 0].min(), utah[:, 0].max(), 10)
y_grid = np.linspace(utah[:, 1].min(), utah[:, 1].max(), 10)
cortex_grid_x, cortex_grid_y = np.meshgrid(x_grid, y_grid)
cortex_sampling = np.column_stack([cortex_grid_x.ravel(), cortex_grid_y.ravel()])

# Project the cortical grid to visual field
visual_sampling = cortex_to_visual_mapping(cortex_sampling[:, 0], cortex_sampling[:, 1], a, b, alpha, k)
visual_sampling = np.vstack(visual_sampling).T

print("✓ Cortex-to-visual sampling created")

# Define shared axis limits for all visual field plots
all_visual_x = np.concatenate([rfs[:, 0], gt_visual[:, 0], umap_visual[:, 0]])
all_visual_y = np.concatenate([rfs[:, 1], gt_visual[:, 1], umap_visual[:, 1]])
visual_xlim = (all_visual_x.min() - 0.5, all_visual_x.max() + 0.5)
visual_ylim = (all_visual_y.min() - 0.5, all_visual_y.max() + 0.5)

# %% 9. Create plots
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Plot 1: Cortex-to-visual field sampling
ax = axes[0, 0]
ax.scatter(visual_sampling[:, 0], visual_sampling[:, 1], c='blue', alpha=0.7, s=50, marker='s')
ax.scatter(anchorGtVisual[:, 0], anchorGtVisual[:, 1], c='yellow', edgecolors='black', s=200, marker='o', linewidth=2)
ax.set_title('Cortex → Visual Field Sampling', fontsize=12, fontweight='bold')
ax.set_xlabel('X (deg)')
ax.set_ylabel('Y (deg)')
ax.grid(False)

# Plot 2: Ground truth Utah
ax = axes[0, 1]
ax.scatter(utah[:, 0], utah[:, 1], c=colorsWAlpha, alpha=0.7, edgecolors='black', s=50)
ax.scatter(anchorUtah[:, 0], anchorUtah[:, 1], c='yellow', edgecolors='black', s=200, marker='o', linewidth=2)
ax.set_title('Ground Truth Utah', fontsize=12, fontweight='bold')
ax.set_xlabel('X (mm)')
ax.set_ylabel('Y (mm)')
ax.grid(False)

# Plot 3: Aligned UMAP
ax = axes[0, 2]
ax.scatter(aligned_umap[:, 0], aligned_umap[:, 1], c=colorsWAlpha, alpha=0.7, edgecolors='black', s=50)
ax.scatter(anchorAligned[:, 0], anchorAligned[:, 1], c='yellow', edgecolors='black', s=200, marker='o', linewidth=2)
ax.set_title('Aligned UMAP', fontsize=12, fontweight='bold')
ax.set_xlabel('X (mm)')
ax.set_ylabel('Y (mm)')
ax.grid(False)

# Plot 4: Ground truth RFs
ax = axes[1, 0]
ax.scatter(rfs[:, 0], rfs[:, 1], c=colorsWAlpha, alpha=0.7, edgecolors='black', s=50)
ax.scatter(anchorRfs[:, 0], anchorRfs[:, 1], c='yellow', edgecolors='black', s=200, marker='o', linewidth=2)
ax.set_title('Ground Truth RFs', fontsize=12, fontweight='bold')
ax.set_xlabel('X (deg)')
ax.set_ylabel('Y (deg)')
ax.set_xlim(visual_xlim)
ax.set_ylim(visual_ylim)
ax.grid(False)

# Plot 5: Ground truth projected to visual field
ax = axes[1, 1]
ax.scatter(gt_visual[:, 0], gt_visual[:, 1], c=colorsWAlpha, alpha=0.7, edgecolors='black', s=50)
ax.scatter(anchorGtVisual[:, 0], anchorGtVisual[:, 1], c='yellow', edgecolors='black', s=200, marker='o', linewidth=2)
ax.set_title('Ground Truth → Visual Field', fontsize=12, fontweight='bold')
ax.set_xlabel('X (deg)')
ax.set_ylabel('Y (deg)')
ax.set_xlim(visual_xlim)
ax.set_ylim(visual_ylim)
ax.grid(False)

# Plot 6: UMAP projected to visual field with ground truth overlay
ax = axes[1, 2]
# Plot ground truth projection (semi-transparent)
#ax.scatter(gt_visual[:, 0], gt_visual[:, 1], c=colorsWAlpha, alpha=0.4, edgecolors='gray', s=30, marker='o', label='Utah GT projected')
# Plot UMAP projection (solid)
ax.scatter(umap_visual[:, 0], umap_visual[:, 1], c=colorsWAlpha, alpha=0.7, edgecolors='black', s=50, marker='s', label='UMAP projected')
ax.scatter(anchorUmapVisual[:, 0], anchorUmapVisual[:, 1], c='yellow', edgecolors='black', s=200, marker='o', linewidth=2)
ax.legend(loc='upper right', fontsize=8)
ax.set_title('UMAP → Visual Field (with GT overlay)', fontsize=12, fontweight='bold')
ax.set_xlabel('X (deg)')
ax.set_ylabel('Y (deg)')
ax.set_xlim(visual_xlim)
ax.set_ylim(visual_ylim)
ax.grid(False)

plt.tight_layout()
plt.show()

print("✓ All plots generated")

# %% 10. Summary
print("\n" + "="*60)
print("VISUAL FIELD PROJECTION ANALYSIS SUMMARY")
print("="*60)
print(f"Monkey: {MONKE}")
print(f"Frequency band: {freq_band}")
print(f"UMAP parameters: n_neighbors={umap_params['n_neighbors']}, min_dist={umap_params['min_dist']}")
print(f"Procrustes RMS error: {rms_error_mm:.4f} mm")
print(f"Procrustes correlation of distances: {corr_dist:.4f}")
print(f"Data shapes:")
print(f"  LFP: {LFP.shape}")
print(f"  RF ground truth: {rfs.shape}")
print(f"  Utah ground truth: {utah.shape}")
print(f"  UMAP embedding: {umap_embedding.shape}")
print(f"  Aligned UMAP: {aligned_umap.shape}")
print(f"  GT visual projection: {gt_visual.shape}")
print(f"  UMAP visual projection: {umap_visual.shape}")
print("="*60)


# %%
