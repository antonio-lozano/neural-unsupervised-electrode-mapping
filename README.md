# Neural Unsupervised Electrode Mapping

Neural mapping analysis using dimensionality reduction techniques for primate electrophysiology data. This repository contains tools for reconstructing electrode positions from neural activity patterns and projecting them to visual field coordinates.

## Repository Contents

### Main Python Files

#### `neumap_demo_v2.py` (373 lines)
Neural mapping demo using dimensionality reduction (MDS + UMAP).

**Features:**
- Loads monkey electrophysiology data
- Performs electrode mapping reconstruction
- Evaluates cortical and visual field metrics
- Uses optimized wedge dipole parameters for each monkey

**Key Functions:**
- `resolve_monkey_name()` - Handles monkey identifier aliases
- `correlation_distance_matrix()` - Computes correlation-based distances
- `run_mds()` - Multidimensional Scaling embedding
- `run_umap_embedding()` - UMAP dimensionality reduction
- `align_to_cortex()` - Procrustes alignment to ground truth
- `project_to_visual_field()` - Cortical to visual field projection
- `compute_metrics()` - Evaluation metrics (IEDC, RMSE)

#### `review_performance_vs_time_VF.py` (1771 lines)
Performance vs time analysis with visual field projection.

**Features:**
- Tests PCA, UMAP, and MDS across different time windows
- Supports both random sampling and continuous chunk sampling
- Generates plots and saves results
- Includes within-array performance metrics

**Configuration Flags:**
- `USE_CONTINUOUS_CHUNKS` - Sampling strategy (random vs continuous)
- `GENERATE_EXAMPLES` - Save example maps for visualization
- `USE_LITERATURE_VALUES` - Wedge parameter mode selection
- `DO_ANALYSIS_DEFAULT` - Control analysis execution

**Usage:**
```bash
# Default: Random sampling with optimized wedge parameters
uv run python review_performance_vs_time_VF.py

# Force run analysis
uv run python review_performance_vs_time_VF.py --do-analysis

# Skip analysis and just generate plots from existing results
uv run python review_performance_vs_time_VF.py --no-analysis
```

#### `review_visual_field_projection.py` (464 lines)
Visual field projection analysis with comprehensive visualization.

**Features:**
- UMAP embedding with multiple runs to find best alignment
- Procrustes alignment of cortical positions
- Wedge dipole parameter optimization (optional)
- Comprehensive visualization of cortical to visual field mapping

**Wedge Parameter Modes:**
1. **OPTIMIZE_WEDGE_PARAMETERS = True** - Runs optimization to find best parameters
2. **USE_LITERATURE_VALUES = True** - Uses original literature values (a=0.61, b=106, k=13.6, alpha=0.86)
3. **Default mode (both flags = False)** - Loads previously optimized parameters

**Usage:**
```bash
# Run with optimized parameters (default, recommended)
uv run python review_visual_field_projection.py

# Optimize wedge parameters (set flag in script)
# Use literature values (set flag in script)
```

#### `utils.py` (4424 lines)
Core utility functions for data loading, processing, and analysis.

**Key Functionality:**
- SNR, MUA, and LFP data loading
- Procrustes alignment functions
- Array separation and indexing utilities
- Signal processing (filtering, binning)
- Electrode position handling

**Main Functions:**
- `load_SNR_instances_openEyes()` - Load SNR instances from LFP responses
- `load_MUA_instances()` - Load MUA (Multi-Unit Activity) data
- `load_valid_rfs()` - Load receptive field locations
- `load_valid_utahLocations()` - Load Utah array electrode positions
- `scipy_Antonio_procrustes()` - Custom Procrustes alignment
- `calculate_corr_of_distances()` - Compute distance correlation (IEDC)
- `separate_arrays_fromIndex()` - Separate data by electrode arrays

#### `utils_extension.py` (465 lines)
Extended data loading utilities with enhanced path resolution.

**Features:**
- Handles both full and 15-second data windows
- Frequency band loading (LFP, alpha, beta, gamma, highGamma)
- Path resolution and file handling
- Maintains compatibility with existing scripts

**Main Functions:**
- `load_monkey_data_band()` - Load full-duration band-specific data
- `load_monkey_data_band_15seconds()` - Load trimmed 15-second data chunk
- `_normalize_frequency()` - Standardize frequency band names
- `_resolve_condition_root()` - Auto-detect data directory structure

### Test Files

#### `tests/test_neumap_demo.py`
Tests for the original demo script.
- Validates grid pruning logic
- Checks script execution
- Requires demo dataset

#### `tests/test_neumap_demo_v2.py`
Tests for neumap_demo_v2.py.
- Validates monkey name resolution
- Tests UMAP grid generation
- Tests correlation distance matrix properties

## Key Features

### Dimensionality Reduction
- **PCA** - Principal Component Analysis
- **UMAP** - Uniform Manifold Approximation and Projection
- **MDS** - Multidimensional Scaling

### Monkey Data Support
- **Monkey L** (Monkey L) - 16 arrays
- **Monkey A** (Monkey A) - 16 arrays

### Frequency Bands
- `LFP` - Local Field Potential (original)
- `low` - Low frequency
- `alpha` - Alpha band
- `beta` - Beta band
- `gamma` - Gamma band
- `highGamma` - High gamma band

### Visual Field Projection
- Cortex-to-visual field mapping using wedge dipole model
- Optimized wedge parameters for each monkey:
  - **Monkey L**: a=0.5251, b=80.0, k=13.64, alpha=0.774
  - **Monkey A**: a=0.6857, b=106.0, k=16.69, alpha=0.948
- Literature values: a=0.61, b=106, k=13.6, alpha=0.86

### Performance Evaluation
- **IEDC** - Inter-Electrode Distance Correlation
- **RMSE** - Root Mean Squared Error
- Metrics computed in both cortical space (mm) and visual field space (degrees)

### Flexible Data Loading
- Full recording duration
- 15-second windows
- Random subsampling
- Continuous chunk sampling

## Data Structure

The code expects data organized in the following structure:
```
data/
├── EYES_CLOSED/
│   ├── Monkey_L/
│   │   ├── RFS/
│   │   ├── LFP/
│   │   └── MUA/
│   └── Monkey_A/
│       ├── RFS/
│       ├── LFP/
│       └── MUA/
├── coordinates_of_electrodes_on_cortex_using_photos_of_arrays/
├── channel_area_mapping/
└── deletedElectrodesDictionary/
```

## Installation

This project uses [UV](https://docs.astral.sh/uv/) for fast, reliable Python package management.

### Setup with UV

1. **Install UV** (if not already installed):
   ```bash
   # Windows (PowerShell)
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
   
   # macOS/Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Install dependencies**:
   ```bash
   uv sync
   ```

3. **Configure data paths**:
   
   Edit `config.yaml` to point to your data directory:
   ```yaml
   # Update the data_root path to your local data directory
   data_root: "LFP-RFs/data"
   ```
   
   The configuration file specifies paths to:
   - `eyes_closed_data` - LFP and MUA recordings during eyes-closed condition
   - `utah_coordinates` - Electrode array position data  
   - `channel_area_mapping` - Channel-to-cortical-area mapping
   - `deleted_electrodes` - Invalid electrode information

### Running Scripts with UV

Use `uv run` to execute any Python script with the project's dependencies:

```bash
# Run the main demo
uv run python neumap_demo_v2.py

# Run performance analysis
uv run python review_performance_vs_time_VF.py

# Run visual field projection analysis
uv run python review_visual_field_projection.py

# Run tests
uv run pytest tests/
```

### Required Dependencies

- numpy>=1.21.0
- matplotlib>=3.4.0
- scikit-learn>=1.0.0
- umap-learn>=0.5.0
- scipy>=1.7.0
- pandas>=1.3.0
- hdbscan>=0.8.0
- seaborn>=0.11.0

## Research Context

This repository is designed for neural electrode mapping and visual field analysis in primate electrophysiology experiments. The code reconstructs electrode positions from neural activity patterns using unsupervised learning techniques and validates them against ground truth cortical positions and receptive field measurements.

## Author

Antonio Lozano

## License

See repository for license information.
