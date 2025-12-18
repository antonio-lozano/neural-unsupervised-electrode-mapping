"""Extended data loading utilities for LFP receptive field analyses.

This module houses enhanced loader routines that preserve retro-compatibility
with existing scripts while exposing clearer function names for new analyses.
The helpers encapsulate the shared logic that was previously embedded in
ad-hoc debugging scripts so other modules can reuse the same robust path
resolution, frequency normalisation, and data assembly steps.
"""

from __future__ import annotations

import os
import pickle
from typing import Dict, List, Sequence, Tuple

import numpy as np
import scipy.io
from sklearn.metrics.pairwise import euclidean_distances

from utils import load_rf_instances, order_RF_instances, load_utah_XING

WINDOW_FULL = "full"
WINDOW_15S = "15s"
_PREFERRED_CONDITION = "EYES_CLOSED"
_VALID_FREQ_BANDS = ["LFP", "low", "alpha", "beta", "gamma", "highGamma"]


def separate_arrays_from_index(raw: np.ndarray, array_number: np.ndarray, num_anchors_per_array: int
                               ) -> Tuple[List[np.ndarray], List[int], List[np.ndarray], List[np.ndarray]]:
    """Replicates the legacy anchor selection pipeline used in downstream analyses."""

    print('Warning: indices in the arrayNumbers start on 1, not 0. Correcting for this')

    array_list: List[np.ndarray] = []
    anchor_indices: List[int] = []
    relative_anchor_idx_list: List[np.ndarray] = []
    selected_channels_indices: List[np.ndarray] = []

    available_arrays = np.unique(array_number)

    for idx in available_arrays:
        indices = np.where(array_number == idx)[0].astype(int)
        array = raw[indices]

        quantiles_indices = np.arange(0, 1, 1.0 / (num_anchors_per_array + 1))
        quantiles_indices = quantiles_indices[1:]  # skip the first one

        relative_anchor_idx = []
        for q in range(num_anchors_per_array):
            quantile = np.quantile(indices, quantiles_indices[q])
            anchor_index = indices.flat[np.abs(indices - quantile).argmin()]

            anchor_indices.append(int(anchor_index))
            anchor_index_per_array = np.where(indices == anchor_index)[0]
            relative_anchor_idx.append(anchor_index_per_array)

        relative_anchor_idx_list.append(np.array(relative_anchor_idx))
        array_list.append(array)
        selected_channels_indices.append(indices)

    return array_list, anchor_indices, relative_anchor_idx_list, selected_channels_indices


def create_array_index_new(raw: np.ndarray, cols: Sequence[Sequence[float]], num_anchors_per_array: int,
                           nonvalid: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray, np.ndarray,
                                                          np.ndarray, np.ndarray, np.ndarray, List[np.ndarray],
                                                          List[np.ndarray], np.ndarray]:
    print(raw.shape)

    color_single_array: List[np.ndarray] = []
    selected_channels_indices: List[np.ndarray] = []

    array_number = np.zeros((1024,))
    colors_new = np.zeros((1024, 3))

    counter = 0
    for i in range(16):
        color_single_array.append(np.array(cols[i]))
        for _ in range(64):
            array_number[counter] = i + 1
            colors_new[counter] = cols[i]
            counter += 1

    array_number_v1 = np.delete(array_number, nonvalid, 0)
    colors_new = np.delete(colors_new, nonvalid, 0)

    all_idx = np.linspace(0, 1023, 1024).astype('int64')
    original_idx = np.delete(all_idx, nonvalid)

    raw_arrays, anchor_indices, relative_anchor_idx_list, selected_indices = separate_arrays_from_index(
        raw, array_number_v1, num_anchors_per_array
    )

    return (
        raw,
        raw_arrays,
        np.asarray(anchor_indices, dtype=int),
        np.asarray(cols),
        colors_new,
        array_number_v1,
        np.array(color_single_array),
        relative_anchor_idx_list,
        selected_indices,
        original_idx,
    )


def _unique_paths(paths: Sequence[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for path in paths:
        norm = os.path.normpath(path)
        if norm not in seen:
            seen.add(norm)
            ordered.append(norm)
    return ordered


def _resolve_condition_root(base_path: str, monkey: str, condition: str = _PREFERRED_CONDITION) -> str:
    base_norm = os.path.normpath(base_path)
    condition_variants = [condition, condition.lower(), condition.upper()]
    candidates = _unique_paths(
        [
            base_norm,
            os.path.join(base_norm, monkey),
        ]
        + [os.path.join(base_norm, cond, monkey) for cond in condition_variants]
    )

    tried = []
    for candidate in candidates:
        rfs_dir = os.path.join(candidate, 'RFS')
        lfp_dir = os.path.join(candidate, 'LFP')
        if os.path.isdir(rfs_dir) and os.path.isdir(lfp_dir):
            return candidate
        tried.append(f"{candidate} (RFS: {os.path.isdir(rfs_dir)}, LFP: {os.path.isdir(lfp_dir)})")

    raise FileNotFoundError(
        f"Unable to locate data root for monkey '{monkey}'. Tried:\n  " + "\n  ".join(tried)
    )


def _resolve_file(directory: str, candidate_filenames: Sequence[str], description: str) -> Tuple[str, str]:
    tried = []
    for name in candidate_filenames:
        full_path = os.path.join(directory, name)
        if os.path.exists(full_path):
            return full_path, name
        tried.append(full_path)
    raise FileNotFoundError(
        f"Could not find {description} in {directory}. Tried:\n  " + "\n  ".join(tried)
    )


def _normalize_frequency(freq_band: str | None) -> str:
    mapping = {
        'original': 'LFP',
        'mattLow': 'low',
        'mattAlpha': 'alpha',
        'mattBeta': 'beta',
        'mattGamma': 'gamma',
        'mattHighGamma': 'highGamma',
        'high_gamma': 'highGamma',
        'highgamma': 'highGamma',
        None: 'LFP',
    }
    normalized = mapping.get(freq_band, freq_band)
    if normalized not in _VALID_FREQ_BANDS:
        raise ValueError(f"Invalid frequency band '{freq_band}'. Choose from {_VALID_FREQ_BANDS}.")
    return normalized


def _lfp_file_candidates(monkey: str, freq_band: str, window: str) -> List[str]:
    if window == WINDOW_15S:
        if freq_band == 'LFP':
            return [
                f'allOrderedLFP_ignore4secs_LFP_{monkey}_outlierChunks_removed_15seconds.npy',
                f'allOrderedLFP_ignore4secs_{monkey}_outlierChunks_removed_15seconds.npy',
                f'allOrderedLFP_ignore4secs_{monkey}_outliers_removed_15seconds.npy',
                f'allOrderedLFP_ignore4secs_{monkey}_15seconds.npy',
            ]
        return [f'allOrderedLFP_ignore4secs_{freq_band}_{monkey}_outlierChunks_removed_15seconds.npy']

    if freq_band == 'LFP':
        return [
            f'allOrderedLFP_ignore4secs_LFP_{monkey}_outlierChunks_removed.npy',
            f'allOrderedLFP_ignore4secs_{monkey}_outlierChunks_removed.npy',
            f'allOrderedLFP_ignore4secs_{monkey}_outliers_removed.npy',
            f'allOrderedLFP_ignore4secs_{monkey}.npy',
        ]

    return [
        f'allOrderedLFP_ignore4secs_{freq_band}_{monkey}_outlierChunks_removed.npy',
    ]


def _mua_file_candidates(monkey: str, window: str) -> List[str]:
    if window == WINDOW_15S:
        return [
            f'allOrderedMUA_binsize_2_ignore4secs_MUA_{monkey}_outlierChunks_removed_15seconds.npy',
            f'allOrderedMUA_binsize_2_ignore4secs_{monkey}_outlierChunks_removed_15seconds.npy',
            f'allOrderedMUA_binsize_2_ignore4secs_{monkey}_15seconds.npy',
        ]
    return [
        f'allOrderedMUA_binsize_2_ignore4secs_{monkey}_outlierChunks_removed.npy',
        f'allOrderedMUA_binsize_2_ignore4secs_MUA_{monkey}_outlierChunks_removed.npy',
        f'allOrderedMUA_binsize_2_ignore4secs_{monkey}.npy',
    ]


def _loader_colors(monkey: str) -> List[Tuple[float, float, float]]:
    cols_monkey_L = [
        (1.0000, 0, 0), (1.0000, 0.3750, 0), (1.0000, 0.7500, 0), (0.8750, 1.0000, 0),
        (0.5000, 1.0000, 0), (0.1250, 1.0000, 0), (0, 1.0000, 0.2500), (0.5451, 0.2706, 0.0745),
        (0, 1.0000, 1.0000), (0, 0.6250, 1.0000), (0, 0.2500, 1.0000), (0.1250, 0, 1.0000),
        (0.5000, 0, 1.0000), (0.8750, 0, 1.0000), (1.0000, 0, 0.7500), (1.0000, 0, 0.3750),
    ]
    cols_monkey_A = [
        (1.0000, 0, 0), (1.0000, 0.3750, 0), (0.8750, 1.0000, 0), (0.5000, 1.0000, 0),
        (1.0000, 0.7500, 0), (0.1250, 1.0000, 0), (0, 1.0000, 0.2500), (0.5451, 0.2706, 0.0745),
        (0, 1.0000, 1.0000), (0, 0.6250, 1.0000), (0.1250, 0, 1.0000), (0.5000, 0, 1.0000),
        (0.8750, 0, 1.0000), (1.0000, 0, 0.7500), (1.0000, 0, 0.3750), (0, 0, 0),
    ]
    return cols_monkey_L if monkey == 'monkey_L' else cols_monkey_A


def _build_return_tuple(data: Dict[str, np.ndarray]):
    array_idx = np.array(data['selectedChannelsIndices'], dtype='object')
    return (
        data['LFP'],
        data['MUABINNED'],
        data['anchorLFP'],
        data['anchorUtah'],
        data['anchorUtahDistances'],
        data['colors'],
        data['arrayNumberV1'],
        array_idx,
        data['colorsWAlpha'],
        data['channelAreaMap'],
        data['arrayNums'],
        data['areas'],
        data['channelNums'],
        data['rfs'],
        data['rfs_distances'],
        data['utah'],
        data['utah_distances'],
        data['badElectrodesDict'],
        data['subjectConditionList'],
        data['nonvalidEList'],
        data['idxNonvalid'],
        data['nonvalidE'],
        data['UTAHMAX'],
        data['selectedChannelsIndices'],
    )


def _load_data_core(monkey: str, STANDARIZE_LFP: bool, EXCLUDE_V4: bool, base_path: str, where_utah_path: str,
                    channel_area_mapping_path: str, delete_elecs_path: str, freq_band: str, load_MUA: bool,
                    LFP_float16: bool, window: str):
    #print(f"DEBUG: _load_data_core called with:")
    print(f"  monkey: {monkey}")
    print(f"  base_path: {base_path}")
    print(f"  where_utah_path: {where_utah_path}")
    print(f"  channel_area_mapping_path: {channel_area_mapping_path}")
    print(f"  delete_elecs_path: {delete_elecs_path}")
    print(f"  freq_band: {freq_band}")
    print(f"  window: {window}")

    condition_root = _resolve_condition_root(base_path, monkey)
    print(f"[loader:{window}] Using condition root: {condition_root}")

    cols = _loader_colors(monkey)
    all_colors = np.vstack([np.tile(c, (64, 1)) for c in cols])

    #print(f"DEBUG: Loading channel area mapping from: {channel_area_mapping_path}")
    channel_area_map = scipy.io.loadmat(channel_area_mapping_path)
    array_nums = channel_area_map['arrayNums']
    areas = channel_area_map['areas']
    channel_nums = channel_area_map['channelNums']
    print(f"DEBUG: Channel area map loaded successfully")

    rfs_dir = os.path.join(condition_root, 'RFS')
    #print(f"DEBUG: Loading RF instances from: {rfs_dir}")
    rf_list = load_rf_instances(rfs_dir)
    #print(f"DEBUG: RF list length: {len(rf_list)}")
    if len(rf_list) == 0:
        raise RuntimeError(f"No RF instances found in {rfs_dir}")
    rf_arrs, _ = order_RF_instances(rf_list, array_nums, channel_nums, V1ONLY=False)
    #print(f"DEBUG: RF arrays length: {len(rf_arrs)}")
    rfs = np.vstack(rf_arrs)  # type: ignore[arg-type]
    #print(f"DEBUG: RFS shape after vstack: {rfs.shape}")

    mat_name = 'allPixelIDs_monkey_L.mat' if monkey == 'monkey_L' else 'allPixelIDs_monkey_A.mat'
    utah_path = os.path.join(where_utah_path, mat_name)
    #print(f"DEBUG: Loading Utah data from: {utah_path}")
    utah = load_utah_XING(where_utah_path, mat_name, all_colors, PLOT=False)
    utah[:, 1] += np.abs(utah[:, 1].min())
    utah_max = utah.max()
    utah /= utah_max
    #print(f"DEBUG: Utah data loaded, shape: {utah.shape}")

    TOLOAD = 'monkey_LClosedEyes' if monkey == 'monkey_L' else 'monkey_AClosedEyes'
    delete_file_path = os.path.join(delete_elecs_path, 'subjectCondition_NonvalidElectrodesList.pkl')
    #print(f"DEBUG: Loading bad electrodes from: {delete_file_path}")
    with open(delete_file_path, 'rb') as f:
        bad_electrodes_dict = pickle.load(f)
    subject_condition_list = np.array(bad_electrodes_dict['subject'])
    nonvalid_e_list = bad_electrodes_dict['nonvalidElectrodesList']
    idx_nonvalid = int(np.where(subject_condition_list == TOLOAD)[0][0])
    nonvalid_e = np.asarray(nonvalid_e_list[idx_nonvalid]).astype(int)
    #print(f"DEBUG: Bad electrodes loaded successfully")

    freq_normalized = _normalize_frequency(freq_band)
    lfp_dir = os.path.join(condition_root, 'LFP')
    lfp_path, lfp_name = _resolve_file(
        lfp_dir,
        _lfp_file_candidates(monkey, freq_normalized, window),
        f"{freq_normalized} LFP ({window})"
    )
    LFP = np.load(lfp_path, allow_pickle=True)
    if LFP_float16:
        LFP = LFP.astype('float16')
    print(f"[loader:{window}] Loaded LFP: {lfp_name} -> {LFP.shape}")

    MUABINNED = None
    mua_path = None
    if load_MUA:
        mua_dir = os.path.join(condition_root, 'MUA')
        mua_path, mua_name = _resolve_file(
            mua_dir,
            _mua_file_candidates(monkey, window),
            f"MUA ({window})"
        )
        MUABINNED = np.load(mua_path, allow_pickle=True)
        print(f"[loader:{window}] Loaded MUA: {mua_name} -> {MUABINNED.shape}")

    num_anchors_per_array = 2
    dummy, raw_arrays, anchor_indices, cols_ret, colors_new, array_number_v1, color_single_array, \
        relative_anchor_idx_list, selected_channels_indices, original_idx = create_array_index_new(
            np.random.rand(1024, 10), cols, num_anchors_per_array, nonvalid_e
        )
    del dummy, raw_arrays

    anchor_indices = np.asarray(anchor_indices, dtype=int)
    anchor_lfp = LFP[anchor_indices]
    anchor_utah = utah[anchor_indices]
    anchor_utah_distances = euclidean_distances(anchor_utah)

    # Remove nonvalid electrodes from utah and rfs
    utah_clean = np.delete(utah, nonvalid_e, axis=0)
    rfs_clean = np.delete(rfs, nonvalid_e, axis=0)
    
    # Check if LFP already has nonvalid electrodes removed
    # The file may already be cleaned (696 channels) or full (1024 channels)
    if LFP.shape[0] == utah.shape[0]:
        # LFP has same size as original utah (1024), need to clean it
        LFP_clean = np.delete(LFP, nonvalid_e, axis=0)
        print(f"[loader:{window}] Removed {len(nonvalid_e)} nonvalid electrodes from LFP: {LFP.shape[0]} -> {LFP_clean.shape[0]}")
    elif LFP.shape[0] == utah_clean.shape[0]:
        # LFP already cleaned, same size as utah_clean
        LFP_clean = LFP
        print(f"[loader:{window}] LFP already has nonvalid electrodes removed ({LFP.shape[0]} channels)")
    else:
        raise RuntimeError(
            f"LFP channel count mismatch: LFP has {LFP.shape[0]} channels, "
            f"but expected either {utah.shape[0]} (full) or {utah_clean.shape[0]} (cleaned)"
        )
    
    rfs_distances = euclidean_distances(rfs_clean)
    utah_distances = euclidean_distances(utah_clean)
    if utah_distances.max() > 0:
        utah_distances /= utah_distances.max()

    colors_arr = colors_new
    colors_w_alpha = []
    for idxs in selected_channels_indices:
        alphas = np.expand_dims(np.linspace(0.3, 1, len(idxs)), -1)
        alphas = np.power(alphas, 2)
        colors_w_alpha.append(np.hstack((colors_arr[idxs], alphas)))
    colors_w_alpha = np.vstack(colors_w_alpha)

    return {
        'LFP': LFP_clean,
        'MUABINNED': MUABINNED,
        'anchorLFP': anchor_lfp,
        'anchorUtah': anchor_utah,
        'anchorUtahDistances': anchor_utah_distances,
        'colors': colors_arr,
        'arrayNumberV1': array_number_v1,
        'colorsWAlpha': colors_w_alpha,
        'channelAreaMap': channel_area_map,
        'arrayNums': array_nums,
        'areas': areas,
        'channelNums': channel_nums,
        'rfs': rfs_clean,
        'rfs_distances': rfs_distances,
        'utah': utah_clean,
        'utah_distances': utah_distances,
        'badElectrodesDict': bad_electrodes_dict,
        'subjectConditionList': subject_condition_list,
        'nonvalidEList': nonvalid_e_list,
        'idxNonvalid': idx_nonvalid,
        'nonvalidE': nonvalid_e,
        'UTAHMAX': utah_max,
        'selectedChannelsIndices': selected_channels_indices,
        'lfp_path': lfp_path,
        'mua_path': mua_path,
    }


def load_monkey_data_band(monkey: str, STANDARIZE_LFP: bool, EXCLUDE_V4: bool, base_path: str, where_utah_path: str,
                          channel_area_mapping_path: str, delete_elecs_path: str, freq_band: str = "LFP",
                          load_MUA: bool = False, LFP_float16: bool = False):
    """Load full-duration band-specific data for a given monkey."""
    data = _load_data_core(monkey, STANDARIZE_LFP, EXCLUDE_V4, base_path, where_utah_path,
                           channel_area_mapping_path, delete_elecs_path, freq_band, load_MUA,
                           LFP_float16, WINDOW_FULL)
    if data['lfp_path'].endswith('_15seconds.npy'):
        raise RuntimeError('Requested full window data but only a 15-second file was located.')
    return _build_return_tuple(data)


def load_monkey_data_band_15seconds(monkey: str, STANDARIZE_LFP: bool, EXCLUDE_V4: bool, base_path: str,
                                    where_utah_path: str, channel_area_mapping_path: str, delete_elecs_path: str,
                                    freq_band: str = "LFP", load_MUA: bool = False, LFP_float16: bool = False):
    """Load the trimmed 15-second data chunk for a given monkey."""
    data = _load_data_core(monkey, STANDARIZE_LFP, EXCLUDE_V4, base_path, where_utah_path,
                           channel_area_mapping_path, delete_elecs_path, freq_band, load_MUA,
                           LFP_float16, WINDOW_15S)
    return _build_return_tuple(data)


__all__ = [
    "load_monkey_data_band",
    "load_monkey_data_band_15seconds",
]
