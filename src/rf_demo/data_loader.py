from __future__ import annotations

import io
from contextlib import redirect_stdout
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .config import (
    DATA_DIR,
    DEFAULT_FREQ_BAND,
    DEFAULT_MONKEY,
    FREQ_BANDS,
    FULL_CHANNEL_MAP_DIR,
    FULL_CHANNEL_MAP_PATH,
    FULL_COORDS_DIR,
    FULL_DATA_DIR,
    FULL_DELETED_DIR,
    canonical_alias,
    resolve_monkey_alias,
)

WindowType = Literal["15s", "full"]

@dataclass(frozen=True)
class DemoBundle:
    monkey: str
    freq_band: str
    window: WindowType
    lfp: np.ndarray
    utah_mm: np.ndarray
    utah_distances: np.ndarray
    rfs_deg: np.ndarray
    rfs_distances: np.ndarray
    monkey_actual: str
    colors: np.ndarray


_DEMO_TEMPLATE = "demo_{monkey}_{freq_band}_{window}.npz"


def _bundle_path(monkey: str, freq_band: str, window: WindowType) -> Path:
    canonical = canonical_alias(monkey)
    filename = _DEMO_TEMPLATE.format(monkey=canonical.lower(), freq_band=freq_band.lower(), window=window)
    return DATA_DIR / filename


def load_demo_bundle(monkey: str = DEFAULT_MONKEY, freq_band: str = DEFAULT_FREQ_BAND,
                     window: WindowType = "15s") -> DemoBundle:
    canonical = canonical_alias(monkey)
    path = _bundle_path(canonical, freq_band, window)
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path.name}. Run `python demo/scripts/extract_demo_data.py` first."
        )
    payload = np.load(path)
    monkey_actual = resolve_monkey_alias(canonical)
    return DemoBundle(
        monkey=canonical,
        freq_band=freq_band,
        window=window,
        lfp=payload["lfp"],
        utah_mm=payload["utah_mm"],
        utah_distances=payload["utah_distances"],
        rfs_deg=payload["rfs_deg"],
        rfs_distances=payload["rfs_distances"],
        monkey_actual=monkey_actual,
        colors=payload["colors"],
    )


def load_concatenated_lfp(monkey: str = DEFAULT_MONKEY, window: WindowType = "15s") -> np.ndarray:
    """Load and concatenate LFP data from all frequency bands along time dimension for UMAP input."""
    canonical = canonical_alias(monkey)
    lfp_arrays = []
    
    for freq_band in FREQ_BANDS:
        path = _bundle_path(canonical, freq_band, window)
        if not path.exists():
            print(f"Warning: {path.name} not found, skipping {freq_band}")
            continue
        payload = np.load(path)
        lfp_arrays.append(payload["lfp"])
    
    if not lfp_arrays:
        raise FileNotFoundError(
            f"Could not find any demo bundles for {canonical}. Run `python demo/scripts/extract_demo_data.py` for all frequency bands first."
        )
    
    # Concatenate along the time dimension (axis 1)
    concatenated = np.concatenate(lfp_arrays, axis=1)
    return concatenated.astype(np.float32)


def extract_demo_bundle(monkey: str = DEFAULT_MONKEY, freq_band: str = DEFAULT_FREQ_BAND,
                        window: WindowType = "15s", overwrite: bool = True) -> Path:
    canonical = canonical_alias(monkey)
    monkey_actual = resolve_monkey_alias(canonical)
    # Developer-only path: these imports require the full project code available
    # in a sibling 'code/' directory. End users running the demo won't call this.
    from utils import get_monkey_pixPerMM_utahMax  # type: ignore
    from utils_extension import load_monkey_data_band, load_monkey_data_band_15seconds  # type: ignore

    loader_fn = load_monkey_data_band_15seconds if window == "15s" else load_monkey_data_band

    print(f"[demo] extracting {canonical} ({freq_band}, window={window})")

    channel_map_path = FULL_CHANNEL_MAP_DIR / f"channelArea_{monkey_actual}.mat"
    if not channel_map_path.exists():
        channel_map_path = FULL_CHANNEL_MAP_PATH

    with io.StringIO() as buffer, redirect_stdout(buffer):
        (lfp,
         _mua,
         _anchor_lfp,
         _anchor_utah,
         _anchor_utah_distances,
         _colors,
         _array_number_v1,
         selected_channels_indices,
         _colors_w_alpha,
         _channel_area_map,
         _array_nums,
         _areas,
         _channel_nums,
         rfs,
         rfs_distances,
         utah,
         utah_distances,
         _bad_electrodes_dict,
         _subject_condition_list,
         _nonvalid_e_list,
         _idx_nonvalid,
         _nonvalid_e,
         utah_max,
         _selected_channels_indices_raw) = loader_fn(
             monkey=monkey_actual,
             STANDARIZE_LFP=False,
             EXCLUDE_V4=True,
             base_path=str(FULL_DATA_DIR),
             where_utah_path=str(FULL_COORDS_DIR),
             channel_area_mapping_path=str(channel_map_path),
             delete_elecs_path=str(FULL_DELETED_DIR),
             freq_band=freq_band,
             load_MUA=False,
             LFP_float16=True,
         )

    chosen = np.array([idx for group in selected_channels_indices for idx in group])
    lfp_demo = lfp[chosen]

    utah_demo = utah[chosen]
    rfs_demo = rfs[chosen]
    colors_demo = np.clip(np.asarray(_colors)[chosen], 0.0, 1.0)

    pixels_per_mm, utah_max_scale = get_monkey_pixPerMM_utahMax(monkey_actual)
    if pixels_per_mm is None or utah_max_scale is None:
        raise RuntimeError(f"Missing scaling information for {canonical}")

    utah_mm = utah_demo * utah_max_scale / pixels_per_mm
    PIXELS_PER_DEG = 25.78
    rfs_deg = rfs_demo / PIXELS_PER_DEG

    utah_dist_demo = utah_distances[np.ix_(chosen, chosen)] * (utah_max_scale / pixels_per_mm)
    rfs_dist_demo = rfs_distances[np.ix_(chosen, chosen)] / PIXELS_PER_DEG

    payload = {
        "lfp": lfp_demo.astype("float32"),
        "utah_mm": utah_mm.astype("float32"),
        "utah_distances": utah_dist_demo.astype("float32"),
        "rfs_deg": rfs_deg.astype("float32"),
        "rfs_distances": rfs_dist_demo.astype("float32"),
        "colors": colors_demo.astype("float32"),
    }

    path = _bundle_path(canonical, freq_band, window)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists and overwrite=False")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(path), **payload)  # type: ignore[arg-type]
    return path
