from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

PACKAGE_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PACKAGE_ROOT.parent
DEMO_ROOT = SRC_ROOT.parent
REPO_ROOT = DEMO_ROOT
DATA_DIR = DEMO_ROOT / "data"
RESULTS_DIR = DEMO_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

DEFAULT_FREQ_BAND = "LFP"
DEFAULT_RANDOM_SEED: int | None = None

# Frequency bands for UMAP concatenation
FREQ_BANDS = ["LFP", "low", "alpha", "beta", "gamma", "highGamma"]

FULL_DATA_DIR = REPO_ROOT / "data"
FULL_COORDS_DIR = FULL_DATA_DIR / "coordinates_of_electrodes_on_cortex_using_photos_of_arrays"
FULL_CHANNEL_MAP_DIR = FULL_DATA_DIR / "channel_area_mapping"
FULL_DELETED_DIR = FULL_DATA_DIR / "deletedElectrodesDictionary"
FULL_CHANNEL_MAP_PATH = FULL_CHANNEL_MAP_DIR / "channel_area_mapping.mat"

_ALIAS_ORDER: Tuple[str, ...] = ("monkey_A", "monkey_L")


def _discover_monkey_mapping() -> Dict[str, str]:
    """Map demo aliases to real names, falling back to identity using bundled NPZs.

    End users only need the demo NPZ bundles. If the full dataset exists, we
    could try to resolve true names, but to keep the demo robust, we simply use
    identity mapping by default.
    """
    try:
        available = {p.stem.split("_")[1] for p in DATA_DIR.glob("demo_*_lfp_*.npz")}
        # Normalize to canonical alias case keys
        mapping: Dict[str, str] = {}
        for alias in _ALIAS_ORDER:
            if alias.lower().split("_")[-1] in available or alias in available:
                mapping[alias] = alias
        if mapping:
            return mapping
    except Exception:
        pass
    # Fallback to identity mapping for both demo aliases
    return {alias: alias for alias in _ALIAS_ORDER}


_ALIAS_TO_REAL = _discover_monkey_mapping()
MONKEY_CHOICES: Tuple[str, ...] = _ALIAS_ORDER
MONKEY_ALIAS_TO_REAL: Dict[str, str] = {
    **_ALIAS_TO_REAL,
    **{alias.lower(): real for alias, real in _ALIAS_TO_REAL.items()},
}


def canonical_alias(alias: str) -> str:
    if alias in _ALIAS_TO_REAL:
        return alias
    lowered = alias.lower()
    for candidate in _ALIAS_ORDER:
        if candidate.lower() == lowered:
            return candidate
    raise KeyError(f"Unknown monkey alias '{alias}'. Valid options: {', '.join(_ALIAS_ORDER)}")


def resolve_monkey_alias(alias: str) -> str:
    canonical = canonical_alias(alias)
    return MONKEY_ALIAS_TO_REAL[canonical]


DEFAULT_MONKEY = _ALIAS_ORDER[-1]

