from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DEMO_SRC = REPO_ROOT / "demo" / "src"
if str(DEMO_SRC) not in sys.path:
    sys.path.insert(0, str(DEMO_SRC))

pytest.importorskip("umap")

from rf_demo.config import DATA_DIR  # type: ignore[import]
from rf_demo.pipeline import fast_umap_grid  # type: ignore[import]

SCRIPT_PATH = REPO_ROOT / "neumap_demo" / "neumap_demo.py"
DATA_AVAILABLE = any(DATA_DIR.glob("demo_*_lfp_15s.npz"))a


def _pruned_grid(num_electrodes: int, max_combos: int) -> List[Dict[str, Any]]:
    full_grid = fast_umap_grid(num_electrodes)
    pruned: List[Dict[str, Any]] = []
    for params in full_grid:
        if len(pruned) >= max_combos:
            break
        if params["n_neighbors"] >= num_electrodes:
            continue
        pruned.append(params)
    return pruned


@pytest.mark.skipif(not DATA_AVAILABLE, reason="Demo dataset not generated yet")
def test_neumap_demo_script_runs(tmp_path: Path) -> None:
    pythonpath_entries = [str(REPO_ROOT), str(DEMO_SRC)]
    existing = os.environ.get("PYTHONPATH")
    if existing:
        pythonpath_entries.append(existing)
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(pythonpath_entries)}
    result = subprocess.run(
        [sys.executable, "-B", str(SCRIPT_PATH)],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    assert "Best candidate" in result.stdout
    assert "Top parameter sets" in result.stdout


def test_pruned_grid_nonempty() -> None:
    combos = _pruned_grid(num_electrodes=256, max_combos=5)
    assert 0 < len(combos) <= 5
    for params in combos:
        assert params["n_neighbors"] < 128
        assert {"n_neighbors", "min_dist"} <= params.keys()
