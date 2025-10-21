from __future__ import annotations

from pathlib import Path
import sys

import pytest

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

pytest.importorskip("umap")

from rf_demo.config import DATA_DIR, DEFAULT_MONKEY  # type: ignore[import]
from rf_demo.pipeline import run_demo_pipeline  # type: ignore[import]


@pytest.mark.skipif(
    not any(DATA_DIR.glob("demo_*_lfp_15s.npz")),
    reason="Demo dataset not generated yet",
)
def test_pipeline_runs():
    output = run_demo_pipeline()
    assert output.bundle.monkey == DEFAULT_MONKEY
    assert output.mds.metrics["cortical_corr"] > 0
    assert output.umap_best.metrics["cortical_corr"] > 0
    assert output.umap_best.surrogate_metrics is not None
    assert {"iedc", "rmse"}.issubset(output.umap_best.surrogate_metrics.keys())
