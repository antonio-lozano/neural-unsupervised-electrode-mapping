from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = SCRIPTS_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from rf_demo.config import RESULTS_DIR, MONKEY_CHOICES, canonical_alias  # type: ignore[import]
from rf_demo.pipeline import PipelineOutput, run_demo_pipeline  # type: ignore[import]
from rf_demo.visualization import (  # type: ignore[import]
    annotate_metrics,
    plot_cortex_alignment,
    plot_distance_heatmap,
    plot_visual_field_projection,
)


def _save_metrics(output: PipelineOutput, destination: Path) -> None:
    umap_candidates = [
        {"params": params, "metrics": metrics}
        for params, metrics in output.umap_candidates
    ]
    payload = {
        "monkey": output.bundle.monkey,
        "freq_band": output.bundle.freq_band,
        "window": output.bundle.window,
        "mds": {
            "metrics": output.mds.metrics,
            "params": output.mds.params,
        },
        "umap": {
            "metrics": output.umap_best.metrics,
            "params": output.umap_best.params,
            "surrogate_metrics": output.umap_best.surrogate_metrics,
        },
        "umap_candidates": umap_candidates,
    }
    destination.write_text(json.dumps(payload, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the RF visual field demo pipeline")
    parser.add_argument("--monkey", default=MONKEY_CHOICES[-1], choices=list(MONKEY_CHOICES), help="Demo subject")
    parser.add_argument("--window", default="15s", choices=["15s", "full"], help="Data window")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (omit for fastest run)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    monkey_alias = canonical_alias(args.monkey)
    print(f"Running demo for {monkey_alias}...")
    
    output = run_demo_pipeline(
        monkey=monkey_alias,
        window=args.window,
        random_state=args.seed,
    )
    print("Pipeline completed.")

    run_dir = RESULTS_DIR / f"{monkey_alias.lower()}_lfp"
    run_dir.mkdir(parents=True, exist_ok=True)
    print("Generating plots...")

    plot_distance_heatmap(output.distance_matrix, save_path=run_dir / "correlation_distance.png")
    plot_cortex_alignment(output, save_path=run_dir / "cortex_alignment.png")
    plot_visual_field_projection(output, save_path=run_dir / "visual_projection.png")

    _save_metrics(output, run_dir / "metrics.json")
    print("Plots and metrics saved.")

    print("Cortical metrics (MDS):")
    print(annotate_metrics(output.mds.metrics.items()))
    print("\nCortical metrics (UMAP tuned):")
    print(annotate_metrics(output.umap_best.metrics.items()))
    if output.umap_best.surrogate_metrics:
        print("\nUMAP surrogate metrics (MDS space):")
        print(annotate_metrics(output.umap_best.surrogate_metrics.items()))
    if "freq_band" in output.umap_best.params:
        print(f"\nWinning frequency band: {output.umap_best.params['freq_band']}")


if __name__ == "__main__":
    main()
