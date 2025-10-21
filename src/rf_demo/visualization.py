from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Circle
import numpy as np
import seaborn as sns

from .pipeline import PipelineOutput, project_to_visual_field

sns.set_style("white")


def _scatter(ax: Axes, points: np.ndarray, colors: np.ndarray, title: str, alpha: float = 0.9) -> None:
    ax.scatter(points[:, 0], points[:, 1], s=18, c=colors, alpha=alpha, edgecolor="k", linewidth=0.3)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(False)


def _add_visual_guides(ax: Axes, radius: float) -> None:
    ax.grid(False)
    ax.axhline(0.0, color="#bbbbbb", linewidth=0.8, linestyle="--")
    ax.axvline(0.0, color="#bbbbbb", linewidth=0.8, linestyle="--")
    for r in range(1, int(max(radius, 0)) + 1):
        circle = Circle((0.0, 0.0), float(r), edgecolor="#d0d0d0", facecolor="none", linewidth=0.8, linestyle=":")
        ax.add_patch(circle)


def plot_cortex_alignment(output: PipelineOutput, save_path: Path | None = None) -> Figure:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    utah = output.bundle.utah_mm
    colors = output.bundle.colors
    params = output.umap_best.params
    freq = params.get('freq_band', 'unknown')
    n_neighbors = params.get('n_neighbors', '?')
    min_dist = params.get('min_dist', '?')
    umap_title = f"UMAP aligned (freq: {freq}, n: {n_neighbors}, d: {min_dist})"
    
    # Compute shared axis limits for equal subplot sizes
    all_points = np.vstack([utah, output.mds.aligned_mm, output.umap_best.aligned_mm])
    x_min, x_max = np.min(all_points[:, 0]), np.max(all_points[:, 0])
    y_min, y_max = np.min(all_points[:, 1]), np.max(all_points[:, 1])
    # Add small padding
    x_range = x_max - x_min
    y_range = y_max - y_min
    padding = 0.05 * max(x_range, y_range)
    x_min -= padding
    x_max += padding
    y_min -= padding
    y_max += padding
    
    _scatter(axes[0], utah, colors, "Ground truth cortex")
    _scatter(axes[1], output.mds.aligned_mm, colors, "MDS aligned")
    _scatter(axes[2], output.umap_best.aligned_mm, colors, umap_title)
    
    for ax in axes:
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal', adjustable='box')
    
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_visual_field_projection(output: PipelineOutput, save_path: Path | None = None) -> Figure:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    utah_visual = project_to_visual_field(output.bundle.utah_mm, output.bundle.monkey)
    mds_visual = project_to_visual_field(output.mds.aligned_mm, output.bundle.monkey)
    umap_visual = project_to_visual_field(output.umap_best.aligned_mm, output.bundle.monkey)
    colors = output.bundle.colors
    all_points = np.vstack([utah_visual, mds_visual, umap_visual])
    max_radius = float(np.ceil(np.max(np.linalg.norm(all_points, axis=1)))) if all_points.size else 0.0
    for ax in axes:
        _add_visual_guides(ax, max_radius)
    # Only show lower right quadrant, but with 2 degrees extra up and left
    params = output.umap_best.params
    freq = params.get('freq_band', 'unknown')
    n_neighbors = params.get('n_neighbors', '?')
    min_dist = params.get('min_dist', '?')
    titles = ["Ground truth VF", "MDS VF", f"UMAP VF (freq: {freq}, n: {n_neighbors}, d: {min_dist})"]
    for idx, pts in enumerate([utah_visual, mds_visual, umap_visual]):
        _scatter(axes[idx], pts, colors, titles[idx])
        # Find min/max for lower right quadrant
        x_min = min(0.0, np.min(pts[:, 0]) - 2)
        y_min = min(0.0, np.min(pts[:, 1]) - 2)
        x_max = np.max(pts[:, 0]) + 2
        y_max = np.max(pts[:, 1]) + 2
        axes[idx].set_xlim(x_min, x_max)
        axes[idx].set_ylim(y_min, y_max)
        axes[idx].set_aspect('equal', adjustable='box')
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_distance_heatmap(distance_matrix: np.ndarray, save_path: Path | None = None) -> Figure:
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(distance_matrix, cmap="mako", ax=ax, square=True)
    ax.set_title("Correlation distance matrix")
    ax.set_xlabel("Channel")
    ax.set_ylabel("Channel")
    ax.grid(False)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def annotate_metrics(metrics: Iterable[Tuple[str, float]]) -> str:
    return "\n".join(f"{name}: {value:.3f}" for name, value in metrics)
