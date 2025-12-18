from __future__ import annotations

import numpy as np
import pytest

import neumap_demo_v2 as demo_v2


def test_resolve_monkey_name_variants() -> None:
    alias, actual = demo_v2.resolve_monkey_name("monkey_L")
    assert (alias, actual) == ("monkey_L", "monkey_L")

    alias, actual = demo_v2.resolve_monkey_name("monkey_a")
    assert (alias, actual) == ("monkey_A", "monkey_A")

    alias, actual = demo_v2.resolve_monkey_name("monkey_A")
    assert (alias, actual) == ("monkey_A", "monkey_A")


@pytest.mark.parametrize("num_electrodes", [4, 16, 256])
def test_fast_umap_grid_small_counts(num_electrodes: int) -> None:
    grid = demo_v2.fast_umap_grid(num_electrodes)
    assert grid, "Grid should never be empty"
    for params in grid:
        assert 2 <= int(params["n_neighbors"]) < max(num_electrodes, 3)
        assert params["min_dist"] in {0.1, 0.5, 0.9}


def test_prune_umap_grid_respects_limits() -> None:
    full_grid = [
        {"n_neighbors": 50.0, "min_dist": 0.1},
        {"n_neighbors": 80.0, "min_dist": 0.5},
        {"n_neighbors": 120.0, "min_dist": 0.9},
    ]
    pruned = demo_v2._prune_umap_grid(full_grid, max_combos=2, num_electrodes=100)
    assert len(pruned) == 2
    assert all(int(p["n_neighbors"]) < 100 for p in pruned)


def test_correlation_distance_matrix_properties() -> None:
    rng = np.random.default_rng(0)
    data = rng.normal(size=(8, 32))
    dist = demo_v2.correlation_distance_matrix(data)

    assert dist.shape == (8, 8)
    assert np.allclose(np.diag(dist), 0.0)
    assert np.all(dist >= 0.0)
    assert np.all(dist <= 2.0)
