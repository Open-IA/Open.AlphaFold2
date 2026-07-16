"""Plotting helpers for protein distance geometry."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib
import numpy as np
from numpy.typing import ArrayLike

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


def plot_distance_matrix(
    distances: ArrayLike,
    output_path: str | Path,
    *,
    title: str | None = None,
    residue_ids: Sequence[str] | None = None,
    dpi: int = 180,
) -> Path:
    """Save a pairwise distance heatmap image."""

    distance_array = np.asarray(distances, dtype=np.float64)
    if distance_array.ndim != 2 or distance_array.shape[0] != distance_array.shape[1]:
        raise ValueError(
            f"distances must have shape [num_res, num_res], got {distance_array.shape}"
        )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=_figure_size(distance_array.shape[0]), constrained_layout=True)
    image = ax.imshow(distance_array, cmap="viridis", origin="lower")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("C-alpha distance (A)")

    ax.set_xlabel("Residue index")
    ax.set_ylabel("Residue index")
    if title:
        ax.set_title(title)

    _set_residue_ticks(ax, residue_ids, distance_array.shape[0])
    fig.savefig(output, dpi=dpi)
    plt.close(fig)
    return output


def _figure_size(num_residues: int) -> tuple[float, float]:
    side = min(max(num_residues / 28.0, 4.0), 12.0)
    return side, side


def _set_residue_ticks(ax: plt.Axes, residue_ids: Sequence[str] | None, num_residues: int) -> None:
    if num_residues <= 1:
        ticks = [0]
    else:
        max_ticks = 8
        step = max(1, (num_residues - 1) // (max_ticks - 1))
        ticks = list(range(0, num_residues, step))
        if ticks[-1] != num_residues - 1:
            ticks.append(num_residues - 1)

    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    if residue_ids is None:
        labels = [str(index + 1) for index in ticks]
    else:
        labels = [residue_ids[index] for index in ticks]

    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
