"""Distance geometry utilities for protein structures."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


def pairwise_distances(
    coords: ArrayLike,
    mask: ArrayLike | None = None,
) -> FloatArray:
    """Compute pairwise Euclidean distances for ``[num_res, 3]`` coordinates.

    Invalid residues indicated by ``mask=False`` produce zero distances for
    their full row and column.
    """

    validated_coords = _validate_coords(coords)
    validated_mask = _validate_mask(mask, len(validated_coords))

    squared_distances = pairwise_squared_distances(validated_coords, validated_mask)
    distances = np.sqrt(squared_distances)
    np.fill_diagonal(distances, 0.0)
    return distances


def pairwise_squared_distances(
    coords: ArrayLike,
    mask: ArrayLike | None = None,
) -> FloatArray:
    """Compute pairwise squared Euclidean distances for ``[num_res, 3]`` coordinates."""

    validated_coords = _validate_coords(coords)
    validated_mask = _validate_mask(mask, len(validated_coords))

    deltas = validated_coords[:, None, :] - validated_coords[None, :, :]
    # First shape: (L, 1, 3), second shape: (1, L, 3)
    # 
    # The broadcasting in numpy:
    # 
    # We use [L, 3] to denote the shape of coords, it has N rows and 3 columns.
    # L represents the number of residues, and 3 represents the dimension of
    # coordinates in 3-dimensional points.
    # 
    # Broadcasting in numpy can make ndarray of different shapes calculate with
    # each other. For example:
    # 
    # A = np.array([
    #     [1, 2, 3],
    #     [4, 5, 6],
    #     [7, 8, 9],
    # ])
    # 
    # v = np.array([10, 20, 30])
    # 
    # The result of A + v:
    # 
    # np.array([
    #     [11, 22, 33],
    #     [14, 25, 36],
    #     [17, 28, 39],
    # ])
    # 
    # In numpy, `None` equals to `np.newaxis`, which can add am extra dimension
    # in given position. For example:
    # 
    # x = np.array([1, 2, 3])
    # print(x.shape)      # (3,)
    # x1 = x[None, :]
    # print(x1.shape)     # (1, 3)
    # x2 = x[:, None]
    # print(x2.shape)     # (3, 1)
    # x3 = x[None, :, None]
    # print(x3.shape)     # (1, 3, 1)

    # deltas[i, j] = coords[i] - coords[j]
    # deltas * deltas computes every square of components
    # deltas * deltas [i, j, 0] = deltas[i, j, 0] ^ 2

    squared_distances = np.sum(deltas * deltas, axis=-1)
    # axis=-1 represents the last dimension of deltas * deltas, which is the
    # result of x^2, y^2 and z^2. Using `np.sum` function will collect them
    # together to calculate x^2 + y^2 + z^2
    squared_distances = np.maximum(squared_distances, 0.0)
    # squared_distances is a matrix of squared distance between two residues
    squared_distances *= pair_mask(validated_mask)
    # when mask is false, the data in `squared_distances` will be hidden.
    np.fill_diagonal(squared_distances, 0.0)
    # force the element at diagonal to be 0
    return squared_distances


def pair_mask(mask: ArrayLike) -> BoolArray:
    """Expand a residue mask ``[num_res]`` to a pair mask ``[num_res, num_res]``."""

    mask_array = np.asarray(mask)
    if mask_array.ndim != 1:
        raise ValueError(f"mask must have shape [num_res], got {mask_array.shape}")

    validated_mask = _validate_mask(mask_array, mask_array.shape[0])
    return np.logical_and(validated_mask[:, None], validated_mask[None, :])


def euclidean_distance(coord_a: ArrayLike, coord_b: ArrayLike) -> float:
    """Compute the Euclidean distance between two 3D coordinates."""

    coord_a_array = _validate_single_coord(coord_a, "coord_a")
    coord_b_array = _validate_single_coord(coord_b, "coord_b")
    return float(np.linalg.norm(coord_a_array - coord_b_array))


def _validate_coords(coords: ArrayLike) -> FloatArray:
    coords_array = np.asarray(coords, dtype=np.float64)

    if coords_array.ndim != 2 or coords_array.shape[1] != 3:
        raise ValueError(f"coords must have shape [num_res, 3], got {coords_array.shape}")

    return coords_array


def _validate_single_coord(coord: ArrayLike, name: str) -> FloatArray:
    coord_array = np.asarray(coord, dtype=np.float64)
    # function np.asarray try to not copy the data but share the same memory.
    # If our input is NDArray, then we should use np.asarray to reuse its data
    # in the memory

    if coord_array.shape != (3,):
        raise ValueError(f"{name} must have shape [3], got {coord_array.shape}")

    return coord_array


def _validate_mask(mask: ArrayLike | None, length: int) -> BoolArray:
    if mask is None:
        return np.ones((length,), dtype=np.bool_)

    mask_array = np.asarray(mask, dtype=np.bool_)

    if mask_array.shape != (length,):
        raise ValueError(f"mask must have shape [{length}], got {mask_array.shape}")

    return mask_array
