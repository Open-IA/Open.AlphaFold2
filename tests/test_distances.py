import numpy as np
import pytest

from open_alphafold2.geometry import (
    euclidean_distance,
    pair_mask,
    pairwise_distances,
    pairwise_squared_distances,
)


def test_pairwise_distances_computes_symmetric_matrix() -> None:
    coords = [
        (0.0, 0.0, 0.0),
        (3.0, 4.0, 0.0),
        (0.0, 0.0, 12.0),
    ]

    distances = pairwise_distances(np.array(coords))

    np.testing.assert_allclose(
        distances,
        np.array(
            [
                [0.0, 5.0, 12.0],
                [5.0, 0.0, 13.0],
                [12.0, 13.0, 0.0],
            ]
        ),
    )


def test_pairwise_distances_zeros_invalid_rows_and_columns() -> None:
    coords = [
        (0.0, 0.0, 0.0),
        (3.0, 4.0, 0.0),
        (0.0, 0.0, 12.0),
    ]

    distances = pairwise_distances(np.array(coords), mask=np.array([True, False, True]))

    np.testing.assert_allclose(
        distances,
        np.array(
            [
                [0.0, 0.0, 12.0],
                [0.0, 0.0, 0.0],
                [12.0, 0.0, 0.0],
            ]
        ),
    )


def test_pairwise_distances_keeps_duplicate_coordinates_at_zero() -> None:
    distances = pairwise_distances(np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]))

    np.testing.assert_allclose(distances, np.zeros((2, 2)))


def test_pairwise_squared_distances_computes_without_sqrt() -> None:
    distances = pairwise_squared_distances(np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]]))

    np.testing.assert_allclose(distances, np.array([[0.0, 25.0], [25.0, 0.0]]))


def test_pair_mask_expands_residue_mask_to_pair_mask() -> None:
    np.testing.assert_array_equal(
        pair_mask(np.array([True, False, True])),
        np.array(
            [
                [True, False, True],
                [False, False, False],
                [True, False, True],
            ]
        ),
    )


def test_euclidean_distance_computes_single_pair() -> None:
    assert euclidean_distance(np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 2.5])) == 2.5


def test_pairwise_distances_rejects_non_3d_coordinates() -> None:
    with pytest.raises(ValueError, match=r"coords must have shape \[num_res, 3\]"):
        pairwise_distances(np.array([[0.0, 0.0]]))


def test_pairwise_distances_rejects_wrong_mask_length() -> None:
    with pytest.raises(ValueError, match=r"mask must have shape \[2\]"):
        pairwise_distances(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]), mask=np.array([True]))


def test_pair_mask_rejects_non_vector_mask() -> None:
    with pytest.raises(ValueError, match=r"mask must have shape \[num_res\]"):
        pair_mask(np.array([[True, False]]))
