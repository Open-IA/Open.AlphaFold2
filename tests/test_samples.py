from pathlib import Path

import numpy as np

from open_alphafold2.constants import RESTYPE_TO_INDEX
from open_alphafold2.data import (
    DEFAULT_DISTOGRAM_BIN_EDGES,
    load_ca_sample,
    make_ca_sample_from_structure,
    make_distogram_bins,
    save_ca_sample,
)


def test_make_ca_sample_from_structure_builds_distance_targets(tmp_path: Path) -> None:
    pdb_path = _write_tiny_pdb(tmp_path)

    sample = make_ca_sample_from_structure(pdb_path, chain_id="A")

    assert sample.structure_id == "tiny"
    assert sample.chain_id == "A"
    assert sample.sequence == "AG"
    assert sample.residue_ids == ["A:ALA1", "A:GLY2"]
    np.testing.assert_array_equal(
        sample.sequence_encoded,
        np.array([RESTYPE_TO_INDEX["A"], RESTYPE_TO_INDEX["G"]], dtype=np.int64),
    )
    expected_five_angstrom_bin = np.digitize(5.0, DEFAULT_DISTOGRAM_BIN_EDGES).item()
    np.testing.assert_allclose(sample.ca_coords, np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]]))
    np.testing.assert_allclose(sample.distances, np.array([[0.0, 5.0], [5.0, 0.0]]))
    np.testing.assert_array_equal(
        sample.distogram_bins,
        np.array([[0, expected_five_angstrom_bin], [expected_five_angstrom_bin, 0]]),
    )
    np.testing.assert_allclose(sample.distogram_bin_edges, DEFAULT_DISTOGRAM_BIN_EDGES)
    np.testing.assert_array_equal(sample.mask, np.array([True, True]))


def test_make_ca_sample_from_structure_masks_missing_ca_targets(tmp_path: Path) -> None:
    pdb_path = tmp_path / "missing_ca.pdb"
    pdb_path.write_text(
        "\n".join(
            [
                "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C",
                "ATOM      2  N   SER A   2       1.000   0.000   0.000  1.00 20.00           N",
                "ATOM      3  CA  GLY A   3       0.000   3.000   4.000  1.00 20.00           C",
                "TER",
                "END",
            ]
        )
        + "\n"
    )

    sample = make_ca_sample_from_structure(pdb_path, chain_id="A")

    assert sample.sequence == "ASG"
    np.testing.assert_array_equal(sample.mask, np.array([True, False, True]))
    np.testing.assert_allclose(
        sample.distances,
        np.array(
            [
                [0.0, 0.0, 5.0],
                [0.0, 0.0, 0.0],
                [5.0, 0.0, 0.0],
            ]
        ),
    )
    np.testing.assert_array_equal(
        sample.distogram_bins < 0,
        np.array(
            [
                [False, True, False],
                [True, True, True],
                [False, True, False],
            ]
        ),
    )


def test_save_and_load_ca_sample_round_trips_npz(tmp_path: Path) -> None:
    pdb_path = _write_tiny_pdb(tmp_path)
    sample = make_ca_sample_from_structure(pdb_path, chain_id="A")
    output_path = tmp_path / "tiny_A.npz"

    saved_path = save_ca_sample(sample, output_path)
    loaded = load_ca_sample(saved_path)

    assert saved_path == output_path
    assert loaded.structure_id == sample.structure_id
    assert loaded.chain_id == sample.chain_id
    assert loaded.sequence == sample.sequence
    assert loaded.residue_ids == sample.residue_ids
    assert loaded.source_path == sample.source_path
    np.testing.assert_array_equal(loaded.sequence_encoded, sample.sequence_encoded)
    np.testing.assert_allclose(loaded.ca_coords, sample.ca_coords)
    np.testing.assert_array_equal(loaded.mask, sample.mask)
    np.testing.assert_allclose(loaded.distances, sample.distances)
    np.testing.assert_array_equal(loaded.distogram_bins, sample.distogram_bins)
    np.testing.assert_allclose(loaded.distogram_bin_edges, sample.distogram_bin_edges)


def test_make_distogram_bins_masks_invalid_pairs() -> None:
    distances = np.array([[0.0, 5.0], [5.0, 0.0]])
    mask = np.array([True, False])

    bins = make_distogram_bins(distances, mask)

    np.testing.assert_array_equal(bins, np.array([[0, -1], [-1, -1]]))


def _write_tiny_pdb(tmp_path: Path) -> Path:
    pdb_path = tmp_path / "tiny.pdb"
    pdb_path.write_text(
        "\n".join(
            [
                "ATOM      1  N   ALA A   1      -1.000   0.000   0.000  1.00 20.00           N",
                "ATOM      2  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C",
                "ATOM      3  C   ALA A   1       1.000   0.000   0.000  1.00 20.00           C",
                "ATOM      4  CA  GLY A   2       3.000   4.000   0.000  1.00 20.00           C",
                "TER",
                "END",
            ]
        )
        + "\n"
    )
    return pdb_path
