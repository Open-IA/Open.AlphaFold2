from pathlib import Path

import numpy as np
import pytest

from open_alphafold2.data import load_ca_coordinates, normalize_pdb_id, pdb_id_to_mmcif_url


def test_load_ca_coordinates_reads_pdb_chain(tmp_path: Path) -> None:
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

    structure = load_ca_coordinates(pdb_path, chain_id="A")

    assert structure.chain_id == "A"
    assert structure.sequence == "AG"
    assert structure.residue_ids == ["A:ALA1", "A:GLY2"]
    np.testing.assert_allclose(structure.coords, np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]]))
    np.testing.assert_array_equal(structure.mask, np.array([True, True]))


def test_load_ca_coordinates_preserves_missing_ca_by_default(tmp_path: Path) -> None:
    pdb_path = tmp_path / "missing_ca.pdb"
    pdb_path.write_text(
        "\n".join(
            [
                "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00 20.00           C",
                "ATOM      2  N   SER A   2       4.000   5.000   6.000  1.00 20.00           N",
                "ATOM      3  CA  GLY A   3       7.000   8.000   9.000  1.00 20.00           C",
                "TER",
                "END",
            ]
        )
        + "\n"
    )

    structure = load_ca_coordinates(pdb_path, chain_id="A")

    assert structure.sequence == "ASG"
    assert structure.residue_ids == ["A:ALA1", "A:SER2", "A:GLY3"]
    np.testing.assert_allclose(
        structure.coords,
        np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0], [7.0, 8.0, 9.0]]),
    )
    np.testing.assert_array_equal(structure.mask, np.array([True, False, True]))


def test_load_ca_coordinates_can_drop_missing_ca(tmp_path: Path) -> None:
    pdb_path = tmp_path / "missing_ca.pdb"
    pdb_path.write_text(
        "\n".join(
            [
                "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00 20.00           C",
                "ATOM      2  N   SER A   2       4.000   5.000   6.000  1.00 20.00           N",
                "ATOM      3  CA  GLY A   3       7.000   8.000   9.000  1.00 20.00           C",
                "TER",
                "END",
            ]
        )
        + "\n"
    )

    structure = load_ca_coordinates(pdb_path, chain_id="A", keep_missing_ca=False)

    assert structure.sequence == "AG"
    assert structure.residue_ids == ["A:ALA1", "A:GLY3"]
    np.testing.assert_array_equal(structure.mask, np.array([True, True]))


def test_load_ca_coordinates_reads_mmcif_chain(tmp_path: Path) -> None:
    mmcif_path = tmp_path / "tiny.cif"
    mmcif_path.write_text(
        "\n".join(
            [
                "data_tiny",
                "#",
                "loop_",
                "_atom_site.group_PDB",
                "_atom_site.id",
                "_atom_site.type_symbol",
                "_atom_site.label_atom_id",
                "_atom_site.label_alt_id",
                "_atom_site.label_comp_id",
                "_atom_site.label_asym_id",
                "_atom_site.label_entity_id",
                "_atom_site.label_seq_id",
                "_atom_site.pdbx_PDB_ins_code",
                "_atom_site.Cartn_x",
                "_atom_site.Cartn_y",
                "_atom_site.Cartn_z",
                "_atom_site.occupancy",
                "_atom_site.B_iso_or_equiv",
                "_atom_site.auth_seq_id",
                "_atom_site.auth_comp_id",
                "_atom_site.auth_asym_id",
                "_atom_site.auth_atom_id",
                "_atom_site.pdbx_PDB_model_num",
                "ATOM 1 C CA . ALA A 1 1 ? 0.000 0.000 0.000 1.00 20.00 1 ALA A CA 1",
                "ATOM 2 C CA . GLY A 1 2 ? 0.000 3.000 4.000 1.00 20.00 2 GLY A CA 1",
                "#",
            ]
        )
        + "\n"
    )

    structure = load_ca_coordinates(mmcif_path, chain_id="A")

    assert structure.chain_id == "A"
    assert structure.sequence == "AG"
    np.testing.assert_allclose(structure.coords, np.array([[0.0, 0.0, 0.0], [0.0, 3.0, 4.0]]))


def test_load_ca_coordinates_rejects_missing_chain(tmp_path: Path) -> None:
    pdb_path = tmp_path / "tiny.pdb"
    pdb_path.write_text(
        "\n".join(
            [
                "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C",
                "END",
            ]
        )
        + "\n"
    )

    try:
        load_ca_coordinates(pdb_path, chain_id="B")
    except ValueError as exc:
        assert "chain 'B' not found" in str(exc)
    else:
        raise AssertionError("expected missing chain to raise ValueError")


def test_normalize_pdb_id_uppercases_valid_id() -> None:
    assert normalize_pdb_id("1crn") == "1CRN"


def test_normalize_pdb_id_rejects_invalid_id() -> None:
    with pytest.raises(ValueError, match="PDB ID must be 4 alphanumeric characters"):
        normalize_pdb_id("abcde")


def test_pdb_id_to_mmcif_url_builds_rcsb_download_url() -> None:
    assert pdb_id_to_mmcif_url("1crn") == "https://files.rcsb.org/download/1CRN.cif"
