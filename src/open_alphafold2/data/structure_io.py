"""Structure file readers backed by Biopython."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import numpy as np
from Bio.PDB import MMCIFParser, PDBParser
from Bio.PDB.Chain import Chain
from Bio.PDB.Model import Model
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
RCSB_DOWNLOAD_BASE_URL = "https://files.rcsb.org/download"

THREE_TO_ONE = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "MSE": "M",
}


@dataclass(frozen=True)
class CAStructure:
    """C-alpha coordinate view of one structure model and chain."""

    structure_id: str
    model_index: int
    chain_id: str
    sequence: str
    residue_ids: list[str]
    coords: FloatArray
    mask: BoolArray


def load_ca_coordinates(
    path: str | Path,
    chain_id: str | None = None,
    model_index: int = 0,
    keep_missing_ca: bool = True,
) -> CAStructure:
    """Load C-alpha coordinates from a PDB or mmCIF structure file.

    Args:
        path: Input `.pdb`, `.cif`, or `.mmcif` path.
        chain_id: Optional chain ID. If omitted, the first chain containing
            at least one C-alpha atom is used.
        model_index: Zero-based model index to read.
        keep_missing_ca: Preserve known amino-acid residues without C-alpha
            atoms and mark them with `mask=False`. If false, those residues are
            dropped.

    Returns:
        A C-alpha-only structure view with coordinates shaped `[num_res, 3]`.
    """

    structure_path = Path(path)
    parser = _parser_for_path(structure_path)
    structure = parser.get_structure(structure_path.stem, str(structure_path))
    models = list(structure.get_models())

    if model_index < 0 or model_index >= len(models):
        raise ValueError(
            f"model_index {model_index} is out of range for {structure_path}; "
            f"found {len(models)} model(s)"
        )

    model = models[model_index]
    chain = _select_chain(model, chain_id)
    return _extract_ca_structure(
        structure_path.stem,
        model_index,
        chain,
        keep_missing_ca=keep_missing_ca,
    )


def download_mmcif(
    pdb_id: str,
    cache_dir: str | Path | None = None,
    overwrite: bool = False,
) -> Path:
    """Download an RCSB mmCIF file for a PDB ID and return its local path."""

    normalized_pdb_id = normalize_pdb_id(pdb_id)
    output_dir = Path(cache_dir) if cache_dir is not None else _default_structure_cache_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{normalized_pdb_id}.cif"

    if output_path.exists() and not overwrite:
        return output_path

    url = pdb_id_to_mmcif_url(normalized_pdb_id)
    try:
        with urlopen(url, timeout=30) as response:
            output_path.write_bytes(response.read())
    except HTTPError as exc:
        raise ValueError(f"could not download PDB ID {normalized_pdb_id}: HTTP {exc.code}") from exc
    except URLError as exc:
        raise ValueError(f"could not download PDB ID {normalized_pdb_id}: {exc.reason}") from exc

    return output_path


def normalize_pdb_id(pdb_id: str) -> str:
    """Validate and normalize a 4-character PDB ID."""

    normalized = pdb_id.strip().upper()
    if len(normalized) != 4 or not normalized.isalnum():
        raise ValueError(f"PDB ID must be 4 alphanumeric characters, got '{pdb_id}'")
    return normalized


def pdb_id_to_mmcif_url(pdb_id: str) -> str:
    """Build the RCSB download URL for a normalized PDB ID."""

    normalized_pdb_id = normalize_pdb_id(pdb_id)
    return f"{RCSB_DOWNLOAD_BASE_URL}/{normalized_pdb_id}.cif"


def _parser_for_path(path: Path) -> PDBParser | MMCIFParser:
    suffix = path.suffix.lower()

    if suffix == ".pdb":
        return PDBParser(QUIET=True)
    if suffix in {".cif", ".mmcif"}:
        return MMCIFParser(QUIET=True)

    raise ValueError(
        f"unsupported structure format '{path.suffix}'; expected .pdb, .cif, or .mmcif"
    )


def _default_structure_cache_dir() -> Path:
    return Path.home() / ".cache" / "open_alphafold2" / "structures"


def _select_chain(model: Model, chain_id: str | None) -> Chain:
    if chain_id is not None:
        if not model.has_id(chain_id):
            available_chains = ", ".join(chain.id for chain in model.get_chains())
            raise ValueError(f"chain '{chain_id}' not found; available chains: {available_chains}")
        chain = model[chain_id]
        if _count_ca_atoms(chain) == 0:
            raise ValueError(f"chain '{chain_id}' does not contain C-alpha atoms")
        return chain

    for chain in model.get_chains():
        if _count_ca_atoms(chain) > 0:
            return chain

    raise ValueError("structure model does not contain any chain with C-alpha atoms")


def _extract_ca_structure(
    structure_id: str,
    model_index: int,
    chain: Chain,
    keep_missing_ca: bool,
) -> CAStructure:
    sequence: list[str] = []
    residue_ids: list[str] = []
    coords: list[FloatArray] = []
    mask: list[bool] = []

    for residue in chain.get_residues():
        residue_name = residue.get_resname().upper()
        if residue_name not in THREE_TO_ONE:
            continue

        has_ca = "CA" in residue
        if not has_ca and not keep_missing_ca:
            continue

        residue_ids.append(_format_residue_id(chain.id, residue.id, residue_name))
        sequence.append(THREE_TO_ONE.get(residue_name, "X"))
        mask.append(has_ca)
        if has_ca:
            coords.append(np.asarray(residue["CA"].coord, dtype=np.float64))
        else:
            coords.append(np.zeros((3,), dtype=np.float64))

    if not coords:
        raise ValueError(f"chain '{chain.id}' does not contain C-alpha atoms")

    coords_array = np.stack(coords, axis=0)
    mask_array = np.asarray(mask, dtype=np.bool_)
    return CAStructure(
        structure_id=structure_id,
        model_index=model_index,
        chain_id=chain.id,
        sequence="".join(sequence),
        residue_ids=residue_ids,
        coords=coords_array,
        mask=mask_array,
    )


def _count_ca_atoms(chain: Chain) -> int:
    return sum(1 for residue in chain.get_residues() if "CA" in residue)


def _format_residue_id(chain_id: str, residue_id: tuple[str, int, str], residue_name: str) -> str:
    hetero_flag, sequence_id, insertion_code = residue_id
    insertion = "" if insertion_code == " " else insertion_code
    hetero_prefix = "" if hetero_flag == " " else f"{hetero_flag}:"
    return f"{chain_id}:{hetero_prefix}{residue_name}{sequence_id}{insertion}"
