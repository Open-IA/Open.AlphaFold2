"""Training sample creation for C-alpha distance supervision."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from open_alphafold2.constants import encode_sequence
from open_alphafold2.data.structure_io import load_ca_coordinates
from open_alphafold2.geometry import pairwise_distances

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]
DEFAULT_DISTOGRAM_BIN_EDGES = np.linspace(2.0, 22.0, 63, dtype=np.float64)


@dataclass(frozen=True)
class CASample:
    """Serializable C-alpha training sample for MiniFold v1."""

    structure_id: str
    model_index: int
    chain_id: str
    sequence: str
    sequence_encoded: IntArray
    residue_ids: list[str]
    ca_coords: FloatArray
    mask: BoolArray
    distances: FloatArray
    distogram_bins: IntArray
    distogram_bin_edges: FloatArray
    source_path: str


def make_ca_sample_from_structure(
    path: str | Path,
    chain_id: str | None = None,
    model_index: int = 0,
    source_path: str | Path | None = None,
    keep_missing_ca: bool = True,
) -> CASample:
    """Create a C-alpha distance training sample from a PDB or mmCIF file."""

    ca_structure = load_ca_coordinates(
        path,
        chain_id=chain_id,
        model_index=model_index,
        keep_missing_ca=keep_missing_ca,
    )
    sequence_encoded = np.asarray(encode_sequence(ca_structure.sequence), dtype=np.int64)
    distances = pairwise_distances(ca_structure.coords, ca_structure.mask)
    distogram_bins = make_distogram_bins(distances, ca_structure.mask)
    resolved_source_path = Path(source_path if source_path is not None else path)

    return CASample(
        structure_id=ca_structure.structure_id,
        model_index=ca_structure.model_index,
        chain_id=ca_structure.chain_id,
        sequence=ca_structure.sequence,
        sequence_encoded=sequence_encoded,
        residue_ids=ca_structure.residue_ids,
        ca_coords=ca_structure.coords,
        mask=ca_structure.mask,
        distances=distances,
        distogram_bins=distogram_bins,
        distogram_bin_edges=DEFAULT_DISTOGRAM_BIN_EDGES,
        source_path=str(resolved_source_path),
    )


def make_distogram_bins(
    distances: FloatArray,
    mask: BoolArray,
    bin_edges: FloatArray = DEFAULT_DISTOGRAM_BIN_EDGES,
) -> IntArray:
    """Convert pairwise distances to distogram class IDs.

    Invalid masked pairs are assigned `-1` so loss code can ignore them.
    """

    bins = np.digitize(distances, bin_edges, right=False).astype(np.int64)
    bins[~np.logical_and(mask[:, None], mask[None, :])] = -1
    return bins


def save_ca_sample(sample: CASample, output_path: str | Path) -> Path:
    """Save a C-alpha training sample as a compressed `.npz` file."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output,
        structure_id=np.array(sample.structure_id),
        model_index=np.array(sample.model_index, dtype=np.int64),
        chain_id=np.array(sample.chain_id),
        sequence=np.array(sample.sequence),
        sequence_encoded=sample.sequence_encoded,
        residue_ids=np.asarray(sample.residue_ids, dtype=np.str_),
        ca_coords=sample.ca_coords,
        mask=sample.mask,
        distances=sample.distances,
        distogram_bins=sample.distogram_bins,
        distogram_bin_edges=sample.distogram_bin_edges,
        source_path=np.array(sample.source_path),
    )
    return output


def load_ca_sample(path: str | Path) -> CASample:
    """Load a C-alpha training sample from a `.npz` file."""

    with np.load(Path(path), allow_pickle=False) as data:
        return CASample(
            structure_id=str(data["structure_id"].item()),
            model_index=int(data["model_index"].item()),
            chain_id=str(data["chain_id"].item()),
            sequence=str(data["sequence"].item()),
            sequence_encoded=np.asarray(data["sequence_encoded"], dtype=np.int64),
            residue_ids=[str(residue_id) for residue_id in data["residue_ids"].tolist()],
            ca_coords=np.asarray(data["ca_coords"], dtype=np.float64),
            mask=np.asarray(data["mask"], dtype=np.bool_),
            distances=np.asarray(data["distances"], dtype=np.float64),
            distogram_bins=np.asarray(data["distogram_bins"], dtype=np.int64),
            distogram_bin_edges=np.asarray(data["distogram_bin_edges"], dtype=np.float64),
            source_path=str(data["source_path"].item()),
        )
