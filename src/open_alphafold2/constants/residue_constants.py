"""Protein residue constants used by the MiniFold data pipeline."""

from __future__ import annotations

RESTYPES = tuple("ARNDCQEGHILKMFPSTWYV")
UNKNOWN_RESTYPE = "X"
RESTYPES_WITH_UNKNOWN = RESTYPES + (UNKNOWN_RESTYPE,)

RESTYPE_TO_INDEX = {restype: index for index, restype in enumerate(RESTYPES_WITH_UNKNOWN)}
INDEX_TO_RESTYPE = {index: restype for restype, index in RESTYPE_TO_INDEX.items()}


def encode_sequence(sequence: str) -> list[int]:
    """Encode an amino-acid sequence as integer residue IDs.

    Unknown or non-canonical residue letters map to ``X``.
    """

    unknown_index = RESTYPE_TO_INDEX[UNKNOWN_RESTYPE]
    return [RESTYPE_TO_INDEX.get(restype.upper(), unknown_index) for restype in sequence]
