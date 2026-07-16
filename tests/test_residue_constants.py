from open_alphafold2.constants import RESTYPES, RESTYPE_TO_INDEX, UNKNOWN_RESTYPE, encode_sequence


def test_restype_vocabulary_has_20_standard_amino_acids() -> None:
    assert len(RESTYPES) == 20
    assert len(set(RESTYPES)) == 20


def test_encode_sequence_maps_known_residues() -> None:
    encoded = encode_sequence("ARN")

    assert encoded == [
        RESTYPE_TO_INDEX["A"],
        RESTYPE_TO_INDEX["R"],
        RESTYPE_TO_INDEX["N"],
    ]


def test_encode_sequence_maps_unknown_residues_to_x() -> None:
    encoded = encode_sequence("AZ-")

    assert encoded == [
        RESTYPE_TO_INDEX["A"],
        RESTYPE_TO_INDEX[UNKNOWN_RESTYPE],
        RESTYPE_TO_INDEX[UNKNOWN_RESTYPE],
    ]
