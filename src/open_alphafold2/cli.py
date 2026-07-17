from __future__ import annotations

import argparse
from pathlib import Path

from open_alphafold2.constants.residue_constants import RESTYPES
from open_alphafold2.data.samples import (
    make_ca_sample_from_structure,
    save_ca_sample,
)
from open_alphafold2.data.structure_io import download_mmcif, load_ca_coordinates
from open_alphafold2.geometry import pairwise_distances
from open_alphafold2.visualization.distance_plot import plot_distance_matrix


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "plot-distance":
        _plot_distance(args)
        return

    if args.command == "make-ca-sample":
        _make_ca_sample(args)
        return

    print("Open.AlphaFold2 MiniFold foundation")
    print(f"Residue vocabulary size: {len(RESTYPES)}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="open-alphafold2")
    subparsers = parser.add_subparsers(dest="command")

    plot_parser = subparsers.add_parser(
        "plot-distance",
        help="Plot a C-alpha pairwise distance heatmap from a local structure or PDB ID.",
    )
    _add_structure_arguments(plot_parser)
    plot_parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Output image path, usually .png.",
    )
    plot_parser.add_argument("--title", help="Optional plot title.")
    plot_parser.add_argument("--dpi", type=int, default=180, help="Output image DPI.")

    sample_parser = subparsers.add_parser(
        "make-ca-sample",
        help="Create a C-alpha distance training sample from a local structure or PDB ID.",
    )
    _add_structure_arguments(sample_parser)
    sample_parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Output .npz sample path.",
    )

    return parser


def _add_structure_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "structure",
        nargs="?",
        type=Path,
        help="Input .pdb, .cif, or .mmcif file. Omit when using --pdb-id.",
    )
    parser.add_argument(
        "--pdb-id",
        help="Download this 4-character PDB ID from RCSB as mmCIF before processing.",
    )
    parser.add_argument(
        "--structure-cache",
        type=Path,
        help="Directory for downloaded PDB ID mmCIF files.",
    )
    parser.add_argument(
        "--overwrite-structure-cache",
        action="store_true",
        help="Redownload the PDB ID even if it already exists in the cache.",
    )
    parser.add_argument("--chain", help="Chain ID to extract. Defaults to the first CA chain.")
    parser.add_argument(
        "--model",
        type=int,
        default=0,
        help="Model index to extract. Defaults to 0.",
    )
    parser.add_argument(
        "--drop-missing-ca",
        action="store_true",
        help="Drop amino-acid residues without C-alpha atoms instead of masking them.",
    )


def _plot_distance(args: argparse.Namespace) -> None:
    structure_path = _resolve_structure_path(args)
    ca_structure = load_ca_coordinates(
        structure_path,
        chain_id=args.chain,
        model_index=args.model,
        keep_missing_ca=not args.drop_missing_ca,
    )
    distances = pairwise_distances(ca_structure.coords, ca_structure.mask)
    title = args.title or f"{structure_path.name} chain {ca_structure.chain_id}"

    plot_distance_matrix(
        distances,
        args.out,
        title=title,
        residue_ids=ca_structure.residue_ids,
        dpi=args.dpi,
    )

    print(
        "Wrote pairwise C-alpha distance plot "
        f"for chain {ca_structure.chain_id} ({len(ca_structure.residue_ids)} residues): {args.out}"
    )


def _make_ca_sample(args: argparse.Namespace) -> None:
    structure_path = _resolve_structure_path(args)
    sample = make_ca_sample_from_structure(
        structure_path,
        chain_id=args.chain,
        model_index=args.model,
        source_path=structure_path,
        keep_missing_ca=not args.drop_missing_ca,
    )
    output_path = save_ca_sample(sample, args.out)

    print(
        "Wrote C-alpha training sample "
        f"for chain {sample.chain_id} ({len(sample.residue_ids)} residues): {output_path}"
    )


def _resolve_structure_path(args: argparse.Namespace) -> Path:
    structure = getattr(args, "structure", None)

    if structure is None and args.pdb_id is None:
        raise SystemExit("error: provide either a local structure path or --pdb-id")

    if structure is not None and args.pdb_id is not None:
        raise SystemExit("error: provide a local structure path or --pdb-id, not both")

    if args.pdb_id is not None:
        return download_mmcif(
            args.pdb_id,
            cache_dir=args.structure_cache,
            overwrite=args.overwrite_structure_cache,
        )

    return structure
