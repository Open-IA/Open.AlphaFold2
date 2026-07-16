# Open.AlphaFold2

This repo will start as a **mini AlphaFold2** implementation: a small,
testable PyTorch model that learns protein backbone structure on short
single-chain proteins before scaling toward more AlphaFold2 components.

The first milestone is not full AlphaFold2. It is **MiniFold v0**:

- define protein constants and tensor conventions
- parse or synthesize simple protein examples
- predict coarse C-alpha geometry for short chains
- overfit one tiny example
- keep every module small enough to test directly

## First Step

Build the foundation before model code:

1. Keep the project reproducible with `uv`.
2. Define the package layout under `src/open_alphafold2`.
3. Add protein constants and basic tests.
4. Add geometry utilities next.
5. Only then implement a tiny sequence-to-distance model.

## Local Setup

```bash
uv sync --dev
uv run pytest
uv run open-alphafold2
```

If tests fail with `ModuleNotFoundError: open_alphafold2`, refresh the editable
install after pulling or uploading the latest repo:

```bash
uv sync --dev --reinstall-package open-alphafold2
uv run pytest
```

Training dependencies are installed on the server rather than on the local
machine:

```bash
uv sync --dev --extra train
```

## Near-Term Roadmap

### MiniFold v0: Foundation

- amino-acid vocabulary
- residue and atom constants
- tensor shape conventions
- geometry utilities
- tests for every low-level helper

### MiniFold v1: Toy Structure Predictor

- sequence embedding
- pair representation
- distogram head
- loss on pairwise C-alpha distances
- overfit one protein

### MiniFold v2: AF2-Style Core

- small Evoformer-like block
- recycling interface
- structure module skeleton
- FAPE-style backbone loss

### MiniFold v3: Realistic Inputs

- MSA feature tensors
- deletion features
- cropped training examples
- distributed training entrypoint

## Hardware Target

The intended training machine is an 8x RTX 4090 server. Start with short
chains and tiny configs before scaling:

- sequence length: 64-128 first, then 256
- MSA depth: 1 first, then 16-64
- batch size per GPU: 1
- mixed precision
- gradient accumulation
- activation checkpointing once the model grows
