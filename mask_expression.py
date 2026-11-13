"""Create a uniform held-out mask over observed nonzero expression entries."""

import argparse
import hashlib
import json
from pathlib import Path
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", required=True, help="Processed, unmasked H5AD")
    parser.add_argument("--output", required=True, help="New masked H5AD")
    parser.add_argument("--heldout", required=True, help="New boolean sparse NPZ; True means held out")
    parser.add_argument("--probability", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2024)
    args = parser.parse_args()
    if not 0 < args.probability < 1:
        raise ValueError("--probability must be between 0 and 1.")
    truth_path = Path(args.truth).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    heldout_path = Path(args.heldout).expanduser().resolve()
    metadata_path = heldout_path.with_suffix(".metadata.json")
    targets = [output, heldout_path, metadata_path]
    if len(set(targets + [truth_path])) != 4:
        raise ValueError("Input, output, mask and mask metadata must have distinct paths.")
    for path in targets:
        if path.exists():
            raise FileExistsError("Refusing to overwrite " + str(path))
    import anndata as ad
    import numpy as np
    from scipy import sparse

    data = ad.read_h5ad(truth_path)
    if not data.obs_names.is_unique or not data.var_names.is_unique:
        raise ValueError("Input spot/gene identifiers must be unique.")
    source = data.X.toarray() if sparse.issparse(data.X) else np.asarray(data.X)
    if not np.isfinite(source).all() or (source < 0).any():
        raise ValueError("Input must contain finite, nonnegative processed expression.")
    heldout = (source > 0) & (np.random.default_rng(args.seed).random(source.shape) < args.probability)
    if not heldout.any():
        raise ValueError("No entries selected; use a larger input or probability.")
    masked = source.copy()
    masked[heldout] = 0
    data.X = sparse.csr_matrix(masked) if sparse.issparse(data.X) else masked
    metadata = {
        "definition": "Uniform independent Bernoulli masking of observed nonzero entries; True=heldout.",
        "probability": args.probability, "seed": args.seed,
        "heldout_count": int(heldout.sum()), "eligible_nonzero_count": int((source > 0).sum()),
        "truth_sha256": hashlib.sha256(truth_path.read_bytes()).hexdigest(),
        "shape": list(source.shape),
    }
    data.uns["spatialtip_mask"] = {k: v for k, v in metadata.items() if k != "shape"}
    for path in targets:
        path.parent.mkdir(parents=True, exist_ok=True)
    # Prepare all artifacts before claiming any output name. Exclusive writes
    # preserve pre-existing files even if another process creates them meanwhile.
    with tempfile.TemporaryDirectory(prefix="spatialtip-mask-") as temporary:
        prepared = Path(temporary) / "masked.h5ad"
        data.write_h5ad(prepared, compression="gzip")
        with output.open("xb") as destination, prepared.open("rb") as source_file:
            import shutil
            shutil.copyfileobj(source_file, destination)
        with heldout_path.open("xb") as destination:
            sparse.save_npz(destination, sparse.csr_matrix(heldout))
        with metadata_path.open("x", encoding="utf-8") as destination:
            json.dump(metadata, destination, indent=2)
            destination.write("\n")
    print(f"Held out {int(heldout.sum())} observed entries; wrote {output} and {heldout_path}.")


if __name__ == "__main__":
    main()
