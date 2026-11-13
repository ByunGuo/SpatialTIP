"""Prepare HEST counts and paired patches using explicit, reusable mappings."""

import argparse
import csv
import json
import os
import re
import tempfile
from pathlib import Path


def _atomic_json(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".spatialtip-", suffix=".json", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _read_manifest(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"sample", "tissue", "h5ad", "patches"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Manifest needs columns sample,tissue,h5ad,patches")
        rows = []
        for row in reader:
            row = {key: (row.get(key) or "").strip() for key in required}
            if not all(row.values()):
                raise ValueError("Manifest rows cannot contain empty required values")
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", row["sample"]):
                raise ValueError(f"Unsafe sample identifier: {row['sample']}")
            for key in ("h5ad", "patches"):
                value = Path(row[key]).expanduser()
                value = value if value.is_absolute() else path.parent / value
                row[key] = value.resolve()
                if not row[key].is_file():
                    raise FileNotFoundError(row[key])
            rows.append(row)
    if not rows or len({row["sample"] for row in rows}) != len(rows):
        raise ValueError("Manifest must be nonempty with unique sample identifiers")
    return rows


def _load_mapping(path, vocabulary=False):
    with path.open(encoding="utf-8") as handle:
        mapping = json.load(handle)
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError(f"Mapping must be a nonempty object: {path}")
    if any(type(value) is not int or value < 0 for value in mapping.values()):
        raise ValueError(f"Mapping IDs must be nonnegative integers: {path}")
    if set(mapping.values()) != set(range(len(mapping))):
        raise ValueError(f"Mapping IDs must be unique and contiguous from zero: {path}")
    if vocabulary and mapping.get("CLS") != 0:
        raise ValueError("Vocabulary must reserve CLS=0")
    return mapping


def _prepare(row, n_top_genes, vocabulary=None):
    import h5py
    import numpy as np
    import scanpy as sc
    from scipy import sparse

    adata = sc.read_h5ad(row["h5ad"])
    if not adata.obs_names.is_unique or not adata.var_names.is_unique:
        raise ValueError(f"{row['sample']}: gene and spot identifiers must be unique")
    if "CLS" in adata.var_names:
        raise ValueError("CLS is reserved for the special token and cannot be a gene")
    if adata.n_obs < 2 or adata.n_vars < 2:
        raise ValueError(f"{row['sample']}: insufficient raw counts")
    values = adata.X.data if sparse.issparse(adata.X) else np.asarray(adata.X)
    if (not np.isfinite(values).all() or (values < 0).any()
            or not np.allclose(values, np.round(values), rtol=0, atol=1e-6)):
        raise ValueError(f"{row['sample']}: X must contain nonnegative raw counts, not log-normalized expression")
    # Select eligible genes using raw counts, while retaining the full input
    # gene axis for library-size normalization, including OOV gene counts.
    hvg_input = adata if vocabulary is None else adata[:, adata.var_names.isin(vocabulary)].copy()
    if hvg_input.n_vars < 2:
        raise ValueError(f"{row['sample']}: insufficient genes after vocabulary restriction")
    sc.pp.highly_variable_genes(hvg_input, n_top_genes=min(n_top_genes, hvg_input.n_vars), flavor="seurat_v3")
    selected_var = hvg_input.var.loc[hvg_input.var["highly_variable"]].copy()
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    adata = adata[:, selected_var.index].copy()
    adata.var = selected_var
    with h5py.File(row["patches"], "r") as handle:
        if "barcode" not in handle or "img" not in handle:
            raise ValueError(f"{row['sample']}: patches require barcode and img datasets")
        barcodes = handle["barcode"][:].reshape(-1)
        if len(barcodes) != len(handle["img"]):
            raise ValueError(f"{row['sample']}: patch image and barcode counts differ")
    def clean(value):
        if isinstance(value, (bytes, np.bytes_)):
            value = value.decode("utf-8")
        return str(value).strip("\x00 \t\r\n")
    barcodes = [clean(value) for value in barcodes]
    if len(set(barcodes)) != len(barcodes):
        raise ValueError(f"{row['sample']}: duplicate patch barcodes")
    clean_obs = [clean(value) for value in adata.obs_names]
    if len(set(clean_obs)) != len(clean_obs):
        raise ValueError(f"{row['sample']}: duplicate normalized expression barcodes")
    adata.obs_names = clean_obs
    adata = adata[adata.obs_names.isin(set(barcodes))].copy()
    if not adata.n_obs:
        raise ValueError(f"{row['sample']}: no expression spot matches a patch barcode")
    return adata


def run(args):
    manifest = Path(args.manifest).expanduser().resolve()
    rows = _read_manifest(manifest)
    output_dir = Path(args.output_dir).expanduser().resolve()
    vocabulary_path = Path(args.vocabulary).expanduser().resolve()
    tissue_path = Path(args.tissue_map).expanduser().resolve()
    if args.n_top_genes <= 0:
        raise ValueError("--n-top-genes must be positive")
    protected = {manifest, *(row[key] for row in rows for key in ("h5ad", "patches"))}
    output_manifest = output_dir / "manifest.csv"
    targets = [output_dir / f"{row['sample']}.h5ad" for row in rows] + [output_manifest]
    if args.command == "fit":
        targets += [vocabulary_path, tissue_path]
    else:
        protected |= {vocabulary_path, tissue_path}
    if len(set(targets)) != len(targets) or any(path in protected for path in targets):
        raise ValueError("Output paths collide with one another or with an input file")
    for target in targets:
        if target.exists() and (not args.overwrite or not target.is_file()):
            raise FileExistsError(f"Refusing to overwrite {target}; use --overwrite only for generated files")
    if args.command == "fit":
        genes = set()
        for row in rows:
            genes.update(_prepare(row, args.n_top_genes).var_names)
        vocabulary = {gene: index for index, gene in enumerate(["CLS"] + sorted(genes))}
        tissues = {tissue: index for index, tissue in enumerate(sorted({row['tissue'] for row in rows}))}
    else:
        vocabulary = _load_mapping(vocabulary_path, vocabulary=True)
        tissues = _load_mapping(tissue_path)
    missing_tissues = {row["tissue"] for row in rows} - tissues.keys()
    if missing_tissues:
        raise ValueError(f"Unknown tissues for this model: {sorted(missing_tissues)}")
    output_dir.mkdir(parents=True, exist_ok=True)
    processed = []
    for row in rows:
        # Transform restricts HVG eligibility to the frozen vocabulary; both
        # commands normalize on the original complete count gene axis.
        adata = _prepare(row, args.n_top_genes, vocabulary if args.command == "transform" else None)
        adata.obs["tissue"] = tissues[row["tissue"]]
        output = output_dir / f"{row['sample']}.h5ad"
        descriptor, temporary = tempfile.mkstemp(prefix=".spatialtip-", suffix=".h5ad", dir=output_dir)
        os.close(descriptor)
        try:
            adata.write_h5ad(temporary)
            os.replace(temporary, output)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        processed.append({"sample": row["sample"], "tissue": row["tissue"],
                          "h5ad": str(output), "patches": str(row["patches"]),
                          "n_spots": adata.n_obs, "n_genes": adata.n_vars})
        print(f"{row['sample']}: {adata.n_obs} spots x {adata.n_vars} genes -> {output}")
    if args.command == "fit":
        _atomic_json(vocabulary, vocabulary_path)
        _atomic_json(tissues, tissue_path)
    with output_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample", "tissue", "h5ad", "patches", "n_spots", "n_genes"])
        writer.writeheader()
        writer.writerows(processed)
    print(f"Vocabulary: {len(vocabulary) - 1} genes; tissues: {len(tissues)}; manifest: {output_manifest}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("fit", "transform"):
        sub = subparsers.add_parser(command, help="Create new mappings" if command == "fit" else "Reuse frozen model mappings")
        sub.add_argument("--manifest", required=True, help="CSV: sample,tissue,h5ad,patches; relative paths resolve beside this CSV")
        sub.add_argument("--output-dir", required=True)
        sub.add_argument("--vocabulary", required=True, help="JSON output for fit, existing JSON input for transform")
        sub.add_argument("--tissue-map", required=True, help="JSON output for fit, existing JSON input for transform")
        sub.add_argument("--n-top-genes", type=int, default=3000)
        sub.add_argument("--overwrite", action="store_true", help="Allow replacement of generated outputs, never input files")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
