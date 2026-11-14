"""Evaluate uploaded full-matrix reference predictions on frozen EDM masks."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import issparse, load_npz

from experiments.edm.metrics import recovery_metrics


ROOT = Path(__file__).resolve().parents[2]
SAMPLES = ["MISC1", "MISC2", "MISC5", "MISC9", "MISC10"]


def dense(value):
    return value.toarray() if issparse(value) else np.asarray(value)


def digest(path):
    result = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True)
    parser.add_argument("--ratio", type=float, choices=[0.2, 0.4, 0.6], default=0.4)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--pattern", default="{sample}_generated.h5ad")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", nargs="+", default=SAMPLES)
    parser.add_argument("--allow-extra-spots", action="store_true",
                        help="Allow prediction-only spots; missing target spots remain an error")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows, audit = [], {
        "method": args.method,
        "mask": "expression_gene",
        "mask_seed": 42,
        "mask_ratio": args.ratio,
        "alignment": "barcode and gene name; no positional fallback",
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "samples": {},
    }
    for sample in args.samples:
        prediction_path = args.prediction_dir / args.pattern.format(sample=sample)
        prediction = ad.read_h5ad(prediction_path)
        truth = ad.read_h5ad(ROOT / "hest_data/st_healthy_opus" / f"{sample}.h5ad")
        mask_dir = ROOT / "experiments/edm/healthy_expression_smoke/expression_gene/42" / f"{sample}_{args.ratio}"
        config = json.loads((mask_dir / "config.json").read_text())
        truth = truth[~truth.obs_names.isin(config["excluded_missing_patch_barcodes"])].copy()

        if not prediction.obs_names.is_unique or not prediction.var_names.is_unique:
            raise ValueError(f"{sample}: prediction IDs must be unique")
        missing = truth.obs_names.difference(prediction.obs_names)
        extra = prediction.obs_names.difference(truth.obs_names)
        if len(missing) or (len(extra) and not args.allow_extra_spots):
            raise ValueError(f"{sample}: spot mismatch, missing={len(missing)}, extra={len(extra)}")
        prediction = prediction[truth.obs_names].copy()
        common_genes = truth.var_names.intersection(prediction.var_names, sort=False)
        if len(common_genes) < 0.95 * truth.n_vars:
            raise ValueError(f"{sample}: only {len(common_genes)}/{truth.n_vars} target genes covered")

        truth_gene_indices = truth.var_names.get_indexer(common_genes)
        pred_gene_indices = prediction.var_names.get_indexer(common_genes)
        heldout = load_npz(mask_dir / "heldout.npz").toarray().astype(bool)[:, truth_gene_indices]
        truth_values = dense(truth.X)[:, truth_gene_indices]
        pred_values = dense(prediction.X)[:, pred_gene_indices]
        if not np.isfinite(pred_values).all() or (pred_values < 0).any():
            raise ValueError(f"{sample}: predictions must be finite and nonnegative")

        result = recovery_metrics(truth_values, pred_values, heldout)
        rows.append({"Sample": sample, "Method": args.method, **result})
        audit["samples"][sample] = {
            "prediction": str(prediction_path),
            "prediction_sha256": digest(prediction_path),
            "prediction_shape": list(prediction.shape),
            "excluded_prediction_only_spots": extra.tolist(),
            "excluded_prediction_only_spot_count": int(len(extra)),
            "target_gene_count": int(truth.n_vars),
            "covered_gene_count": int(len(common_genes)),
            "gene_coverage": float(len(common_genes) / truth.n_vars),
            "missing_target_genes": truth.var_names.difference(prediction.var_names).tolist(),
            "heldout_total": int(load_npz(mask_dir / "heldout.npz").sum()),
            "heldout_evaluated": int(heldout.sum()),
            "heldout_coverage": float(heldout.sum() / load_npz(mask_dir / "heldout.npz").sum()),
            "prediction_min": float(pred_values.min()),
            "prediction_max": float(pred_values.max()),
        }

    table = pd.DataFrame(rows)
    table.to_csv(args.output_dir / "metrics_internal.csv", index=False)
    manuscript = table[["Sample", "Method", "heldout_count", "RMSE", "MAE", "spot_PCC", "spot_CSS", "gene_PCC", "gene_CSS"]].rename(columns={
        "heldout_count": "Held-out entries",
        "spot_PCC": "Spot-wise PCC",
        "spot_CSS": "Spot-wise CSS",
        "gene_PCC": "Gene-wise PCC",
        "gene_CSS": "Gene-wise CSS",
    })
    manuscript.to_csv(args.output_dir / "imputation_metrics_manuscript.csv", index=False)
    macro = manuscript.drop(columns=["Sample", "Held-out entries"]).groupby("Method", as_index=False).mean(numeric_only=True)
    macro.to_csv(args.output_dir / "imputation_metrics_macro.csv", index=False)
    (args.output_dir / "alignment_audit.json").write_text(json.dumps(audit, indent=2))
    print(manuscript.to_string(index=False))
    print("\nMacro average\n", macro.to_string(index=False))


if __name__ == "__main__":
    main()
