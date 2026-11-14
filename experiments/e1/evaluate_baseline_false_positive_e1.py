"""Compare manuscript baselines on frozen E1 false-positive proxy labels.

The cross-section and layer-marker labels must already exist.  This script never
rebuilds labels from predictions.  All methods are evaluated on the per-slice
gene intersection shared by every method, so the small OmiCLIP coverage gap does
not give any method a different evaluation set.
"""
import argparse
import os
import datetime
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.neighbors import NearestNeighbors
from spatial_smoothing import spatial_smoothing


ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "experiments/e1/runs"
OUT = RUNS / "baseline_false_positive_v1"
SAMPLES = ["MISC1", "MISC2", "MISC5", "MISC9", "MISC10"]
METHODS = [
    "SpatialTIP", "MAGIC", "GraphST", "STAGATE", "SEDR", "MIST",
    "stLearn", "BLEEP", "OmiCLIP",
]
PROXY_DIR = RUNS / "cross_section_zero_proxy_v1"
MARKER_DIR = RUNS / "layer_marker_distortion_v1"


def dense(value):
    return value.toarray() if hasattr(value, "toarray") else np.asarray(value)


def mask_dir(sample):
    return ROOT / f"experiments/e1/healthy_expression_smoke/expression_gene/42/{sample}_0.4"


def prediction_path(method, sample):
    manifest = os.environ.get("SPATIALTIP_PREDICTION_MANIFEST")
    if manifest:
        return Path(json.loads(Path(manifest).read_text())[method][sample])
    paths = {
        "SpatialTIP": ROOT / ("experiments/e1/pilot_MISC1/formal_expression_gene_42_0.4/prediction.h5ad" if sample == "MISC1" else f"experiments/e1/runs/{sample}_expression_gene_42_0.4_formal/prediction.h5ad"),
        "MAGIC": RUNS / f"healthy_magic_expression_gene_42_0.4/{sample}_prediction.h5ad",
        "GraphST": RUNS / f"uploaded_graph_baselines_expression_gene_42_0.4_formal/GraphST/GraphST_{sample}_prediction.h5ad",
        "STAGATE": RUNS / f"uploaded_graph_baselines_expression_gene_42_0.4_formal/STAGATE/STAGATE_{sample}_prediction.h5ad",
        "SEDR": RUNS / f"uploaded_graph_baselines_expression_gene_42_0.4_formal/SEDR/SEDR_{sample}_prediction.h5ad",
        "MIST": RUNS / f"MIST_expression_gene_42_0.4_formal/{sample}/MIST_{sample}_prediction.h5ad",
        "stLearn": RUNS / f"stLearn_expression_gene_42_0.4_formal/stLearn_{sample}_prediction.h5ad",
        "BLEEP": ROOT.parent / f"stGPT/BLEEP_results/predicted_healthy_opus/{sample}_generated.h5ad",
        "OmiCLIP": ROOT.parent / f"stGPT/OmiCLIP_results/predicted_healthy_opus/{sample}_generated.h5ad",
    }
    return paths[method]


def read_target(sample):
    cfg = json.loads((mask_dir(sample) / "config.json").read_text())
    target = ad.read_h5ad(ROOT / f"hest_data/st_healthy_opus/{sample}.h5ad")
    return target[~target.obs_names.isin(cfg["excluded_missing_patch_barcodes"])].copy()


def shared_genes(sample, target, methods):
    shared = set(target.var_names)
    audit = {}
    for method in methods:
        path = prediction_path(method, sample)
        if not path.exists():
            raise FileNotFoundError(path)
        obj = ad.read_h5ad(path, backed="r")
        if not obj.obs_names.is_unique or not obj.var_names.is_unique:
            raise ValueError(f"{method}/{sample}: non-unique IDs")
        if set(obj.obs_names) != set(target.obs_names):
            raise ValueError(f"{method}/{sample}: barcode mismatch")
        covered = target.var_names.intersection(obj.var_names, sort=False)
        audit[method] = {
            "path": str(path), "shape": list(obj.shape),
            "covered_target_genes": int(len(covered)),
            "target_gene_coverage": float(len(covered) / target.n_vars),
        }
        shared &= set(obj.var_names)
        obj.file.close()
    genes = target.var_names[target.var_names.isin(shared)]
    if len(genes) < 0.95 * target.n_vars:
        raise ValueError(f"{sample}: shared gene coverage only {len(genes)}/{target.n_vars}")
    return genes, audit


def read_prediction(method, sample, target, genes, truth, heldout):
    obj = ad.read_h5ad(prediction_path(method, sample), backed="r")
    # h5py permits one fancy axis per read. Select genes on disk first (critical
    # for BLEEP's 33,538-gene files), then reorder spots in memory.
    subset = obj[:, genes].to_memory()
    obj.file.close()
    values = dense(subset[target.obs_names].X).astype(np.float32)
    if not np.isfinite(values).all():
        raise ValueError(f"{method}/{sample}: non-finite predictions")
    if method != "SpatialTIP":
        return values
    # Reproduce the manuscript SpatialTIP output: direct model imputation followed
    # by the same radius-150 spatial smoothing used in the completed E1 analysis.
    return spatial_smoothing(np.where(heldout,0,truth), values,
                             target.obsm["spatial"],radius=150).astype(np.float32)


def proxy_metrics(sample, method, rule, positive, negative, scores):
    selected = positive | negative
    labels = positive[selected].astype(np.uint8)
    values = scores[selected]
    fpr, tpr, _ = roc_curve(labels, values, drop_intermediate=False)
    allowed = tpr[fpr <= 0.05 + 1e-12]
    return {
        "Sample": sample,
        "Method": method,
        "Proxy rule": rule,
        "Positive proxies": int(positive.sum()),
        "Negative proxies": int(negative.sum()),
        "Proxy AUROC": float(roc_auc_score(labels, values)),
        "Proxy AUPRC": float(average_precision_score(labels, values)),
        "Recall at 5% FPR": float(allowed.max()) if len(allowed) else 0.0,
        "Positive mean prediction": float(scores[positive].mean()),
        "Negative mean prediction": float(scores[negative].mean()),
    }


def marker_metrics(sample, method, target, genes, truth, scores, labels):
    # Restore every originally observed nonzero so this endpoint isolates the
    # effect of imputing original zeros.
    completed = np.where(truth == 0, scores, truth)
    gene_to_index = {gene: index for index, gene in enumerate(genes)}
    rows = []
    labels = labels[labels["Gene"].isin(gene_to_index)].copy()
    labels["Shared gene index"] = labels["Gene"].map(gene_to_index)
    layer_values = target.obs["_e1_layer"].to_numpy()
    valid = layer_values != "nan"
    for layer, group in labels.groupby("Layer"):
        inside = valid & (layer_values == str(layer))
        outside = valid & (layer_values != str(layer))
        effect = completed[inside].mean(0) - completed[outside].mean(0)
        indices = group["Shared gene index"].to_numpy(int)
        y = group["Label"].to_numpy(int)
        score = effect[indices]
        rows.append({
            "Sample": sample, "Method": method, "Layer": str(layer),
            "Positive gene-layers": int(y.sum()),
            "Negative gene-layers": int((y == 0).sum()),
            "Marker AUROC": float(roc_auc_score(y, score)) if len(set(y)) == 2 else np.nan,
            "Marker AUPRC": float(average_precision_score(y, score)) if y.sum() else np.nan,
            "Positive marker mean effect": float(score[y == 1].mean()) if y.sum() else np.nan,
            "Negative marker mean effect": float(score[y == 0].mean()) if (y == 0).sum() else np.nan,
            "Negative-marker false enrichment": float((score[y == 0] > 0.25).mean()) if (y == 0).sum() else np.nan,
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", nargs="+", choices=SAMPLES, default=SAMPLES)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    annotation = pd.read_csv(
        ROOT / "experiments/e1/reference_annotations/dlpfc_barcode_level_layer_map.tsv",
        sep="\t", header=None, names=["barcode", "sid", "layer"], dtype=str,
    )
    section_ids = {"MISC1": "151676", "MISC2": "151675", "MISC5": "151672", "MISC9": "151510", "MISC10": "151509"}
    audit = {
        "status": "running",
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "mask": "E1 expression_gene, seed 42, ratio 0.4",
        "methods": args.methods, "samples": args.samples,
        "comparison_universe": "per-slice spot set and gene intersection shared by all evaluated methods",
        "labels": "pre-existing prediction-blind cross-section and external layer-marker proxies",
        "samples_audit": {},
    }
    (args.output_dir / "evaluation_config.json").write_text(json.dumps(audit, indent=2))
    proxy_rows, marker_rows = [], []
    for sample in args.samples:
        target = read_target(sample)
        genes, coverage = shared_genes(sample, target, args.methods)
        target = target[:, genes].copy()
        truth = dense(target.X).astype(np.float32)
        target_indices = read_target(sample).var_names.get_indexer(genes)
        heldout = load_npz(mask_dir(sample) / "heldout.npz")[:, target_indices].toarray().astype(bool)
        layers = annotation[annotation.sid == section_ids[sample]].set_index("barcode").layer.reindex(target.obs_names)
        target.obs["_e1_layer"] = layers.astype(str).str.replace("L", "", regex=False).replace({"2/3": "3"}).to_numpy()
        positive = load_npz(PROXY_DIR / f"{sample}_positive_dropout_proxy.npz")[:, target_indices].toarray().astype(bool)
        negatives = {
            "primary_5pct": load_npz(PROXY_DIR / f"{sample}_negative_primary_5pct.npz")[:, target_indices].toarray().astype(bool),
            "strict_2pct": load_npz(PROXY_DIR / f"{sample}_negative_strict_2pct.npz")[:, target_indices].toarray().astype(bool),
        }
        marker_labels = pd.read_csv(MARKER_DIR / f"{sample}_external_marker_labels.csv")
        audit["samples_audit"][sample] = {
            "target_spots": int(target.n_obs), "target_genes": 6000,
            "shared_genes": int(len(genes)), "shared_gene_coverage": float(len(genes) / 6000),
            "methods": coverage,
        }
        for method in args.methods:
            scores = read_prediction(method, sample, target, genes, truth, heldout)
            for rule, negative in negatives.items():
                proxy_rows.append(proxy_metrics(sample, method, rule, positive, negative, scores))
            marker_rows.extend(marker_metrics(sample, method, target, genes, truth, scores, marker_labels))
        pd.DataFrame(proxy_rows).to_csv(args.output_dir / "proxy_metrics_by_slice.csv", index=False)
        pd.DataFrame(marker_rows).to_csv(args.output_dir / "layer_marker_metrics.csv", index=False)
        (args.output_dir / "evaluation_config.json").write_text(json.dumps(audit, indent=2))

    proxy = pd.DataFrame(proxy_rows)
    proxy_macro = proxy.groupby(["Method", "Proxy rule"], as_index=False)[
        ["Positive proxies", "Negative proxies", "Proxy AUROC", "Proxy AUPRC", "Recall at 5% FPR", "Positive mean prediction", "Negative mean prediction"]
    ].mean()
    proxy_macro.to_csv(args.output_dir / "proxy_metrics_macro.csv", index=False)
    marker = pd.DataFrame(marker_rows)
    marker_macro = marker.groupby("Method", as_index=False)[
        ["Marker AUROC", "Marker AUPRC", "Positive marker mean effect", "Negative marker mean effect", "Negative-marker false enrichment"]
    ].mean()
    observed_rows = []
    # Compute the common-universe observed-input marker denominator per slice.
    for sample in args.samples:
        target = read_target(sample)
        genes, _ = shared_genes(sample, target, args.methods)
        target = target[:, genes].copy()
        truth = dense(target.X).astype(np.float32)
        layers = annotation[annotation.sid == section_ids[sample]].set_index("barcode").layer.reindex(target.obs_names)
        target.obs["_e1_layer"] = layers.astype(str).str.replace("L", "", regex=False).replace({"2/3": "3"}).to_numpy()
        labels = pd.read_csv(MARKER_DIR / f"{sample}_external_marker_labels.csv")
        observed_rows.extend(marker_metrics(sample, "Observed input", target, genes, truth, truth, labels))
    observed_effect = pd.DataFrame(observed_rows)["Positive marker mean effect"].mean()
    marker_macro["Positive-marker effect retention"] = marker_macro["Positive marker mean effect"] / observed_effect
    marker_macro.to_csv(args.output_dir / "layer_marker_metrics_macro.csv", index=False)
    primary = proxy_macro[proxy_macro["Proxy rule"] == "primary_5pct"]
    final = primary[["Method", "Proxy AUPRC", "Recall at 5% FPR", "Proxy AUROC"]].merge(
        marker_macro[["Method", "Negative-marker false enrichment", "Positive-marker effect retention", "Marker AUROC", "Marker AUPRC"]],
        on="Method", validate="one_to_one",
    )
    final["Method"] = pd.Categorical(final["Method"], METHODS, ordered=True)
    final = final.sort_values("Method")
    final.to_csv(args.output_dir / "main_comparison_table.csv", index=False)
    audit.update(
        status="completed",
        completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        observed_positive_marker_mean_effect=float(observed_effect),
        limitations=[
            "Proxy labels are operational cross-section evidence, not molecular ground truth.",
            "All methods use the same shared gene universe per slice; OmiCLIP determines the small intersection loss.",
            "Marker effect endpoints remain sensitive to method-specific output calibration.",
        ],
    )
    (args.output_dir / "evaluation_config.json").write_text(json.dumps(audit, indent=2))
    print(final.to_string(index=False))


if __name__ == "__main__":
    main()
