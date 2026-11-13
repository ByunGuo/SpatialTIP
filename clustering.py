"""Complete expression, apply spatial smoothing, and run PCA/mclust/refinement."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np


def complete_expression(original, predicted):
    """Retain observed nonzeros and fill original zero entries with predictions."""
    if original.shape != predicted.shape:
        raise ValueError("Original and predicted expression shapes differ")
    return np.where(original != 0, original, predicted)


def smooth_expression(matrix, coordinates, mode="radius", radius=150.0, knn_neighbors=10):
    """Match the original sequential update, including the center spot.

    Updating one row changes the values available to subsequent rows. This is
    deliberately distinct from simultaneous averaging of a frozen matrix.
    """
    if mode == "none":
        return matrix.copy()
    from sklearn.neighbors import NearestNeighbors
    if mode == "radius":
        if radius <= 0:
            raise ValueError("--radius must be positive")
        model = NearestNeighbors(radius=radius).fit(coordinates)
        indices = model.radius_neighbors(coordinates, return_distance=False)
    elif mode == "knn":
        if not 0 <= knn_neighbors < len(coordinates):
            raise ValueError("--knn-neighbors must be between zero and n_spots-1")
        model = NearestNeighbors(n_neighbors=knn_neighbors + 1).fit(coordinates)
        indices = model.kneighbors(coordinates, return_distance=False)
    else:
        raise ValueError(f"Unknown smoothing mode: {mode}")
    neighbors = {}
    for center, members in enumerate(indices):
        for member in members:
            member = int(member)
            neighbors.setdefault(center, set()).add(member)
            neighbors.setdefault(member, set()).add(center)
    output = np.asarray(matrix, dtype=np.result_type(matrix.dtype, np.float32)).copy()
    for center, members in neighbors.items():
        output[center] = output[list(members)].mean(axis=0)
    return output


def refine_labels(labels, coordinates, n_neighbors=30):
    """Simultaneous vote; tied labels follow first appearance among neighbors.

    Neighbors are ordered by distance, matching the historical vote rule.
    Exactly equal distances are ordered by spot index in this release; that
    deterministic distance-tie rule does not imply historical bitwise identity.
    """
    if n_neighbors == 0:
        return np.asarray(labels).copy()
    if not 0 < n_neighbors < len(labels):
        raise ValueError("--refine-neighbors must be zero or smaller than n_spots")
    from scipy.spatial import cKDTree
    tree = cKDTree(coordinates)
    distances, _ = tree.query(coordinates, k=n_neighbors + 1)
    output = []
    labels = np.asarray(labels, dtype=int)
    for center, distance in enumerate(distances):
        # Include all spots at the cutoff distance before breaking exact
        # distance ties by index; a k-query alone can omit tied candidates.
        cutoff = np.nextafter(distance[-1], np.inf)
        members = np.asarray(tree.query_ball_point(coordinates[center], cutoff), dtype=int)
        member_distances = np.linalg.norm(coordinates[members] - coordinates[center], axis=1)
        members = members[np.lexsort((members, member_distances))]
        members = members[members != center][:n_neighbors]
        neighbor_labels = labels[members].tolist()
        output.append(max(neighbor_labels, key=neighbor_labels.count))
    return np.asarray(output, dtype=int)


def run_mclust(embedding, n_clusters, seed, rscript="Rscript"):
    executable = shutil.which(rscript)
    if executable is None:
        raise RuntimeError("Rscript was not found. Install R and its mclust package, or set --rscript.")
    code = (
        'args<-commandArgs(TRUE); library(mclust); set.seed(as.integer(args[3])); '
        'x<-as.matrix(read.csv(args[1],header=FALSE)); '
        'fit<-Mclust(x,G=as.integer(args[4]),modelNames="EEE",verbose=FALSE); '
        'if(is.null(fit$classification))stop("mclust did not return labels"); '
        'write.table(fit$classification,args[2],row.names=FALSE,col.names=FALSE)'
    )
    with tempfile.TemporaryDirectory(prefix="spatialtip-mclust-") as temporary:
        source = Path(temporary) / "pca.csv"
        destination = Path(temporary) / "labels.txt"
        np.savetxt(source, embedding, delimiter=",", fmt="%.17g")
        result = subprocess.run([executable, "-e", code, str(source), str(destination),
                                 str(seed), str(n_clusters)], capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(f"R/mclust failed:\n{result.stderr.strip()}")
        labels = np.loadtxt(destination, dtype=int).reshape(-1)
    if len(labels) != len(embedding):
        raise RuntimeError("mclust label count does not match input spots")
    return labels


def run(args):
    import anndata as ad
    import pandas as pd
    from scipy import sparse
    from sklearn.decomposition import PCA
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    original_path = Path(args.original).expanduser().resolve()
    imputed_path = Path(args.imputed).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    if output_path in {original_path, imputed_path}:
        raise ValueError("--output cannot overwrite either input")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {output_path}; use --overwrite to replace generated output")
    raw = ad.read_h5ad(original_path)
    predicted = ad.read_h5ad(imputed_path)
    for name, value in (("original", raw), ("imputed", predicted)):
        if not value.obs_names.is_unique or not value.var_names.is_unique:
            raise ValueError(f"{name} spot and gene identifiers must be unique")
    if set(raw.obs_names) != set(predicted.obs_names) or set(raw.var_names) != set(predicted.var_names):
        raise ValueError("Inputs must contain identical spot and gene sets; preprocess/alignment is required")
    predicted = predicted[raw.obs_names, raw.var_names].copy()
    source = raw.X.toarray() if sparse.issparse(raw.X) else np.asarray(raw.X)
    prediction = predicted.X.toarray() if sparse.issparse(predicted.X) else np.asarray(predicted.X)
    if not np.isfinite(source).all() or not np.isfinite(prediction).all():
        raise ValueError("Expression contains nonfinite values")
    if "spatial" not in raw.obsm:
        raise ValueError("Original H5AD needs obsm['spatial'] coordinates")
    coordinates = np.asarray(raw.obsm["spatial"], dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2 or not np.isfinite(coordinates).all():
        raise ValueError("Spatial coordinates must be finite with shape (n_spots,2)")
    if not 1 < args.n_clusters < raw.n_obs:
        raise ValueError("--n-clusters must be between 2 and n_spots-1")
    if not 0 < args.pca_components < min(raw.shape):
        raise ValueError("--pca-components must be positive and smaller than both input dimensions")
    if not 0 <= args.refine_neighbors < raw.n_obs:
        raise ValueError("--refine-neighbors must be zero or smaller than n_spots")
    if args.label_key and args.label_key not in raw.obs:
        raise ValueError(f"Unknown original obs label column: {args.label_key}")
    if args.already_integrated:
        matrix = prediction.copy()
        smoothing = "none"
        print("--already-integrated: using the supplied imputed matrix unchanged; completion and spatial smoothing are skipped.")
    else:
        matrix = complete_expression(source, prediction)
        matrix = smooth_expression(matrix, coordinates, args.smoothing, args.radius, args.knn_neighbors)
        smoothing = args.smoothing
    embedding = PCA(n_components=args.pca_components, svd_solver="randomized", random_state=args.seed).fit_transform(matrix)
    labels = run_mclust(embedding, args.n_clusters, args.seed, args.rscript)
    refined = refine_labels(labels, coordinates, args.refine_neighbors)
    output = raw.copy()
    output.X = matrix
    output.obsm["X_pca"] = embedding
    output.obs["mclust"] = pd.Categorical(labels.astype(str))
    output.obs["spatialtip_cluster"] = pd.Categorical(refined.astype(str))
    if args.refine_neighbors:
        output.obs["mclust_refined"] = pd.Categorical(refined.astype(str))
    metadata = {"n_clusters": args.n_clusters, "pca_components": args.pca_components,
                "pca_solver": "randomized", "seed": args.seed, "mclust_model": "EEE",
                "smoothing": smoothing, "smoothing_update": "sequential_including_self",
                "radius": args.radius, "knn_neighbors": args.knn_neighbors,
                "refine_neighbors": args.refine_neighbors, "refinement_update": "simultaneous_first_nearest_label_vote_tie",
                "refinement_distance_tie": "spot_index_deterministic_release_rule",
                "already_integrated": args.already_integrated}
    if args.label_key:
        if args.label_key not in output.obs:
            raise ValueError(f"Unknown original obs label column: {args.label_key}")
        truth = output.obs[args.label_key]
        strings = truth.astype(str).str.strip()
        valid = truth.notna() & ~strings.str.lower().isin(["", "unknown", "na", "nan", "none"])
        if valid.sum() < 2:
            raise ValueError("Fewer than two valid reference labels; omit --label-key for unsupervised clustering")
        metadata["n_labeled_spots"] = int(valid.sum())
        for name, values in (("mclust", labels), ("spatialtip_cluster", refined)):
            metadata[f"{name}_ARI"] = float(adjusted_rand_score(strings[valid], values[valid.to_numpy()]))
            metadata[f"{name}_NMI"] = float(normalized_mutual_info_score(strings[valid], values[valid.to_numpy()]))
    output.uns["spatialtip_clustering"] = metadata
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".spatialtip-", suffix=".h5ad", dir=output_path.parent)
    os.close(descriptor)
    try:
        output.write_h5ad(temporary)
        os.replace(temporary, output_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    print(json.dumps(metadata, indent=2))
    print(f"Saved clustered expression and labels to {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", required=True, help="Processed log-normalized original H5AD")
    parser.add_argument("--imputed", required=True, help="Model prediction H5AD with matching spot and gene identifiers")
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-clusters", type=int, required=True, help="User-specified K; never inferred from reference labels")
    parser.add_argument("--pca-components", type=int, default=20)
    parser.add_argument("--smoothing", choices=["radius", "knn", "none"], default="radius")
    parser.add_argument("--radius", type=float, default=150.0, help="Smoothing radius in obsm['spatial'] coordinate units")
    parser.add_argument("--knn-neighbors", type=int, default=10, help="Neighbors excluding self for KNN smoothing")
    parser.add_argument("--refine-neighbors", type=int, default=30, help="Refinement neighbors excluding self; 0 disables")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--label-key", help="Optional original obs column for ARI/NMI; labels are not required to cluster")
    parser.add_argument("--rscript", default="Rscript", help="Rscript executable, found on PATH by default")
    parser.add_argument("--already-integrated", action="store_true", help="Use a previously completed/smoothed imputed matrix unchanged; skip completion and smoothing")
    parser.add_argument("--overwrite", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
