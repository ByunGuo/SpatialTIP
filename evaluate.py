"""Evaluate explicitly held-out nonzeros, with optional expression completion/smoothing."""

import argparse
import csv
import hashlib
import json
from pathlib import Path


def recovery_metrics(truth, prediction, heldout):
    import numpy as np

    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    heldout = np.asarray(heldout)
    if truth.shape != prediction.shape or heldout.shape != truth.shape or heldout.dtype != bool:
        raise ValueError("Aligned matrices and a boolean True=heldout mask are required.")
    if not heldout.any() or (truth[heldout] <= 0).any():
        raise ValueError("Evaluation requires held-out observed nonzero entries.")
    if not np.isfinite(truth[heldout]).all() or not np.isfinite(prediction[heldout]).all():
        raise ValueError("Nonfinite truth or prediction at held-out entries.")
    errors = truth[heldout] - prediction[heldout]
    result = {"RMSE": float(np.sqrt(np.mean(errors ** 2))),
              "MAE": float(np.mean(np.abs(errors))), "heldout_count": int(heldout.sum())}
    target = np.where(heldout, truth, 0)
    predicted = np.where(heldout, prediction, 0)
    for name, axis in (("spot", 1), ("gene", 0)):
        counts = heldout.sum(axis=axis, keepdims=True)
        target_centered = (target - target.sum(axis=axis, keepdims=True) / np.maximum(counts, 1)) * heldout
        prediction_centered = (predicted - predicted.sum(axis=axis, keepdims=True) / np.maximum(counts, 1)) * heldout
        denominator = np.sqrt((target_centered ** 2).sum(axis=axis) * (prediction_centered ** 2).sum(axis=axis))
        valid = (counts.reshape(-1) >= 2) & (denominator > 0)
        correlation = (target_centered * prediction_centered).sum(axis=axis)[valid] / denominator[valid]
        norm = np.sqrt((target ** 2).sum(axis=axis) * (predicted ** 2).sum(axis=axis))
        valid_cosine = norm > 0
        cosine = (target * predicted).sum(axis=axis)[valid_cosine] / norm[valid_cosine]
        for metric, values in (("PCC", correlation), ("CSS", cosine)):
            values = values[np.isfinite(values)]
            result[f"{name}_{metric}"] = float(values.mean()) if len(values) else None
            result[f"{name}_{metric}_valid_count"] = len(values)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", required=True)
    parser.add_argument("--imputed", required=True, help="Dense predictions or another explicitly chosen prediction stage")
    parser.add_argument("--heldout", required=True, help="Boolean sparse NPZ in the truth H5AD axis order")
    parser.add_argument("--output", required=True, help="New CSV file")
    parser.add_argument("--input", help="Masked input, required for completion and optional smoothing")
    parser.add_argument("--smoothing", choices=("none", "radius", "knn"), default="none")
    parser.add_argument("--radius", type=float, default=150.0)
    parser.add_argument("--knn-neighbors", type=int, default=10)
    args = parser.parse_args()
    import anndata as ad
    import numpy as np
    from scipy import sparse

    inputs = [Path(getattr(args, key)).expanduser().resolve() for key in ("truth", "imputed", "heldout")]
    output = Path(args.output).expanduser().resolve()
    metadata_path = output.with_suffix(".metadata.json")
    if args.input:
        inputs.append(Path(args.input).expanduser().resolve())
    if output in inputs or metadata_path in inputs or output == metadata_path:
        raise ValueError("Output cannot overwrite an input file.")
    for path in (output, metadata_path):
        if path.exists():
            raise FileExistsError("Refusing to overwrite " + str(path))
    truth = ad.read_h5ad(inputs[0])
    if not truth.obs_names.is_unique or not truth.var_names.is_unique:
        raise ValueError("Truth identifiers must be unique.")

    def align(path):
        data = ad.read_h5ad(path)
        if not data.obs_names.is_unique or not data.var_names.is_unique:
            raise ValueError("Prediction/input identifiers must be unique.")
        if set(data.obs_names) != set(truth.obs_names) or set(data.var_names) != set(truth.var_names):
            raise ValueError("Prediction/input must have exactly the truth's spot and gene sets.")
        return data[truth.obs_names, truth.var_names].copy()

    def dense(data):
        return data.X.toarray() if sparse.issparse(data.X) else np.asarray(data.X)

    predictions = dense(align(inputs[1]))
    target = dense(truth)
    heldout = sparse.load_npz(inputs[2]).toarray()
    mask_metadata = inputs[2].with_suffix(".metadata.json")
    if mask_metadata.is_file():
        saved = json.loads(mask_metadata.read_text(encoding="utf-8"))
        expected = saved.get("truth_sha256")
        if expected and hashlib.sha256(inputs[0].read_bytes()).hexdigest() != expected:
            raise ValueError("Truth SHA256 does not match the held-out mask metadata.")
    rows = [{"output": "supplied_prediction", **recovery_metrics(target, predictions, heldout)}]
    if args.input:
        from clustering import complete_expression, smooth_expression
        masked_input = align(inputs[3])
        observed = dense(masked_input)
        if heldout.dtype != bool or heldout.shape != observed.shape or np.any(observed[heldout] != 0):
            raise ValueError("Held-out entries must be hidden as zero in --input.")
        completed = complete_expression(observed, predictions)
        rows.append({"output": "completed", **recovery_metrics(target, completed, heldout)})
        if args.smoothing != "none":
            coordinates = np.asarray(masked_input.obsm["spatial"], dtype=float)
            if coordinates.shape != (truth.n_obs, 2) or not np.isfinite(coordinates).all():
                raise ValueError("Input needs finite two-dimensional spatial coordinates.")
            smoothed = smooth_expression(completed, coordinates, args.smoothing, args.radius, args.knn_neighbors)
            rows.append({"output": "smoothed", **recovery_metrics(target, smoothed, heldout)})
    elif args.smoothing != "none":
        raise ValueError("--smoothing requires --input, so observed expression can be retained first.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {"mask_definition": "boolean True=heldout, indexed by the truth H5AD order",
                "metric_definition": "RMSE/MAE over pooled heldout entries; spot/gene PCC and CSS are means over valid vectors only.",
                "smoothing": args.smoothing, "smoothing_update": "sequential_including_self",
                "radius": args.radius, "knn_neighbors": args.knn_neighbors,
                "inputs_sha256": {key: hashlib.sha256(path.read_bytes()).hexdigest()
                                  for key, path in zip(("truth", "imputed", "heldout", "input"), inputs)}}
    with metadata_path.open("x", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)
        stream.write("\n")
    print(json.dumps(rows, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
