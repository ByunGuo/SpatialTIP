"""Rank model-internal query/key gene associations from exported attention logits."""

import argparse
import csv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attention", required=True, help="NPZ produced by output_attentions.py")
    parser.add_argument("--output", required=True, help="New edge CSV")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--query", nargs="+", help="Optional query genes; default is all genes")
    args = parser.parse_args()
    import numpy as np

    source = Path(args.attention).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if output == source or output.exists():
        raise FileExistsError("Refusing to overwrite " + str(output))
    with np.load(source, allow_pickle=False) as archive:
        genes = archive["gene_names"].astype(str)
        scores = archive["mean_self_attention_logits"]
    if scores.shape != (len(genes), len(genes)) or len(set(genes)) != len(genes):
        raise ValueError("Attention matrix must be square with unique gene identifiers.")
    if not np.isfinite(scores).all():
        raise ValueError("Attention scores must be finite.")
    if not 0 < args.top_k < len(genes):
        raise ValueError("--top-k must be positive and smaller than the number of genes.")
    index = {gene: i for i, gene in enumerate(genes)}
    queries = args.query if args.query is not None else genes
    if any(gene not in index for gene in queries):
        raise ValueError("A requested query gene is absent from the attention matrix.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["query_gene", "key_gene", "rank", "mean_self_attention_logit"])
        for gene in queries:
            query_index = index[gene]
            ranking = np.argsort(-scores[query_index], kind="stable")
            ranking = ranking[ranking != query_index][:args.top_k]
            for rank, key_index in enumerate(ranking, 1):
                writer.writerow([gene, genes[key_index], rank, float(scores[query_index, key_index])])
    print(f"Saved ranked query/key associations to {output}.")


if __name__ == "__main__":
    main()
