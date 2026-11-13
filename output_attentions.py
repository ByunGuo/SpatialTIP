"""Export mean gene self-attention pre-softmax logits, not probabilities.

The output averages the actual returned self-attention scores over spots, heads
and layers after removing the CLS row/column. These model-internal associations
do not establish causal gene regulation. All spots are processed unless an
explicit --max-spots limit is provided. Attention extraction uses the explicit
attention backend and can require substantial GPU/CPU memory.
"""

import argparse
import json
from pathlib import Path

from impute import add_inference_arguments, iter_model_outputs, validate_inference_arguments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_inference_arguments(parser)
    parser.add_argument('--max-spots', type=int, help='Explicitly process only this many leading spots; omitted means every spot')
    args = parser.parse_args()
    validate_inference_arguments(args)
    import anndata as ad
    import numpy as np
    import torch

    reference = ad.read_h5ad(args.expression, backed='r')
    try:
        genes = np.asarray(reference.var_names, dtype=str)
        total_spots = reference.n_obs
    finally:
        reference.file.close()
    accumulator = None
    processed = 0
    layer_count = None
    head_counts = None
    for predictions, attentions in iter_model_outputs(args, output_attentions=True):
        if not attentions:
            raise ValueError('The model returned no self-attention scores')
        counts = [int(layer.shape[1]) for layer in attentions]
        if layer_count is None:
            layer_count = len(attentions)
            head_counts = counts
        elif layer_count != len(attentions) or head_counts != counts:
            raise ValueError('Attention layer/head dimensions changed between batches')
        for layer in attentions:
            expected = (len(genes) + 1, len(genes) + 1)
            if layer.ndim != 4 or tuple(layer.shape[-2:]) != expected:
                raise ValueError('Attention scores do not match the gene axis plus CLS')
            if not torch.isfinite(layer).all().item():
                raise FloatingPointError('Attention extraction produced non-finite scores')
            contribution = layer[:, :, 1:, 1:].float().mean(dim=1).sum(dim=0).cpu().numpy()
            if accumulator is None:
                accumulator = contribution.astype(np.float64)
            else:
                accumulator += contribution
        processed += predictions.shape[0]
        print('Accumulated attention from {} spots'.format(processed), flush=True)
    expected_spots = min(total_spots, args.max_spots) if args.max_spots else total_spots
    if processed != expected_spots or accumulator is None:
        raise ValueError('Expected {} processed spots, received {}'.format(expected_spots, processed))
    mean_scores = (accumulator / (processed * layer_count)).astype(np.float32)
    definition = ('Gene self-attention pre-softmax logits returned by SpatialTIP, averaged equally over heads within each layer, then over layers and processed spots; CLS removed. '
                  'Rows are query genes and columns are key genes. Scores include the additive attention-mask constants used by the explicit backend. '
                  'Values are not normalized probabilities or evidence of causal regulation.')
    metadata = {'definition': definition, 'input_mask': args.input_mask,
                'processed_spots': processed, 'total_input_spots': total_spots,
                'max_spots': args.max_spots, 'layers': layer_count,
                'heads_per_layer': head_counts, 'seed': args.seed,
                'checkpoint': Path(args.checkpoint).name}
    with Path(args.output).open('xb') as handle:
        np.savez_compressed(handle, mean_self_attention_logits=mean_scores, gene_names=genes,
                            processed_spots=np.int64(processed), definition=np.asarray(definition),
                            metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)))
    print('Saved {} x {} mean attention logits to {}'.format(len(genes), len(genes), args.output), flush=True)


if __name__ == '__main__':
    main()
