"""Export dense gene-expression predictions without replacing observed values.

The default input mask hides zero entries (zeros -> -1). The explicit 'observed'
option reproduces the old impute.py behavior of hiding nonzero entries instead;
it is a different input condition. Spatial smoothing and observed-value merging
are separate operations and are not performed by this command.
"""

import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import random
import uuid


def add_inference_arguments(parser):
    parser.add_argument('--expression', required=True)
    parser.add_argument('--patches', required=True)
    parser.add_argument('--vocabulary', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True, help='New output file; an existing file is never overwritten')
    parser.add_argument('--device', default='auto', help='auto, cpu, cuda, or cuda:N')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--input-mask', choices=['zeros', 'observed', 'none'], default='zeros')
    parser.add_argument('--resize-images', action='store_true')
    parser.add_argument('--seed', type=int, default=2024)


def validate_inference_arguments(args):
    for name in ['expression', 'patches', 'vocabulary', 'config', 'checkpoint']:
        path = Path(getattr(args, name)).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError('{} does not exist: {}'.format(name, path))
        setattr(args, name, str(path))
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError('batch-size must be positive and num-workers nonnegative')
    if getattr(args, 'max_spots', None) is not None and args.max_spots < 1:
        raise ValueError('max-spots must be positive when supplied')
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError('Refusing to overwrite output: {}'.format(output))
    output.parent.mkdir(parents=True, exist_ok=True)
    args.output = str(output)


def iter_model_outputs(args, *, output_attentions=False):
    """Yield (N x G numpy predictions, optional layer tensors) in input order."""
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from dataset import SpatialTIPDataset
    from runtime import load_config, load_model
    from utils import tokenize_batch

    value = args.device
    if value == 'auto':
        value = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = torch.device(value)
    if device.type not in {'cpu', 'cuda'}:
        raise ValueError('Only CPU and CUDA devices are supported')
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA was requested but is unavailable')
        torch.cuda.set_device(device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
    dataset = SpatialTIPDataset(args.expression, args.patches, args.vocabulary, resize_images=args.resize_images)
    try:
        config = load_config(args.config)
        if int(dataset.tissue_type.max()) >= config['tissue_type']:
            raise ValueError('Expression tissue IDs exceed the configured tissue embedding table')
        if dataset.exp_mtx.shape[1] + 1 > config['max_seq_len']:
            raise ValueError('The gene panel plus CLS exceeds configured max_seq_len')
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=device.type == 'cuda', drop_last=False)
        model = load_model(args.config, args.vocabulary, checkpoint=args.checkpoint, device=str(device))
        model.eval()
        processed = 0
        limit = getattr(args, 'max_spots', None)
        with torch.inference_mode():
            for batch in loader:
                if limit is not None:
                    remaining = limit - processed
                    if remaining <= 0:
                        break
                    batch = {key: value[:remaining] if torch.is_tensor(value) else value for key, value in batch.items()}
                batch = {key: value.to(device, non_blocking=True) for key, value in batch.items() if torch.is_tensor(value)}
                batch['gene_id'] = tokenize_batch(batch['gene_id'], append_cls=True, cls_id=0)
                expression = batch['gene_exp'].clone()
                if args.input_mask == 'zeros':
                    expression[expression == 0] = -1
                elif args.input_mask == 'observed':
                    expression[expression != 0] = -1
                batch['gene_exp'] = expression
                batch.pop('labels', None)
                autocast = torch.autocast(device_type='cuda', dtype=torch.float16) if device.type == 'cuda' else contextlib.nullcontext()
                with autocast:
                    output = model(batch, neighbors=None, mode='test', output_attentions=output_attentions)
                if output_attentions:
                    predictions, attention = output
                else:
                    predictions, attention = output, None
                if predictions.ndim != 2 or predictions.shape[1] != expression.shape[1]:
                    raise ValueError('Model output does not match the input gene axis')
                if not torch.isfinite(predictions).all().item():
                    raise FloatingPointError('Model produced non-finite predictions')
                processed += predictions.shape[0]
                yield predictions.float().cpu().numpy(), attention
    finally:
        close = getattr(dataset, 'close', None)
        if close is not None:
            close()


def run_imputation(args):
    import anndata as ad
    import numpy as np

    validate_inference_arguments(args)
    adata = ad.read_h5ad(args.expression)
    predictions = np.empty(adata.shape, dtype=np.float32)
    position = 0
    for batch, _ in iter_model_outputs(args):
        predictions[position:position + batch.shape[0]] = batch
        position += batch.shape[0]
    if position != adata.n_obs:
        raise ValueError('Expected predictions for all {} spots; received {}'.format(adata.n_obs, position))
    adata.X = predictions
    adata.uns['spatialtip_inference'] = {
        'input_mask': args.input_mask, 'processed_spots': position, 'seed': args.seed,
        'checkpoint': str(Path(args.checkpoint).name),
        'vocabulary_sha256': hashlib.sha256(Path(args.vocabulary).read_bytes()).hexdigest(),
        'output_definition': 'Dense model predictions in the original spot/gene order; no observed-value merging or spatial smoothing.',
    }
    output = Path(args.output)
    temporary = output.with_name('.' + output.name + '.' + uuid.uuid4().hex + '.tmp.h5ad')
    try:
        adata.write_h5ad(temporary, compression='gzip')
        # Hard-link creation is atomic and refuses an existing destination.
        os.link(str(temporary), str(output))
    finally:
        if temporary.exists():
            temporary.unlink()
    print('Saved {} spots x {} genes to {}'.format(adata.n_obs, adata.n_vars, output), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_inference_arguments(parser)
    run_imputation(parser.parse_args())


if __name__ == '__main__':
    main()
