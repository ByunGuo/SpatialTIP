"""Train SpatialTIP from a sample manifest without mixing gene panels in a batch.

This release retains warmup/cosine restarts and makes the stage-specific split
and schedule explicit, with reliable single-process or torchrun management.
"""

import argparse
import contextlib
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random


def add_training_arguments(parser, *, finetune=False):
    parser.add_argument('--config', required=True)
    parser.add_argument('--vocabulary', required=True)
    if finetune:
        parser.add_argument('--checkpoint', required=True)
        parser.set_defaults(vision_checkpoint=None)
    else:
        initialization = parser.add_mutually_exclusive_group(required=True)
        initialization.add_argument('--vision-checkpoint', help='UNI initialization for a new SpatialTIP model')
        initialization.add_argument('--checkpoint', help='Initialize from a complete SpatialTIP model checkpoint; optimizer state is not resumed')
    parser.add_argument('--output-dir', required=True, help='New or empty directory; existing results are never adopted or overwritten')
    parser.add_argument('--epochs', type=int, default=60 if finetune else 80)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--min-learning-rate', type=float, default=5e-5)
    parser.add_argument('--warmup-epochs', type=int, default=2 if finetune else 5)
    parser.add_argument('--cycle-epochs', type=int, default=30 if finetune else 40)
    parser.add_argument('--cycle-decay', type=float, default=0.98, help='Multiply the peak learning rate by this factor after each restart')
    parser.add_argument('--train-fraction', type=float, default=0.8 if finetune else 0.9)
    parser.add_argument('--mask-prob', type=float, default=0.6 if finetune else 0.15, help='MLM corruption probability for nonzero expression')
    parser.add_argument('--zero-mask-prob', type=float, default=0.05, help='MLM corruption probability for observed zero values')
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--device', default='auto', help='auto, cpu, cuda, or cuda:N; torchrun uses LOCAL_RANK on CUDA')
    parser.add_argument('--patience', type=int, default=10, help='Stop after this many epochs without validation improvement; 0 disables early stopping')
    parser.add_argument('--resize-images', action='store_true', help='Explicitly resize RGB patches to 224 x 224 with bilinear interpolation')


def _existing_file(value, label):
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError('{} does not exist: {}'.format(label, path))
    return path


def read_manifest(path):
    manifest = _existing_file(path, 'Manifest')
    samples = []
    names = set()
    with manifest.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        required = {'sample', 'tissue', 'h5ad', 'patches'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError('Manifest must have columns: sample,tissue,h5ad,patches')
        for line_number, row in enumerate(reader, 2):
            if not any((value or '').strip() for value in row.values()):
                continue
            if any(not (row.get(key) or '').strip() for key in required):
                raise ValueError('Missing manifest value at line {}'.format(line_number))
            name = row['sample'].strip()
            if name in names:
                raise ValueError('Duplicate manifest sample: {}'.format(name))
            names.add(name)
            record = {'sample': name, 'tissue': row['tissue'].strip()}
            for source_key, target_key in [('h5ad', 'expression'), ('patches', 'patches')]:
                value = Path(row[source_key].strip()).expanduser()
                if not value.is_absolute():
                    value = manifest.parent / value
                record[target_key] = str(_existing_file(value, '{} for {}'.format(source_key, name)))
            record['heldout'] = None
            samples.append(record)
    if not samples:
        raise ValueError('Manifest contains no samples')
    return samples


def _seed_worker(worker_id):
    import numpy as np
    import torch
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _setup_device(value):
    import torch
    import torch.distributed as dist
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    if value == 'auto':
        value = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(value)
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA was requested but is unavailable')
        index = local_rank if world_size > 1 else (device.index if device.index is not None else 0)
        device = torch.device('cuda', index)
        torch.cuda.set_device(device)
    elif device.type != 'cpu':
        raise ValueError('Only CPU and CUDA devices are supported')
    initialized_here = False
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend='nccl' if device.type == 'cuda' else 'gloo', init_method='env://')
        initialized_here = True
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    return device, rank, world_size, initialized_here


def _reserve_output_directory(path, rank, world_size):
    import torch.distributed as dist
    output = Path(path).expanduser().resolve()
    error = None
    if rank == 0:
        try:
            if output.exists() and (not output.is_dir() or any(output.iterdir())):
                raise ValueError('Output directory must be new or empty: {}'.format(output))
            output.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            error = '{}: {}'.format(type(exc).__name__, exc)
    if world_size > 1:
        errors = [error]
        dist.broadcast_object_list(errors, src=0)
        error = errors[0]
    if error:
        raise RuntimeError(error)
    return output


def _autocast(device):
    import torch
    return torch.autocast(device_type='cuda', dtype=torch.float16) if device.type == 'cuda' else contextlib.nullcontext()


def _make_scaler(device):
    import torch
    if hasattr(torch, 'amp') and hasattr(torch.amp, 'GradScaler'):
        return torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    return torch.cuda.amp.GradScaler(enabled=device.type == 'cuda')


def _epoch(model, loader, args, device, *, optimizer=None, scaler=None):
    import torch
    from utils import mlm_mask, tokenize_batch
    total_loss = 0.0
    total_spots = 0
    training = optimizer is not None
    context = contextlib.nullcontext() if training else torch.no_grad()
    with context:
        for batch in loader:
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items() if torch.is_tensor(value)}
            batch['gene_id'] = tokenize_batch(batch['gene_id'], append_cls=True, cls_id=0)
            batch['gene_exp'], batch['labels'] = mlm_mask(
                batch['gene_exp'], mask_token_id=-1,
                mask_prob_zero=args.zero_mask_prob, mask_prob_nonzero=args.mask_prob,
                heldout=batch.get('heldout'),
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
            with _autocast(device):
                loss = model(batch, neighbors=None, mode='train', method='contrastive')
            if not torch.isfinite(loss).item():
                raise FloatingPointError('Training or validation produced a non-finite loss')
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            count = batch['gene_exp'].shape[0]
            total_loss += float(loss.detach().item()) * count
            total_spots += count
    if total_spots == 0:
        raise ValueError('A sample produced an empty training or validation loader')
    return total_loss, total_spots


def _reduce_totals(loss_sum, spots, device, world_size):
    import torch
    import torch.distributed as dist
    values = torch.tensor([loss_sum, spots], dtype=torch.float64, device=device)
    if world_size > 1:
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    return values[0].item() / values[1].item(), int(values[1].item())


def _save_checkpoint(path, model, config, vocabulary_sha256, epoch, validation_loss):
    import torch
    source_model = model.module if hasattr(model, 'module') else model
    temporary = path.with_name('.' + path.name + '.tmp')
    payload = {
        'state_dict': source_model.state_dict(),
        'config': config,
        'vocabulary_sha256': vocabulary_sha256,
        'epoch': epoch,
        'validation_loss': validation_loss,
        'training_management': 'release_warmup_cosine_restarts_v1',
    }
    try:
        torch.save(payload, temporary)
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def run_training(args, samples):
    import numpy as np
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    from torch.utils.data import DataLoader, random_split
    from torch.utils.data.distributed import DistributedSampler
    from dataset import SpatialTIPDataset
    from runtime import load_config, load_model

    for key in ['config', 'vocabulary']:
        setattr(args, key, str(_existing_file(getattr(args, key), key)))
    if args.checkpoint:
        args.checkpoint = str(_existing_file(args.checkpoint, 'Checkpoint'))
    if args.vision_checkpoint:
        args.vision_checkpoint = str(_existing_file(args.vision_checkpoint, 'Vision checkpoint'))
    if args.epochs < 1 or args.batch_size < 1 or args.num_workers < 0 or args.patience < 0:
        raise ValueError('epochs/batch-size must be positive; num-workers/patience must be nonnegative')
    if not 0 <= args.mask_prob <= 1 or not 0 <= args.zero_mask_prob <= 1:
        raise ValueError('Mask probabilities must lie in [0, 1]')
    if args.learning_rate <= 0 or not 0 <= args.min_learning_rate <= args.learning_rate:
        raise ValueError('Require 0 <= min-learning-rate <= learning-rate and learning-rate > 0')
    if not 0 <= args.warmup_epochs < args.cycle_epochs:
        raise ValueError('Require 0 <= warmup-epochs < cycle-epochs')
    if not 0 < args.cycle_decay <= 1 or not 0 < args.train_fraction < 1:
        raise ValueError('Require 0 < cycle-decay <= 1 and 0 < train-fraction < 1')
    device, rank, world_size, initialized_here = _setup_device(args.device)
    datasets = []
    try:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(args.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        output = _reserve_output_directory(args.output_dir, rank, world_size)
        config = load_config(args.config)
        vocabulary_hash = hashlib.sha256(Path(args.vocabulary).read_bytes()).hexdigest()
        loaders = []
        split_summary = []
        for index, sample in enumerate(samples):
            dataset = SpatialTIPDataset(
                sample['expression'], sample['patches'], args.vocabulary,
                resize_images=args.resize_images, heldout_path=sample.get('heldout'),
            )
            datasets.append(dataset)
            if len(dataset) < 2:
                raise ValueError('At least two spots are required for sample {}'.format(sample['sample']))
            if int(np.max(dataset.tissue_type)) >= config['tissue_type']:
                raise ValueError("Sample {} has a tissue ID outside config tissue_type={}".format(sample['sample'], config['tissue_type']))
            if dataset.adata.n_vars + 1 > config['max_seq_len']:
                raise ValueError('Sample {} has {} genes plus CLS, exceeding config max_seq_len={}'.format(sample['sample'], dataset.adata.n_vars, config['max_seq_len']))
            train_size = min(len(dataset) - 1, max(1, int(args.train_fraction * len(dataset))))
            training, validation = random_split(
                dataset, [train_size, len(dataset) - train_size],
                generator=torch.Generator().manual_seed(args.seed),
            )
            train_sampler = DistributedSampler(training, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed) if world_size > 1 else None
            validation_sampler = DistributedSampler(validation, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None
            shared = dict(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=device.type == 'cuda', drop_last=False, worker_init_fn=_seed_worker)
            train_loader = DataLoader(training, sampler=train_sampler, shuffle=train_sampler is None, generator=torch.Generator().manual_seed(args.seed + index), **shared)
            validation_loader = DataLoader(validation, sampler=validation_sampler, shuffle=False, **shared)
            loaders.append((train_loader, validation_loader))
            split_summary.append({'sample': sample['sample'], 'train_spots': len(training), 'validation_spots': len(validation)})
        model = load_model(args.config, args.vocabulary, checkpoint=args.checkpoint,
                           vision_checkpoint=args.vision_checkpoint, device=str(device))
        if world_size > 1:
            options = {'find_unused_parameters': True}
            if device.type == 'cuda':
                options['device_ids'] = [device.index]
            model = DistributedDataParallel(model, **options)
        optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=args.learning_rate)
        scaler = _make_scaler(device)
        def learning_rate_multiplier(epoch):
            # Independent expression of the periodic schedule; no scBERT code
            # is bundled. Epoch zero starts at the minimum learning rate.
            if epoch == 0:
                return args.min_learning_rate / args.learning_rate
            cycle, position = divmod(epoch, args.cycle_epochs)
            peak = args.learning_rate * args.cycle_decay ** cycle
            if args.warmup_epochs and position < args.warmup_epochs:
                rate = args.min_learning_rate + (peak - args.min_learning_rate) * position / args.warmup_epochs
            else:
                phase = (position - args.warmup_epochs) / (args.cycle_epochs - args.warmup_epochs)
                rate = args.min_learning_rate + (peak - args.min_learning_rate) * (1 + math.cos(math.pi * phase)) / 2
            return rate / args.learning_rate

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_multiplier)
        if rank == 0:
            metadata = {'arguments': vars(args), 'samples': samples, 'split': split_summary,
                        'config': config, 'vocabulary_sha256': vocabulary_hash,
                        'world_size': world_size, 'device': str(device),
                        'training_management': 'release_warmup_cosine_restarts_v1',
                        'scheduler': 'Periodic cosine with linear warmup; pretraining defaults cycle=40/warmup=5, fine-tuning cycle=30/warmup=2, peak decay=0.98.',
                        'distributed_sampling': 'DistributedSampler pads each subset to equal per-rank length; validation loss includes padded repeats when needed.',
                        'batching': 'Samples are loaded separately; batches never mix different gene panels.',
                        'initialization': 'Model weights only; optimizer/scheduler state starts fresh.'}
            (output / 'run.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
            with (output / 'training.csv').open('w', encoding='utf-8', newline='') as handle:
                csv.writer(handle).writerow(['epoch', 'learning_rate', 'train_loss', 'validation_loss', 'train_spots_including_padding', 'validation_spots_including_padding'])
        best_loss = float('inf')
        best_epoch = 0
        for epoch in range(args.epochs):
            model.train()
            train_sum, train_count = 0.0, 0
            order = list(range(len(loaders)))
            random.Random(args.seed + epoch).shuffle(order)
            for index in order:
                train_loader = loaders[index][0]
                if world_size > 1:
                    train_loader.sampler.set_epoch(epoch)
                value, count = _epoch(model, train_loader, args, device, optimizer=optimizer, scaler=scaler)
                train_sum += value
                train_count += count
            train_loss, train_spots = _reduce_totals(train_sum, train_count, device, world_size)
            model.eval()
            validation_sum, validation_count = 0.0, 0
            fork_devices = [device.index] if device.type == 'cuda' else []
            with torch.random.fork_rng(devices=fork_devices):
                torch.manual_seed(args.seed + 10000 + rank)
                if device.type == 'cuda':
                    torch.cuda.manual_seed(args.seed + 10000 + rank)
                for _, validation_loader in loaders:
                    value, count = _epoch(model, validation_loader, args, device)
                    validation_sum += value
                    validation_count += count
            validation_loss, validation_spots = _reduce_totals(validation_sum, validation_count, device, world_size)
            improved = validation_loss < best_loss
            if improved:
                best_loss, best_epoch = validation_loss, epoch + 1
            if rank == 0:
                rate = optimizer.param_groups[0]['lr']
                print('Epoch {}/{}: train={:.6f}, validation={:.6f}, lr={:.6g}'.format(epoch + 1, args.epochs, train_loss, validation_loss, rate), flush=True)
                with (output / 'training.csv').open('a', encoding='utf-8', newline='') as handle:
                    csv.writer(handle).writerow([epoch + 1, rate, train_loss, validation_loss, train_spots, validation_spots])
                _save_checkpoint(output / 'last.pt', model, config, vocabulary_hash, epoch + 1, validation_loss)
                if improved:
                    _save_checkpoint(output / 'best.pt', model, config, vocabulary_hash, epoch + 1, validation_loss)
            if world_size > 1:
                dist.barrier()
            scheduler.step()
            if args.patience and epoch + 1 - best_epoch >= args.patience:
                break
        if rank == 0:
            print('Best epoch: {}; validation loss: {:.6f}; output: {}'.format(best_epoch, best_loss, output), flush=True)
    finally:
        for dataset in datasets:
            close = getattr(dataset, 'close', None)
            if close is not None:
                close()
        if initialized_here and dist.is_initialized():
            dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, help='CSV columns sample,tissue,h5ad,patches; relative file paths resolve against the CSV directory')
    add_training_arguments(parser)
    args = parser.parse_args()
    run_training(args, read_manifest(args.manifest))


if __name__ == '__main__':
    main()
