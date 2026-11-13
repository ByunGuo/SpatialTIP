"""Tokenization, training masks and distributed helpers used by the core model."""

import random

import numpy as np
import torch
import torch.distributed as dist


def tokenize_batch(gene_ids, append_cls=True, cls_id=0):
    if not append_cls:
        return gene_ids
    cls = torch.full((gene_ids.shape[0], 1), cls_id, device=gene_ids.device,
                     dtype=gene_ids.dtype)
    return torch.cat((cls, gene_ids), dim=1)


def mlm_mask(data, mask_token_id=-1, mask_prob_zero=0.05,
             mask_prob_nonzero=0.15, heldout=None):
    """Sample zeros/nonzeros separately; external heldout entries never become targets.

    Labels use -2 only to mark external heldout entries. The decoder preserves the
    original loss reduction over the full batch x gene matrix.
    """
    if not 0 <= mask_prob_zero <= 1 or not 0 <= mask_prob_nonzero <= 1:
        raise ValueError("Mask probabilities must be in [0, 1].")
    if data.ndim != 2 or not torch.isfinite(data).all() or (data < 0).any():
        raise ValueError("MLM input must be a finite nonnegative 2D expression tensor.")
    zero_mask = (data == 0) & (torch.rand_like(data) < mask_prob_zero)
    nonzero_mask = (data > 0) & (torch.rand_like(data) < mask_prob_nonzero)
    selected = zero_mask | nonzero_mask
    labels = data.detach().clone()
    if heldout is not None:
        if heldout.dtype != torch.bool or heldout.shape != data.shape:
            raise ValueError("heldout must be boolean True=heldout and match expression.")
        if (data[heldout] != 0).any():
            raise ValueError("External heldout entries must already be hidden as zero.")
        selected &= ~heldout
        labels[heldout] = -2
    masked = data.detach().clone()
    masked[selected] = mask_token_id
    if heldout is not None:
        masked[heldout] = mask_token_id
    return masked, labels


def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()


@torch.no_grad()
def concat_all_gather(tensor):
    """Preserve the original model's gather without remote-rank gradients."""
    if not is_dist_avail_and_initialized():
        return tensor
    output = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(output, tensor)
    return torch.cat(output, dim=0)


class GatherLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        gradients = torch.stack(grads)
        dist.all_reduce(gradients)
        return gradients[dist.get_rank()]


def all_gather_with_grad(tensor):
    if not is_dist_avail_and_initialized() or dist.get_world_size() == 1:
        return tensor
    return torch.cat(GatherLayer.apply(tensor), dim=0)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def cleanup():
    if is_dist_avail_and_initialized():
        dist.destroy_process_group()


class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg = self.sum = 0.0
        self.count = 0

    def update(self, value, count=1):
        self.sum += float(value) * count
        self.count += count
        self.avg = self.sum / self.count if self.count else float("nan")


def get_lr(optimizer):
    return optimizer.param_groups[0]["lr"]


def get_reduced(tensor, current_device, dest_device, world_size):
    tensor = torch.as_tensor(tensor, device=current_device).detach().clone()
    dist.reduce(tensor, dst=dest_device)
    return tensor.item() / world_size
