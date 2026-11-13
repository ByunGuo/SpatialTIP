"""Aligned expression/image input for SpatialTIP training and inference."""

import json
import os
from pathlib import Path

import h5py
import numpy as np
import scanpy as sc
import torch
from PIL import Image
from scipy import sparse
from torch.utils.data import Dataset


def normalize_barcode(value):
    """Decode HEST byte strings and remove surrounding whitespace/padding."""
    if isinstance(value, (bytes, np.bytes_)):
        value = value.decode("utf-8")
    return str(value).strip("\x00 \t\r\n")


class SpatialTIPDataset(Dataset):
    """Read a processed H5AD and match image patches by barcode, never by position.

    H5AD must contain numeric ``obs['tissue']`` and genes present in the supplied
    vocabulary. Patch HDF5 requires ``img`` (N, H, W, 3 uint8) and ``barcode``.
    Expression is materialized in memory; image handles are opened per worker.
    An optional boolean sparse NPZ uses True for held-out entries, which must
    already have been set to zero in the expression input.
    """

    def __init__(self, expression_path, patches_path, vocabulary_path,
                 resize_images=False, heldout_path=None):
        self.expression_path = str(Path(expression_path).resolve())
        self.patches_path = str(Path(patches_path).resolve())
        self.adata = sc.read_h5ad(self.expression_path)
        if not self.adata.obs_names.is_unique or not self.adata.var_names.is_unique:
            raise ValueError("Expression spot and gene identifiers must be unique")
        with open(vocabulary_path, encoding="utf-8") as handle:
            self.vocab = json.load(handle)
        if self.vocab.get("CLS") != 0:
            raise ValueError("Vocabulary must reserve CLS=0")
        missing = self.adata.var_names[~self.adata.var_names.isin(self.vocab)]
        if len(missing):
            raise ValueError(f"{len(missing)} genes are absent from vocabulary: {list(missing[:5])}")
        self.gene_id = np.asarray([self.vocab[g] for g in self.adata.var_names], dtype=np.int64)
        matrix = self.adata.X
        self.exp_mtx = np.asarray(matrix.toarray() if sparse.issparse(matrix) else matrix, dtype=np.float32)
        if self.exp_mtx.size == 0 or not np.isfinite(self.exp_mtx).all() or (self.exp_mtx < 0).any():
            raise ValueError("Expression must be nonempty, finite and nonnegative")
        if "tissue" not in self.adata.obs:
            raise ValueError("H5AD obs['tissue'] is required; run preprocessing first")
        tissue = np.asarray(self.adata.obs["tissue"], dtype=np.float64)
        if not np.isfinite(tissue).all() or (tissue < 0).any() or (tissue != np.floor(tissue)).any():
            raise ValueError("Tissue values must be nonnegative integer IDs")
        self.tissue_type = tissue.astype(np.int64)
        self.barcodes = [normalize_barcode(b) for b in self.adata.obs_names]
        if len(set(self.barcodes)) != len(self.barcodes):
            raise ValueError("Expression barcodes are duplicated after decoding/stripping")
        with h5py.File(self.patches_path, "r") as handle:
            if "img" not in handle or "barcode" not in handle:
                raise ValueError("Patch HDF5 requires img and barcode datasets")
            patch_barcodes = [normalize_barcode(b) for b in handle["barcode"][:].reshape(-1)]
            images = handle["img"]
            if images.ndim != 4 or images.shape[-1] != 3 or images.dtype != np.uint8:
                raise ValueError("Patch img dataset must have shape (N,H,W,3) and uint8 dtype")
            if len(images) != len(patch_barcodes):
                raise ValueError("Patch image count and barcode count differ")
        if len(set(patch_barcodes)) != len(patch_barcodes):
            raise ValueError("Patch barcodes must be unique after decoding/stripping")
        lookup = {barcode: i for i, barcode in enumerate(patch_barcodes)}
        missing = [b for b in self.barcodes if b not in lookup]
        if missing:
            raise ValueError(f"{len(missing)} expression spots have no image patch: {missing[:5]}")
        self.patch_index = np.asarray([lookup[b] for b in self.barcodes], dtype=np.int64)
        self.resize_images = bool(resize_images)
        self.heldout = None
        if heldout_path is not None:
            mask = sparse.load_npz(heldout_path)
            if mask.dtype != np.bool_ or mask.shape != self.exp_mtx.shape:
                raise ValueError("Held-out NPZ must be boolean True=heldout and match expression shape")
            self.heldout = mask.toarray()
            if np.any(self.exp_mtx[self.heldout] != 0):
                raise ValueError("Held-out expression entries must already be zero in the input")
        self._patch_file = None
        self._patch_pid = None

    def _images(self):
        pid = os.getpid()
        if self._patch_file is None or self._patch_pid != pid:
            self.close()
            self._patch_file = h5py.File(self.patches_path, "r")
            self._patch_pid = pid
        return self._patch_file["img"]

    def __getitem__(self, index):
        index = int(index)
        patch = self._images()[int(self.patch_index[index])]
        image = Image.fromarray(patch)
        if self.resize_images:
            image = image.resize((224, 224), Image.Resampling.BILINEAR)
        pixels = np.array(image, dtype=np.float32, copy=True) / 255.0
        tensor = torch.from_numpy(pixels.transpose(2, 0, 1).copy())
        mean = tensor.new_tensor([0.485, 0.456, 0.406])[:, None, None]
        std = tensor.new_tensor([0.229, 0.224, 0.225])[:, None, None]
        item = {
            "image": (tensor - mean) / std,
            "gene_exp": torch.from_numpy(self.exp_mtx[index].copy()),
            "gene_id": torch.from_numpy(self.gene_id.copy()),
            "tissue": torch.tensor(self.tissue_type[index], dtype=torch.long),
            "barcode": torch.tensor(index, dtype=torch.long),
        }
        if self.heldout is not None:
            item["heldout"] = torch.from_numpy(self.heldout[index].copy())
        return item

    def __len__(self):
        return self.exp_mtx.shape[0]

    def close(self):
        if getattr(self, "_patch_file", None) is not None:
            self._patch_file.close()
        self._patch_file = None
        self._patch_pid = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_patch_file"] = None
        state["_patch_pid"] = None
        return state

    def __del__(self):
        self.close()
