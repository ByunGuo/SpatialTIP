import os
import scanpy as sc
import json
import torch.utils
os.environ['OPENCV_IO_MAX_IMAGE_PIXELS'] = pow(2, 40).__str__()
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import h5py


class stGPTData(torch.utils.data.Dataset):
    def __init__(self, adata_path, patches_path, vocabulary_path, heldout_path=None, resize_images=False):
        
        self.adata = sc.read_h5ad(adata_path)

        self.patches_h5 = h5py.File(patches_path, 'r')

        self.vocab = json.load(open(vocabulary_path, 'r'))

        self.transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        )

        if resize_images:
            self.transform = transforms.Compose([transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR), self.transform])

        self.heldout = None
        if heldout_path:
            from scipy.sparse import load_npz
            self.heldout = load_npz(heldout_path).toarray()
            if self.heldout.dtype != bool:
                raise ValueError('E1 heldout must be boolean True=heldout, not legacy keep-mask')
            if self.heldout.shape != self.adata.shape:
                raise ValueError('heldout shape mismatch')
        barcodes = self.patches_h5['barcode'][:].reshape(-1)
        barcodes = [b.decode() if isinstance(b, bytes) else str(b) for b in barcodes]
        if len(barcodes) != len(set(barcodes)):
            raise ValueError('Duplicate patch barcodes')
        self.patch_index = {b:i for i,b in enumerate(barcodes)}
        self.index = self.adata.obs.index
        if any(b not in self.patch_index for b in self.index):
            raise ValueError('Align expression and patch barcodes first')
        self.exp_mtx = self.adata.X.toarray()
        if self.heldout is not None and (self.exp_mtx[self.heldout] != 0).any():
            raise ValueError('Prepared holdout entries must be hidden in input')
        self.gene_id = np.array([self.vocab[gene] for gene in self.adata.var_names])
        self.tissue_type = self.adata.obs['tissue'].values

        self.id_spot_tran = dict(zip(range(self.exp_mtx.shape[0]), np.array(self.index)))
        self.id_spot_tran = {v: k for k, v in self.id_spot_tran.items()}
       
    def __getitem__(self, idx):
        item = {}
        barcode = self.index[idx]
        index = self.id_spot_tran[barcode]
        
        # 从h5文件中根据barcode读取patches
        patches = self.patches_h5['img'][self.patch_index[barcode]]
        image = Image.fromarray(patches)
        image = self.transform(image)

        item['barcode'] = torch.tensor(index).float()
        item['image'] = image.clone().detach().float()
        item['gene_exp'] = torch.tensor(self.exp_mtx[idx, :]).float()
        item['gene_id'] = torch.tensor(self.gene_id).long()
        item['tissue'] = torch.tensor(self.tissue_type[idx]).long()

        if self.heldout is not None:
            item['heldout'] = torch.tensor(self.heldout[idx])
        return item

    def __len__(self):
        return self.exp_mtx.shape[0]
    
        
