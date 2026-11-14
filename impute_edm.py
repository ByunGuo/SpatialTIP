import scipy as sp
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from spatialtip_edm_model import stGPTModel
import argparse
import os
from dataset_edm import stGPTData
from utils import cleanup, AvgMeter, get_lr, get_reduced, Cal_Spatial_Net, tokenize_batch
import torch.nn.functional as F
import tqdm
import numpy as np
import pandas as pd
import math
import random
from functools import reduce
import scanpy as sc

parser = argparse.ArgumentParser(description='Test model')
parser.add_argument('--batch_size', type=int, default=32, help='input batch size for training')
parser.add_argument('--num_workers', type=int, default=8, help='number of workers for data loader')
parser.add_argument('--save_dir', type=str, default='stgptmodel', help='model saving directory')
parser.add_argument('--seed', type=int, default=2024, help='random seed')
parser.add_argument('--test_sample', type=str, default='MISC2', help='test sample')
parser.add_argument('--test_mask_prob', type=float, default=1.0, help='test mask prob')
parser.add_argument('--tissue_type', type=int, default=5, help='tissue type, default for healthy') 
parser.add_argument('--vocab_size', type=int, default=20938, help='total gene number, default for healthy')
parser.add_argument('--max_seq_len', type=int, default=6001, help='gene number in one spot, plus one for [cls] token, default for healthy')
parser.add_argument('--adata_path', required=True)
parser.add_argument('--patches_path', required=True)
parser.add_argument('--vocabulary_path', default='hest_data/vocabulary_healthy_sonnet.json')
parser.add_argument('--checkpoint_path', required=True)
parser.add_argument('--output_path', required=True)
parser.add_argument('--resize_images', action='store_true')
parser.add_argument('--run_mode', choices=['smoke', 'pilot', 'formal'], default='smoke')
parser.add_argument('--max_batches', type=int, default=None)


def data_loader_fn(args, test_samples, test_mask_prob):
    
    hest_dataset = stGPTData(adata_path=args.adata_path,
                                patches_path=args.patches_path, 
                                vocabulary_path=args.vocabulary_path, resize_images=args.resize_images)
    
    test_loader = DataLoader(hest_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False)

    return test_loader


def main():
    print('Start imputing!')
    args = parser.parse_args()
    if args.run_mode == 'pilot' and args.test_sample != 'MISC1':
        raise ValueError('Current reviewed development pilot is restricted to MISC1')
    if args.run_mode == 'smoke':
        args.max_batches = min(args.max_batches or 2, 2)
        args.batch_size = min(args.batch_size, 2)
    if args.batch_size < 1 or (args.max_batches is not None and args.max_batches < 1):
        raise ValueError('batch_size and max_batches must be positive')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    test_sample = args.test_sample
    test_mask_prob = args.test_mask_prob

    print('Test sample:', test_sample)
    print('Test mask prob:', test_mask_prob)

    adata_path = args.adata_path
    adata_save_path = args.output_path

    os.makedirs(os.path.dirname(os.path.abspath(adata_save_path)), exist_ok=True)

    if not os.path.exists(adata_path):
        from pathlib import Path
        from experiments.edm.prepare import materialize
        materialize(Path(adata_path).parent, adata_path)

    test_loader = data_loader_fn(args, test_sample, test_mask_prob)

    model = stGPTModel(vision_model='vit_large_patch16_224', 
                       vocab_size=args.vocab_size, 
                       max_seq_len=args.max_seq_len, 
                       tissue_type=args.tissue_type,
                       vision_checkpoint=None,
                       use_vision=True,
                       use_tissue=True,
                       contrastive_weight=0.5).cuda(device)
                       
    model_path = args.checkpoint_path
    # model_path = os.path.join(args.save_dir, 'spavlm_hest_healthy_sonnet_truncated_model.pt')  # using the pre-trained model instead of the fine-tuned model

    print('Model Path:', model_path)
    stat_dict = torch.load(model_path, map_location='cpu')

    new_state_dict = {}
    for key in stat_dict.keys():
        new_key = key.replace('module.', '')  # remove 'module.' of DDP
        new_state_dict[new_key] = stat_dict[key]

    model.load_state_dict(new_state_dict)
    del stat_dict, new_state_dict
    print('Load model!')

    imputed_gene_expression = []
    tqdm_object = tqdm.tqdm(test_loader, total=len(test_loader))

    model.eval()
    for step, batch in enumerate(tqdm_object):
        if args.max_batches is not None and step >= args.max_batches:
            break
        batch = {k: v.cuda() for k, v in batch.items() if k == "gene_exp" or k == 'image' or k == 'barcode' or k == 'gene_id' or k == 'tissue'}
        batch['gene_id'] = tokenize_batch(batch['gene_id'], append_cls=True, cls_id=0)
        batch['labels'] = batch['gene_exp'].clone().detach()
        batch['gene_exp'][batch['gene_exp'] == 0] = -1

        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                decoded_gene_expression = model(batch, neighbors=None, mode='test')
        
        decoded_gene_expression = decoded_gene_expression.float().cpu().detach().numpy()
        imputed_gene_expression.append(decoded_gene_expression)

    imputed_gene_expression = np.concatenate(imputed_gene_expression, axis=0)

    adata = sc.read_h5ad(adata_path)
    adata = adata[:imputed_gene_expression.shape[0]].copy()
    adata.uns['edm_execution'] = {'run_mode':args.run_mode, 'step_limited':args.max_batches is not None, 'ablation':'full'}
    adata.X = imputed_gene_expression
    adata.write_h5ad(adata_save_path, compression='gzip')

    print('Inference done!')
    
if __name__ == '__main__':
     main()
