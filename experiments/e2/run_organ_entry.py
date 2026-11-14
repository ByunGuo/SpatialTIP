"""Existing E1 entrypoints with an explicit unknown-organ tissue-vector policy."""
import sys,runpy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import torch
from stGPT_con_e1 import TissueEncoder
original=TissueEncoder.forward
def no_condition_unknown(self,x):
    if (x < -1).any() or (x >= self.embedding.num_embeddings).any():raise ValueError('Invalid tissue code')
    known=original(self,x.clamp_min(0))
    unknown=torch.zeros_like(known)
    return torch.where((x==-1).unsqueeze(-1),unknown,known)
TissueEncoder.forward=no_condition_unknown
if __name__=='__main__':
    mode=sys.argv.pop(1)
    if mode not in ['train','infer']:raise ValueError(mode)
    runpy.run_path(str(ROOT/('finetune_evaluate_e1.py' if mode=='train' else 'impute_evaluate_e1.py')),run_name='__main__')
