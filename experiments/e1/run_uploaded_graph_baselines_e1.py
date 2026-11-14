"""Run uploaded GraphST, STAGATE, and SEDR code on frozen E1 masks.

The uploaded baseline sources are imported without modification. This adapter
materializes the common masked input, follows each uploaded evaluation protocol,
and evaluates the shared held-out set.
"""
import argparse, datetime, json, random, sys, time
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from scipy.sparse import csr_matrix, load_npz
ROOT=Path(__file__).resolve().parents[2]; BASELINES=ROOT.parent/'baselines'
sys.path.insert(0,str(ROOT))
from experiments.e1.metrics import recovery_metrics
from experiments.e1.prepare import materialize

def dense(x): return x.toarray() if hasattr(x,'toarray') else np.asarray(x)
def seed_all(seed=42):
 random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
 torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False

def graphst(a,device,smoke,epochs):
 sys.path.insert(0,str(BASELINES/'GraphST')); from GraphST.GraphST import GraphST
 n=epochs if epochs is not None else (2 if smoke else 600)
 result=GraphST(a,device=device,epochs=n).train()
 return np.asarray(result.obsm['emb']),{'uploaded_entrypoint':'GraphST/evaluate.py','epochs':n,'model_random_seed':2024,'parameters':'GraphST defaults'}

def stagate(a,device,smoke,epochs):
 sys.path.insert(0,str(BASELINES/'STAGATE')); from utils import Cal_Spatial_Net; from Train_STAGATE import train_STAGATE
 Cal_Spatial_Net(a,rad_cutoff=150,model='Radius',verbose=False); n=epochs if epochs is not None else (2 if smoke else 1000)
 result=train_STAGATE(a,n_epochs=n,save_reconstrction=True,device=device,verbose=False)
 return np.asarray(result.layers['STAGATE_ReX']),{'uploaded_entrypoint':'STAGATE/evaluate.py','rad_cutoff':150,'graph_model':'Radius','epochs':n,'model_random_seed':2024,'parameters':'train_STAGATE defaults'}

def sedr(a,device,smoke,epochs):
 sys.path.insert(0,str(BASELINES/'SEDR')); from SEDR.graph_func import graph_construction; from SEDR.SEDR_model import Sedr
 model=Sedr(dense(a.X),graph_construction(a,12),mode='imputation',device=device)
 if smoke or epochs is not None:
  n=epochs if epochs is not None else 2; model.train_without_dec(epochs=n); training='train_without_dec smoke/explicit override'
 else:
  n='200 initial + up to 200 DEC'; model.train_with_dec(); training='train_with_dec defaults'
 return np.asarray(model.recon()),{'uploaded_entrypoint':'SEDR/evaluate.py','neighbors':12,'mode':'imputation','training':training,'epochs':n,'parameters':'Sedr defaults'}
RUNNERS={'GraphST':graphst,'STAGATE':stagate,'SEDR':sedr}

def main():
 p=argparse.ArgumentParser(); p.add_argument('--methods',nargs='+',choices=list(RUNNERS),default=list(RUNNERS)); p.add_argument('--samples',nargs='+',default=['MISC1','MISC2','MISC5','MISC9','MISC10']); p.add_argument('--ratio',type=float,choices=[0.2,0.4,0.6],default=0.4); p.add_argument('--device',default='cuda:0' if torch.cuda.is_available() else 'cpu'); p.add_argument('--smoke',action='store_true'); p.add_argument('--epochs',type=int); p.add_argument('--smoke-spots',type=int,default=128); p.add_argument('--smoke-genes',type=int,default=256); p.add_argument('--output-dir',type=Path); args=p.parse_args()
 device=torch.device(args.device); mode='smoke' if args.smoke else 'formal'; out=args.output_dir or (ROOT/'experiments/e1/runs'/f'uploaded_graph_baselines_expression_gene_42_{args.ratio}_{mode}'); out.mkdir(parents=True,exist_ok=True)
 status={'status':'running','started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'methods':args.methods,'samples':args.samples,'device':str(device),'mask':f'frozen expression_gene seed=42 ratio={args.ratio}; True=held-out observed nonzero','source_policy':'uploaded sources unchanged; adapter follows uploaded evaluate.py','jobs':{}}
 def save(): status['updated_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat(); (out/'execution_status.json').write_text(json.dumps(status,indent=2))
 save(); rows=[]
 try:
  for method in args.methods:
   for sample in args.samples:
    key=f'{method}:{sample}'; start=time.time(); status['jobs'][key]={'status':'running'}; save(); seed_all()
    folder=ROOT/f'experiments/e1/healthy_expression_smoke/expression_gene/42/{sample}_{args.ratio}'; temp=out/f'.{method}_{sample}_input.h5ad'; materialize(folder,temp)
    try:
     masked=sc.read_h5ad(temp); original=sc.read_h5ad(ROOT/f'hest_data/st_healthy_opus/{sample}.h5ad'); excluded=json.loads((folder/'config.json').read_text())['excluded_missing_patch_barcodes']; truth=original[~original.obs_names.isin(excluded)].copy(); truth=truth[masked.obs_names,masked.var_names]; held=load_npz(folder/'heldout.npz').toarray().astype(bool)
     if args.smoke: masked=masked[:args.smoke_spots,:args.smoke_genes].copy(); truth=truth[masked.obs_names,masked.var_names].copy(); held=held[:masked.n_obs,:masked.n_vars]
     if np.any(dense(masked.X)[held]!=0) or np.any(dense(truth.X)[held]<=0): raise ValueError('frozen mask alignment failed')
     pred,protocol=RUNNERS[method](masked,device,args.smoke,args.epochs)
     if pred.shape!=masked.shape or not np.isfinite(pred).all(): raise ValueError(f'invalid {method} prediction {pred.shape}')
     metrics=recovery_metrics(dense(truth.X),pred,held); elapsed=time.time()-start; rows.append({'Sample':sample,'Method':method,**metrics,'elapsed_seconds':elapsed})
     if not args.smoke:
      result=ad.AnnData(X=csr_matrix(pred),obs=masked.obs.copy(),var=masked.var.copy(),obsm={'spatial':np.asarray(masked.obsm['spatial']).copy()}); result.uns['e1_protocol']=protocol; result.write_h5ad(out/f'{method}_{sample}_prediction.h5ad',compression='gzip')
     pd.DataFrame(rows).to_csv(out/'metrics_internal.csv',index=False); status['jobs'][key]={'status':'completed','elapsed_seconds':elapsed,'protocol':protocol}; save()
    finally: temp.unlink(missing_ok=True)
  d=pd.DataFrame(rows); formal=d[['Sample','Method','heldout_count','RMSE','spot_CSS','gene_CSS']].rename(columns={'heldout_count':'Held-out entries','spot_CSS':'Spot-wise CSS','gene_CSS':'Gene-wise CSS'}); formal.to_csv(out/'imputation_metrics_manuscript.csv',index=False); status.update(status='completed',ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()); save(); print(formal.to_string(index=False))
 except Exception as e: status.update(status='failed',error=f'{type(e).__name__}: {e}'); save(); raise
if __name__=='__main__': main()
