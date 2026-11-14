"""MAGIC baseline on frozen E1 inputs, following manuscript Scanpy-default protocol."""
import argparse,datetime,hashlib,json,time
from pathlib import Path
import numpy as np,pandas as pd,scanpy as sc
import scanpy.external as sce
from scipy.sparse import load_npz
from experiments.e1.prepare import materialize,digest
from experiments.e1.metrics import recovery_metrics
ROOT=Path(__file__).resolve().parents[2]
def dense(x):return x.toarray() if hasattr(x,'toarray') else np.asarray(x)
def main():
 p=argparse.ArgumentParser();p.add_argument('--samples',nargs='+',default=['MISC1','MISC2','MISC5','MISC9','MISC10']);p.add_argument('--ratio',type=float,choices=[0.2,0.4,0.6],default=0.4);p.add_argument('--smoke',action='store_true');p.add_argument('--n-jobs',type=int,default=8);p.add_argument('--output-dir',type=Path);a=p.parse_args()
 out=a.output_dir or ROOT/f'experiments/e1/runs/healthy_magic_expression_gene_42_{a.ratio}';out.mkdir(parents=True,exist_ok=True);rows=[];state={'status':'running','samples':a.samples,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'protocol':{'implementation':'scanpy.external.pp.magic','scanpy':sc.__version__,'magic':'3.0.0','parameters':{'knn':5,'decay':1,'knn_max':None,'t':3,'n_pca':100,'solver':'exact','knn_dist':'euclidean','random_state':42},'input':f'same frozen expression-dependent masked input used by SpatialTIP; ratio={a.ratio}','postprocessing':'none'}}
 def save():
  state['updated_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();(out/'execution_status.json').write_text(json.dumps(state,indent=2))
 save()
 try:
  for sample in a.samples:
   start=time.time();folder=ROOT/f'experiments/e1/healthy_expression_smoke/expression_gene/42/{sample}_{a.ratio}';inp=materialize(folder);ad=sc.read_h5ad(inp);truth=sc.read_h5ad(ROOT/f'hest_data/st_healthy_opus/{sample}.h5ad')[ad.obs_names,ad.var_names];held=load_npz(folder/'heldout.npz').toarray();assert held.dtype==bool and not (dense(ad.X)[held]!=0).any()
   if a.smoke:ad=ad[:128,:256].copy();truth=truth[:128,:256].copy();held=held[:128,:256]
   result=sce.pp.magic(ad,knn=5,decay=1,knn_max=None,t=3,n_pca=min(100,ad.n_obs-1,ad.n_vars-1),solver='exact',knn_dist='euclidean',random_state=42,n_jobs=a.n_jobs,verbose=True,copy=True)
   y=dense(result.X);z=dense(truth.X);assert y.shape==z.shape and np.isfinite(y).all();metric=recovery_metrics(z,y,held);record={'sample':sample,'Method':'MAGIC',**metric,'elapsed_seconds':time.time()-start};rows.append(record)
   if not a.smoke:
    result.obs=ad.obs.copy();result.var=ad.var.copy();result.obsm=ad.obsm.copy();result.write_h5ad(out/f'{sample}_prediction.h5ad',compression='gzip')
   pd.DataFrame(rows).to_csv(out/('smoke_metrics.csv' if a.smoke else 'metrics_internal.csv'),index=False);state.setdefault('jobs',{})[sample]={'status':'completed','elapsed_seconds':record['elapsed_seconds']};save();inp.unlink(missing_ok=True)
  if not a.smoke:
   d=pd.DataFrame(rows);formal=d[['sample','Method','heldout_count','RMSE','spot_CSS','gene_CSS']].rename(columns={'sample':'Sample','heldout_count':'Held-out entries','spot_CSS':'Spot-wise CSS','gene_CSS':'Gene-wise CSS'});formal.to_csv(out/'imputation_metrics_manuscript.csv',index=False)
  state.update(status='completed',ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());save();print(pd.DataFrame(rows).to_string(index=False))
 except Exception as e:
  state.update(status='failed',error=f'{type(e).__name__}: {e}');save();raise
if __name__=='__main__':main()
