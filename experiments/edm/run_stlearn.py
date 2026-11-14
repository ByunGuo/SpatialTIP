"""Run uploaded stLearn SME on frozen EDM masks using HEST's aligned spot patches."""
import argparse,datetime,json,sys,time
from pathlib import Path
import anndata as ad
import h5py,numpy as np,pandas as pd,scanpy as sc
from scipy.sparse import csr_matrix,load_npz
from sklearn.decomposition import PCA
ROOT=Path(__file__).resolve().parents[2]; STLEARN=ROOT.parent/'baselines/stlearn'; sys.path[:0]=[str(STLEARN),str(ROOT)]
from experiments.edm.prepare import materialize
from experiments.edm.metrics import recovery_metrics

def dense(x): return x.toarray() if hasattr(x,'toarray') else np.asarray(x)
def morphology(adata,sample,batch_size):
 from stlearn.image_preprocessing.model_zoo import Model
 patch=ROOT/f'hest_data/st/patches/{sample}.h5'; model=Model('resnet50',batch_size=batch_size); feats=[]
 with h5py.File(patch,'r') as h:
  bars=[x[0].decode() if isinstance(x[0],bytes) else str(x[0]) for x in h['barcode'][:]]; pos={b:i for i,b in enumerate(bars)}; idx=np.array([pos[b] for b in adata.obs_names])
  for start in range(0,len(idx),batch_size): feats.append(model.predict(h['img'][idx[start:start+batch_size]]))
 x=np.concatenate(feats,axis=0); return PCA(n_components=min(50,len(adata)-1,x.shape[1]),random_state=1).fit_transform(x)
def main():
 p=argparse.ArgumentParser(); p.add_argument('--samples',nargs='+',default=['MISC1','MISC2','MISC5','MISC9','MISC10']); p.add_argument('--ratio',type=float,choices=[0.2,0.4,0.6],default=0.4); p.add_argument('--smoke',action='store_true'); p.add_argument('--batch-size',type=int,default=32); p.add_argument('--output-dir',type=Path); a=p.parse_args(); mode='smoke' if a.smoke else 'formal'; out=a.output_dir or ROOT/f'experiments/edm/runs/stLearn_expression_gene_42_{a.ratio}_{mode}'; out.mkdir(parents=True,exist_ok=True)
 state={'status':'running','started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'samples':a.samples,'mask':f'frozen expression_gene seed=42 ratio={a.ratio}','source':'uploaded baselines/stlearn unchanged','protocol':{'image':'HEST aligned 224x224 spot patches','morphology':'uploaded TensorFlow ImageNet ResNet50 + PCA(50, random_state=1)','expression':'already normalized EDM masked matrix; PCA(50) computed after masking','SME':'SME_impute0 defaults, weights_matrix_all, Visium'},'jobs':{}}
 def save(): state['updated_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat(); (out/'execution_status.json').write_text(json.dumps(state,indent=2))
 save(); rows=[]
 try:
  from stlearn.spatials.SME.impute import SME_impute0
  for sample in a.samples:
   start=time.time(); state['jobs'][sample]={'status':'running'}; save(); folder=ROOT/f'experiments/edm/healthy_expression_smoke/expression_gene/42/{sample}_{a.ratio}'; temp=out/f'.{sample}_input.h5ad'; materialize(folder,temp)
   try:
    masked=sc.read_h5ad(temp); original=sc.read_h5ad(ROOT/f'hest_data/st_healthy_opus/{sample}.h5ad'); excluded=json.loads((folder/'config.json').read_text())['excluded_missing_patch_barcodes']; truth=original[~original.obs_names.isin(excluded)].copy(); truth=truth[masked.obs_names,masked.var_names]; held=load_npz(folder/'heldout.npz').toarray().astype(bool)
    if a.smoke: masked=masked[:128,:256].copy(); truth=truth[masked.obs_names,masked.var_names].copy(); held=held[:masked.n_obs,:masked.n_vars]
    masked.obs['imagecol']=masked.obs['pxl_col_in_fullres'].astype(float); masked.obs['imagerow']=masked.obs['pxl_row_in_fullres'].astype(float); sc.pp.pca(masked,n_comps=min(50,masked.n_obs-1,masked.n_vars-1),random_state=1); masked.obsm['X_morphology']=morphology(masked,sample,a.batch_size); SME_impute0(masked); pred=np.asarray(masked.obsm['raw_SME_imputed'])
    metrics=recovery_metrics(dense(truth.X),pred,held); elapsed=time.time()-start; rows.append({'Sample':sample,'Method':'stLearn',**metrics,'elapsed_seconds':elapsed}); pd.DataFrame(rows).to_csv(out/'metrics_internal.csv',index=False)
    if not a.smoke:
     result=ad.AnnData(X=csr_matrix(pred),obs=masked.obs.copy(),var=masked.var.copy(),obsm={'spatial':np.asarray(masked.obsm['spatial']).copy()}); result.uns['edm_protocol']=state['protocol']; result.write_h5ad(out/f'stLearn_{sample}_prediction.h5ad',compression='gzip')
    state['jobs'][sample]={'status':'completed','elapsed_seconds':elapsed}; save()
   finally: temp.unlink(missing_ok=True)
  d=pd.DataFrame(rows); d[['Sample','Method','heldout_count','RMSE','spot_CSS','gene_CSS']].rename(columns={'heldout_count':'Held-out entries','spot_CSS':'Spot-wise CSS','gene_CSS':'Gene-wise CSS'}).to_csv(out/'imputation_metrics_manuscript.csv',index=False); state.update(status='completed',ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()); save(); print(d.to_string(index=False))
 except Exception as e: state.update(status='failed',error=f'{type(e).__name__}: {e}'); save(); raise
if __name__=='__main__': main()
