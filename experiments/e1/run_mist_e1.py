"""Run the uploaded MIST baseline on frozen E1 masks without editing MIST source."""
import argparse,datetime,json,random,sys,time
from pathlib import Path
import anndata as ad
import numpy as np,pandas as pd,scanpy as sc
from scipy.sparse import csr_matrix,load_npz
ROOT=Path(__file__).resolve().parents[2]; MIST=ROOT.parent/'baselines/MIST'; sys.path[:0]=[str(MIST),str(ROOT)]
from experiments.e1.prepare import materialize
from experiments.e1.metrics import recovery_metrics
from ReST import ReST

def dense(x): return x.toarray() if hasattr(x,'toarray') else np.asarray(x)
def main():
 p=argparse.ArgumentParser(); p.add_argument('--samples',nargs='+',default=['MISC1','MISC2','MISC5','MISC9','MISC10']); p.add_argument('--ratio',type=float,choices=[0.2,0.4,0.6],default=0.4); p.add_argument('--smoke',action='store_true'); p.add_argument('--ncores',type=int,default=10); p.add_argument('--n-experts',type=int,default=10); p.add_argument('--output-dir',type=Path); a=p.parse_args()
 random.seed(42); np.random.seed(42); mode='smoke' if a.smoke else 'formal'; out=a.output_dir or ROOT/f'experiments/e1/runs/MIST_expression_gene_42_{a.ratio}_{mode}'; out.mkdir(parents=True,exist_ok=True)
 state={'status':'running','started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'samples':a.samples,'mask':f'frozen expression_gene seed=42 ratio={a.ratio}','source':'uploaded baselines/MIST, unchanged','protocol':{'preprocess':{'species':'Human','hvg_prop':0.8,'n_pcs':10},'extract_regions':{'min_sim':0.1,'min_size':20,'gap':0.02},'impute':{'method':'MIST','ncores':a.ncores,'nExperts':a.n_experts}},'jobs':{}}
 def save(): state['updated_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat(); (out/'execution_status.json').write_text(json.dumps(state,indent=2))
 save(); rows=[]
 try:
  for sample in a.samples:
   start=time.time(); state['jobs'][sample]={'status':'running'}; save(); folder=ROOT/f'experiments/e1/healthy_expression_smoke/expression_gene/42/{sample}_{a.ratio}'; temp=out/f'.{sample}_input.h5ad'; materialize(folder,temp)
   try:
    masked=sc.read_h5ad(temp); original=sc.read_h5ad(ROOT/f'hest_data/st_healthy_opus/{sample}.h5ad'); excluded=json.loads((folder/'config.json').read_text())['excluded_missing_patch_barcodes']; truth=original[~original.obs_names.isin(excluded)].copy(); truth=truth[masked.obs_names,masked.var_names]; held=load_npz(folder/'heldout.npz').toarray().astype(bool)
    ncores,nexperts=a.ncores,a.n_experts
    # Region detection must use the complete formal gene matrix; reducing it made the
    # MIST threshold objective degenerate and was not representative of a formal run.
    rd=ReST(adata=masked); rd.preprocess(species='Human',hvg_prop=0.8,n_pcs=min(10,masked.n_obs-1,masked.n_vars-1)); rd.extract_regions(min_sim=0.1,min_size=20,gap=0.02)
    region_counts=rd.adata.obs['region_ind'].value_counts()
    state['jobs'][sample]['regions']={'threshold':float(rd.epsilon),'core_sizes':{str(k):int(v) for k,v in region_counts.items() if str(k)!='isolated'},'isolated_spots':int(region_counts.get('isolated',0))}; save()
    if a.smoke:
     core_labels=[x for x in region_counts.index if str(x)!='isolated'][:3]
     if len(core_labels)<2: raise ValueError('MIST smoke needs at least two formal-input core regions')
     selected=[]
     for label in core_labels: selected.extend(rd.adata.obs_names[rd.adata.obs['region_ind']==label][:16])
     selected.extend(rd.adata.obs_names[rd.adata.obs['region_ind']=='isolated'][:16])
     gene_idx=np.unique(np.linspace(0,masked.n_vars-1,min(128,masked.n_vars),dtype=int)); spot_idx=masked.obs_names.get_indexer(selected)
     rd.adata=rd.adata[selected,gene_idx].copy(); masked=masked[selected,gene_idx].copy(); truth=truth[selected,gene_idx].copy(); held=held[np.ix_(spot_idx,gene_idx)]; ncores=1; nexperts=2
     state['jobs'][sample]['smoke_subset']={'spots':len(selected),'genes':len(gene_idx),'core_regions':len(core_labels)}; save()
    pred_df=rd.impute(method='MIST',ncores=ncores,nExperts=nexperts,layer='CPM')
    # Native MIST labels prediction rows by array coordinates, not barcodes.
    coordinate_ids=rd.adata.obs.loc[masked.obs_names,'new_idx']
    if not coordinate_ids.is_unique: raise ValueError('MIST coordinate IDs must uniquely map to barcodes')
    pred=pred_df.loc[coordinate_ids.tolist(),masked.var_names].to_numpy()
    metrics=recovery_metrics(dense(truth.X),pred,held); elapsed=time.time()-start; rows.append({'Sample':sample,'Method':'MIST',**metrics,'elapsed_seconds':elapsed}); pd.DataFrame(rows).to_csv(out/'metrics_internal.csv',index=False)
    if not a.smoke:
     result=ad.AnnData(X=csr_matrix(pred),obs=masked.obs.copy(),var=masked.var.copy(),obsm={'spatial':np.asarray(masked.obsm['spatial']).copy()}); result.uns['e1_protocol']=state['protocol']; result.write_h5ad(out/f'MIST_{sample}_prediction.h5ad',compression='gzip')
    state['jobs'][sample].update(status='completed',elapsed_seconds=elapsed); save()
   finally: temp.unlink(missing_ok=True)
  d=pd.DataFrame(rows); d[['Sample','Method','heldout_count','RMSE','spot_CSS','gene_CSS']].rename(columns={'heldout_count':'Held-out entries','spot_CSS':'Spot-wise CSS','gene_CSS':'Gene-wise CSS'}).to_csv(out/'imputation_metrics_manuscript.csv',index=False); state.update(status='completed',ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()); save(); print(d.to_string(index=False))
 except Exception as e: state.update(status='failed',error=f'{type(e).__name__}: {e}'); save(); raise
if __name__=='__main__': main()
