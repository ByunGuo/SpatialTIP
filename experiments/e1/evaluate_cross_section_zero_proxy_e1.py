"""Leave-one-section-out DLPFC zero proxies and SpatialTIP false-positive analysis.

Labels are operational proxies, never biological ground truth. They are built and
saved before predictions are loaded. The target section is excluded from all
cross-section evidence.
"""
import os
import datetime, hashlib, json
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, load_npz, save_npz
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from spatial_smoothing import spatial_smoothing
from experiments.e1.prepare import digest, ensure_annotations

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'experiments/e1/runs/cross_section_zero_proxy_v1'
SAMPLES=['MISC1','MISC2','MISC5','MISC9','MISC10']
MAP={'MISC1':'151676','MISC2':'151675','MISC3':'151674','MISC5':'151672','MISC6':'151671','MISC8':'151669','MISC9':'151510','MISC10':'151509','MISC12':'151507'}
FAMILIES=[['MISC1','MISC2','MISC3'],['MISC5','MISC6','MISC8'],['MISC9','MISC10','MISC12']]
REFS={s:[x for family in FAMILIES if s in family for x in family if x!=s] for s in MAP}
NEG_LIMITS={'primary_5pct':.05,'strict_2pct':.02,'failed_1pct':.01}
POS_LIMIT=.20; ELSEWHERE_LIMIT=.05; RADIUS=150
FIXED=[0,.01,.05,.1,.25,.5,.75,1,1.5,2]

def dense(x): return x.toarray() if hasattr(x,'toarray') else np.asarray(x)
def norm_layer(x): return x.astype(str).str.replace('L','',regex=False).replace({'2/3':'3'}).to_numpy()
def sha_array(h,*arrays):
 for x in arrays:h.update(np.ascontiguousarray(x).view(np.uint8))

def paths(sample):
 folder=ROOT/f'experiments/e1/healthy_expression_smoke/expression_gene/42/{sample}_0.4'
 run=ROOT/('experiments/e1/pilot_MISC1/formal_expression_gene_42_0.4' if sample=='MISC1' else f'experiments/e1/runs/{sample}_expression_gene_42_0.4_formal')
 manifest=os.environ.get("SPATIALTIP_PREDICTION_MANIFEST")
 if manifest:run=Path(json.loads(Path(manifest).read_text())["SpatialTIP"][sample]).parent
 return folder,run

def build_labels(sample,layers):
 folder,_=paths(sample);cfg=json.loads((folder/'config.json').read_text())
 target=ad.read_h5ad(ROOT/f'hest_data/st_healthy_opus/{sample}.h5ad');target=target[~target.obs_names.isin(cfg['excluded_missing_patch_barcodes'])].copy();x=dense(target.X);zero=x==0
 target_layer=norm_layer(layers[layers.sid==MAP[sample]].set_index('barcode').layer.reindex(target.obs_names))
 annotated=pd.notna(target_layer)&(target_layer!='nan')
 neighbors=NearestNeighbors(radius=RADIUS).fit(target.obsm['spatial']).radius_neighbors(target.obsm['spatial'],return_distance=False)
 local=np.vstack([(x[ix]>0).mean(0) for ix in neighbors])
 rates=[]; elsewhere=[]; ref_audit={}
 for ref in REFS[sample]:
  a=ad.read_h5ad(ROOT/f'hest_data/st/{ref}.h5ad');ref_layer=norm_layer(layers[layers.sid==MAP[ref]].set_index('barcode').layer.reindex(a.obs_names));valid=pd.notna(ref_layer)&(ref_layer!='nan')
  idx=a.var_names.get_indexer(target.var_names)
  if (idx<0).any():raise ValueError(f'{ref}: missing target genes')
  xr=a.X[:,idx]
  per={};other={}
  for layer in sorted(set(target_layer[annotated])):
   w=valid&(ref_layer==layer);per[layer]=np.asarray((xr[w]>0).mean(0)).ravel() if w.sum() else np.full(target.n_vars,np.nan)
   w=valid&(ref_layer!=layer);other[layer]=(np.asarray((xr[w]>0).sum(0)).ravel(),int(w.sum()))
  rates.append(per);elsewhere.append(other);ref_audit[ref]={'section_id':MAP[ref],'spots':a.n_obs,'annotated_spots':int(valid.sum()),'raw_counts_sha256':digest(ROOT/f'hest_data/st/{ref}.h5ad')}
 positives=np.zeros(zero.shape,bool);negatives={name:np.zeros(zero.shape,bool) for name in NEG_LIMITS}
 for layer in sorted(set(target_layer[annotated])):
  w=target_layer==layer;r1,r2=rates[0][layer],rates[1][layer]
  elsewhere_rate=(elsewhere[0][layer][0]+elsewhere[1][layer][0])/(elsewhere[0][layer][1]+elsewhere[1][layer][1])
  positives[w]=zero[w]&(local[w]>=POS_LIMIT)&(r1>=POS_LIMIT)&(r2>=POS_LIMIT)
  for name,limit in NEG_LIMITS.items():
   negatives[name][w]=zero[w]&(local[w]==0)&(r1<=limit)&(r2<=limit)&(elsewhere_rate>=ELSEWHERE_LIMIT)
 for name,n in negatives.items():
  if (n&positives).any():raise ValueError('Proxy labels overlap')
 OUT.mkdir(parents=True,exist_ok=True);save_npz(OUT/f'{sample}_positive_dropout_proxy.npz',csr_matrix(positives))
 for name,n in negatives.items():save_npz(OUT/f'{sample}_negative_{name}.npz',csr_matrix(n))
 h=hashlib.sha256();sha_array(h,positives,*[negatives[x] for x in NEG_LIMITS])
 audit={'sample':sample,'section_id':MAP[sample],'references':ref_audit,'shape':list(zero.shape),'annotated_spots':int(annotated.sum()),'observed_zero_count_annotated':int(zero[annotated].sum()),'positive_dropout_proxy_count':int(positives.sum()),'positive_genes':int(positives.any(0).sum()),'negative_counts':{k:int(v.sum()) for k,v in negatives.items()},'negative_genes':{k:int(v.any(0).sum()) for k,v in negatives.items()},'label_sha256':h.hexdigest(),'target_truth_sha256':cfg['input_sha256']}
 return target,positives,negatives,audit

def metrics(sample,source,rule,method,truth,scores,pos,neg):
 selected=pos|neg;t=pos[selected];q=scores[selected];active=q>0;tp=int((active&t).sum());fp=int((active&~t).sum());p=int(t.sum());n=int((~t).sum())
 return {'Sample':sample,'Source':source,'Proxy rule':rule,'Postprocessing':method,'Positive proxies':p,'Negative proxies':n,'TP':tp,'FP':fp,'FN':p-tp,'TN':n-fp,'proxy_FPR':fp/n if n else np.nan,'proxy_recall':tp/p if p else np.nan,'proxy_precision':tp/(tp+fp) if tp+fp else np.nan,'proxy_F1':2*tp/(p+tp+fp) if p else np.nan,'proxy_AUROC':roc_auc_score(t,q) if p and n and method=='raw' else np.nan,'proxy_AUPRC':average_precision_score(t,q) if p and n and method=='raw' else np.nan,'negative_mean_prediction':float(scores[neg].mean()) if n else np.nan,'negative_median_prediction':float(np.median(scores[neg])) if n else np.nan,'positive_mean_prediction':float(scores[pos].mean()) if p else np.nan,'positive_median_prediction':float(np.median(scores[pos])) if p else np.nan}

def main():
 annotation=ensure_annotations()
 layers=pd.read_csv(annotation,sep='\t',header=None,names=['barcode','sid','layer'],dtype=str)
 labels={sample:build_labels(sample,layers) for sample in SAMPLES}
 rows=[]
 for sample,(target,pos,negs,audit) in labels.items():
  folder,run=paths(sample);z=dense(target.X);pred=ad.read_h5ad(run/'prediction.h5ad')
  if not pred.obs_names.equals(target.obs_names) or not pred.var_names.equals(target.var_names):raise ValueError('Prediction alignment mismatch')
  y=dense(pred.X);held=load_npz(folder/'heldout.npz').toarray();completed=np.where((z==0)|held,y,z)
  smoothed=spatial_smoothing(np.where(held,0,z),y,target.obsm['spatial'],radius=RADIUS)
  for source,q in [('without_spatial_smoothing',y),('SpatialTIP',smoothed)]:
   for rule in ['primary_5pct','strict_2pct']:rows.append(metrics(sample,source,rule,'raw',z,q,pos,negs[rule]))
 pd.DataFrame(rows).to_csv(OUT/'proxy_metrics_by_slice.csv',index=False)
if __name__=='__main__':main()
