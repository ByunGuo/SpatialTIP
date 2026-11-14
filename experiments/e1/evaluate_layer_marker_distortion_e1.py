"""External same-patient section validation of DLPFC layer-marker distortion."""
import os
import datetime,json
from pathlib import Path
import anndata as ad
import numpy as np,pandas as pd
from sklearn.metrics import average_precision_score,roc_auc_score
from sklearn.neighbors import NearestNeighbors
from spatial_smoothing import spatial_smoothing
from scipy.sparse import load_npz
from experiments.e1.prepare import digest, ensure_annotations
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'experiments/e1/runs/layer_marker_distortion_v1';OUT.mkdir(parents=True,exist_ok=True)
SAMPLES=['MISC1','MISC2','MISC5','MISC9','MISC10'];MAP={'MISC1':'151676','MISC2':'151675','MISC3':'151674','MISC5':'151672','MISC6':'151671','MISC8':'151669','MISC9':'151510','MISC10':'151509','MISC12':'151507'};FAMS=[['MISC1','MISC2','MISC3'],['MISC5','MISC6','MISC8'],['MISC9','MISC10','MISC12']];REFS={s:[x for f in FAMS if s in f for x in f if x!=s] for s in MAP}
def dense(x):return x.toarray() if hasattr(x,'toarray') else np.asarray(x)
def layer_values(frame,sid,index):return frame[frame.sid==sid].set_index('barcode').layer.reindex(index).astype(str).str.replace('L','',regex=False).replace({'2/3':'3'}).to_numpy()
def runpath(s):
 manifest=os.environ.get('SPATIALTIP_PREDICTION_MANIFEST')
 if manifest:return Path(json.loads(Path(manifest).read_text())['SpatialTIP'][s]).parent
 return ROOT/('experiments/e1/pilot_MISC1/formal_expression_gene_42_0.4' if s=='MISC1' else f'experiments/e1/runs/{s}_expression_gene_42_0.4_formal')
annotation=ensure_annotations();layers=pd.read_csv(annotation,sep='\t',header=None,names=['barcode','sid','layer'],dtype=str)
config={'status':'building_external_labels','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'labels':'same-patient two-reference-section agreement; target excluded','positive':'both references: layer detection>=.20 and normalized mean layer-minus-other >=.25','negative':'both references: layer detection<=.05, pooled other-layer detection>=.05, and layer-minus-other<=0','normalization':'raw counts normalize_total(1e4), log1p','target_score':'mean normalized expression in layer minus other annotated layers','prediction_use':'replace only original observed zeros; restore all target observed nonzeros, including synthetic heldouts','threshold_sensitivity':.5,'samples':{}}
(OUT/'evaluation_config.json').write_text(json.dumps(config,indent=2))
labels_by_sample={}
for s in SAMPLES:
 cfg=json.loads((ROOT/f'experiments/e1/healthy_expression_smoke/expression_gene/42/{s}_0.4/config.json').read_text());a=ad.read_h5ad(ROOT/f'hest_data/st_healthy_opus/{s}.h5ad');a=a[~a.obs_names.isin(cfg['excluded_missing_patch_barcodes'])].copy();tl=layer_values(layers,MAP[s],a.obs_names);valid=tl!='nan';refs=[]
 for r in REFS[s]:
  q=ad.read_h5ad(ROOT/f'hest_data/st/{r}.h5ad');rl=layer_values(layers,MAP[r],q.obs_names);rv=rl!='nan';idx=q.var_names.get_indexer(a.var_names);x=dense(q.X[:,idx]).astype(np.float64);x=np.log1p(x/np.maximum(x.sum(1,keepdims=True),1)*1e4);refs.append((x,rl,rv,dense(q.X[:,idx])>0))
 labelrows=[]
 for layer in sorted(set(tl[valid])):
  rates=[];effects=[];other_detect=[]
  for x,rl,rv,det in refs:
   inside=rv&(rl==layer);outside=rv&(rl!=layer);rates.append(det[inside].mean(0));other_detect.append(det[outside].sum(0));effects.append(x[inside].mean(0)-x[outside].mean(0))
  pooled_other=(other_detect[0]+other_detect[1])/sum(int((rv&(rl!=layer)).sum()) for _,rl,rv,_ in refs)
  positive=(rates[0]>=.2)&(rates[1]>=.2)&(effects[0]>=.25)&(effects[1]>=.25)
  negative=(rates[0]<=.05)&(rates[1]<=.05)&(pooled_other>=.05)&(effects[0]<=0)&(effects[1]<=0)
  for j in np.flatnonzero(positive|negative):labelrows.append({'Sample':s,'Layer':layer,'Gene':a.var_names[j],'Gene index':j,'Label':int(positive[j]),'Reference effect 1':effects[0][j],'Reference effect 2':effects[1][j],'Reference layer rate 1':rates[0][j],'Reference layer rate 2':rates[1][j],'Pooled other-layer rate':pooled_other[j]})
 lab=pd.DataFrame(labelrows);lab.to_csv(OUT/f'{s}_external_marker_labels.csv',index=False);labels_by_sample[s]=(a,tl,valid,lab);config['samples'][s]={'section_id':MAP[s],'references':[MAP[x] for x in REFS[s]],'positive_gene_layers':int((lab.Label==1).sum()),'negative_gene_layers':int((lab.Label==0).sum()),'layers':sorted(set(tl[valid])),'truth_sha256':cfg['input_sha256']}
config['status']='labels_frozen_before_prediction_evaluation';config['labels_frozen_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();(OUT/'evaluation_config.json').write_text(json.dumps(config,indent=2))
rows=[];toprows=[]
for s,(a,tl,valid,lab) in labels_by_sample.items():
 z=dense(a.X);pred=ad.read_h5ad(runpath(s)/'prediction.h5ad');y=dense(pred.X);held=load_npz(ROOT/f'experiments/e1/healthy_expression_smoke/expression_gene/42/{s}_0.4/heldout.npz').toarray();masked=np.where(held,0,z);completed=np.where(masked==0,y,masked);smoothed=spatial_smoothing(masked,y,a.obsm['spatial'],radius=150);
 # Isolate original-zero effects: exact original nonzeros are restored.
 outputs={'Observed input':z,'SpatialTIP':np.where(z==0,smoothed,z)}
 for name,x in outputs.items():
  for layer,group in lab.groupby('Layer'):
   inside=valid&(tl==layer);outside=valid&(tl!=layer);effect=x[inside].mean(0)-x[outside].mean(0);idx=group['Gene index'].to_numpy(int);truth=group.Label.to_numpy(int);score=effect[idx]
   rows.append({'Sample':s,'Method':name,'Layer':layer,'Positive gene-layers':int(truth.sum()),'Negative gene-layers':int((truth==0).sum()),'AUROC':roc_auc_score(truth,score) if len(set(truth))==2 else np.nan,'AUPRC':average_precision_score(truth,score) if truth.sum() else np.nan,'Positive mean effect':float(score[truth==1].mean()) if truth.sum() else np.nan,'Negative mean effect':float(score[truth==0].mean()) if (truth==0).sum() else np.nan,'Negative false-enrichment rate (>0.25)':float((score[truth==0]>.25).mean()) if (truth==0).sum() else np.nan})
   order=np.argsort(-score);k=min(50,len(order));top=truth[order[:k]];toprows.append({'Sample':s,'Method':name,'Layer':layer,'K':k,'External-positive precision@K':float(top.mean()),'External-positive recall@K':float(top.sum()/truth.sum()) if truth.sum() else np.nan,'External-negative count@K':int((top==0).sum())})
pd.DataFrame(rows).to_csv(OUT/'layer_marker_metrics.csv',index=False);pd.DataFrame(toprows).to_csv(OUT/'layer_marker_top50.csv',index=False)
d=pd.DataFrame(rows);macro=d.groupby('Method',as_index=False)[['AUROC','AUPRC','Positive mean effect','Negative mean effect','Negative false-enrichment rate (>0.25)']].mean();macro.to_csv(OUT/'layer_marker_metrics_macro.csv',index=False)
config['status']='completed';config['completed_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();config['limitations']=['External labels are same-patient adjacent-section proxies, not molecular ground truth.','Layer markers do not resolve cell-type mixtures within Visium spots.','Restoring target observed nonzeros isolates effects of imputing original zeros and is not the historical whole-matrix smoothing output.'];(OUT/'evaluation_config.json').write_text(json.dumps(config,indent=2));print(macro.to_string(index=False))
