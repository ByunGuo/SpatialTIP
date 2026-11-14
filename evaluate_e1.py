import scanpy as sc
import os
from utils import Cal_Spatial_Net
import argparse
import numpy as np
import pandas as pd
import scipy as sp

parser = argparse.ArgumentParser(description='Test model')
parser.add_argument('--sample', type=str, default='MISC9', help='sample')
parser.add_argument('--mask_prob', type=float, default=0.6, help='mask prob')
parser.add_argument('--seed', type=int, default=2024, help='random seed')
parser.add_argument('--adata_path', required=True)
parser.add_argument('--truth_path', required=True)
parser.add_argument('--prediction_path', required=True)
parser.add_argument('--heldout_path', required=True)
parser.add_argument('--output_dir', required=True)
parser.add_argument('--radius', type=float, default=150)
parser.add_argument('--n_neighbors', type=int, default=None, help='KNN excluding self; overrides radius')
parser.add_argument('--save_smoothed', action='store_true', help='Optional; metrics always include smoothed output')
args = parser.parse_args()


def evaluate_metrics(true_exp_matrix, imputed_exp_matrix, mask, save_path):
    mask = mask == 0

    RMSE = np.sqrt(np.mean((true_exp_matrix[mask] - imputed_exp_matrix[mask]) ** 2))

    MAE = np.mean(np.abs(true_exp_matrix[mask] - imputed_exp_matrix[mask]))

    # Calculate Spot-wise Pearson Correlation Coefficient
    spot_corr = np.array([np.corrcoef(imputed_exp_matrix[i, mask[i]], true_exp_matrix[i, mask[i]])[0, 1] 
                          for i in range(imputed_exp_matrix.shape[0])])
    spot_corr = spot_corr[~np.isnan(spot_corr)]

    # Calculate Spot-wise Cosine Similarity
    spot_cosine = np.array([np.dot(imputed_exp_matrix[i, mask[i]], true_exp_matrix[i, mask[i]]) /
                            (np.linalg.norm(imputed_exp_matrix[i, mask[i]]) * np.linalg.norm(true_exp_matrix[i, mask[i]]))
                            for i in range(imputed_exp_matrix.shape[0])])
    spot_cosine = spot_cosine[~np.isnan(spot_cosine)]

    # Calculate Gene-wise Pearson Correlation Coefficient
    gene_corr = np.array([np.corrcoef(imputed_exp_matrix[:, i][mask[:, i]], true_exp_matrix[:, i][mask[:, i]])[0, 1]
                            for i in range(imputed_exp_matrix.shape[1])])
    gene_corr = gene_corr[~np.isnan(gene_corr)]

    # Calculate Gene-wise Cosine Similarity
    gene_cosine = np.array([np.dot(imputed_exp_matrix[:, i][mask[:, i]], true_exp_matrix[:, i][mask[:, i]]) /
                            (np.linalg.norm(imputed_exp_matrix[:, i][mask[:, i]]) * np.linalg.norm(true_exp_matrix[:, i][mask[:, i]]))
                            for i in range(imputed_exp_matrix.shape[1])])
    gene_cosine = gene_cosine[~np.isnan(gene_cosine)]

    # Save the evaluation results to a csv file
    evaluation_results = pd.DataFrame({'RMSE': [RMSE], 'MAE': [MAE], 'Spot-wise PCC': [np.mean(spot_corr)],
                                        'Spot-wise Cosine Similarity': [np.mean(spot_cosine)],
                                        'Gene-wise PCC': [np.mean(gene_corr)], 'Gene-wise Cosine Similarity': [np.mean(gene_cosine)]})
    evaluation_results.to_csv(save_path, index=False)


from pathlib import Path
from experiments.e1.metrics import recovery_metrics, smooth
from spatial_smoothing import spatial_neighbors, complete_expression
from experiments.e1.prepare import digest
import json

def dense(a):
    return a.X.toarray() if hasattr(a.X,'toarray') else np.asarray(a.X)

if not Path(args.adata_path).exists():
    from experiments.e1.prepare import materialize
    materialize(Path(args.adata_path).parent, args.adata_path)
adata=sc.read_h5ad(args.adata_path)
truth=sc.read_h5ad(args.truth_path)[adata.obs_names,adata.var_names].copy()
pred=sc.read_h5ad(args.prediction_path)
if not pred.obs_names.equals(adata.obs_names) or not pred.var_names.equals(adata.var_names):
    raise ValueError('Prediction barcode/gene alignment mismatch')
heldout=sp.sparse.load_npz(args.heldout_path).toarray()
if heldout.dtype != bool or heldout.shape != adata.shape:
    raise ValueError('Expected True=heldout boolean mask')
x,y,z=dense(adata),dense(pred),dense(truth)
if not np.isfinite(y).all():raise ValueError('Nonfinite model output')
completed=complete_expression(x,y)
neighbors=spatial_neighbors(adata.obsm['spatial'],args.radius,args.n_neighbors)
smoothed=smooth(completed,neighbors)
results=[]
for name,matrix in [('raw_input',x),('raw_model',y),('completed',completed),('smoothed',smoothed),('neighbor_mean_baseline',smooth(x,neighbors))]:
    results.append({'output':name,**recovery_metrics(z,matrix,heldout)})
out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
pd.DataFrame(results).to_csv(out/'recovery_metrics.csv',index=False)
if args.save_smoothed:
    a=adata.copy();a.X=smoothed.astype(np.float32);a.write_h5ad(out/'smoothed.h5ad',compression='gzip')
(out/'evaluation_config.json').write_text(json.dumps({'sample':args.sample,'seed':args.seed,'ratio':args.mask_prob,
    'radius':args.radius,'n_neighbors':args.n_neighbors,'smoothing':'one-pass mean including self on undirected radius/KNN graph',
    'truth_sha256':digest(args.truth_path),'input_sha256':digest(args.adata_path),
    'prediction_sha256':digest(args.prediction_path),'heldout_sha256':digest(args.heldout_path),
    'code_sha256':digest(__file__)},indent=2))
print(pd.DataFrame(results).to_string(index=False))
