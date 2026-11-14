"""Export matched counts and normalized expression for reference-method adapters."""
import argparse,json
from pathlib import Path
import anndata as ad
import h5py
import numpy as np
from scipy.io import mmwrite
from scipy.sparse import csr_matrix
from experiments.edm.prepare import ROOT,TEST,digest,matrix
p=argparse.ArgumentParser()
p.add_argument('--sample',choices=TEST,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
x_path=ROOT/f'hest_data/st_healthy_opus/{a.sample}.h5ad'
c_path=ROOT/f'hest_data/st/{a.sample}.h5ad'
patch_path=ROOT/f'hest_data/st/patches/{a.sample}.h5'
x=ad.read_h5ad(x_path);counts=ad.read_h5ad(c_path)
counts.var_names_make_unique();counts.obs_names_make_unique()
with h5py.File(patch_path,'r') as h:
 bar=[b.decode() if isinstance(b,bytes) else str(b) for b in h['barcode'][:].reshape(-1)]
 if len(bar)!=len(set(bar)):raise ValueError('Duplicate patch barcodes')
rows=x.obs_names.isin(bar)
excluded=x.obs_names[~rows].tolist()
x=x[rows].copy();counts=counts[x.obs_names,x.var_names].copy()
c=matrix(counts);n=matrix(x)
if not np.allclose(c,np.rint(c)) or ((c==0)!=(n==0)).any():
 raise ValueError('Counts and expression are not aligned')
a.output.mkdir(parents=True,exist_ok=False)
# Both matrices are spot x gene; scImpute adapter must transpose to gene x spot.
mmwrite(a.output/'counts_spot_gene.mtx',csr_matrix(c))
mmwrite(a.output/'normalized_spot_gene.mtx',csr_matrix(n))
(a.output/'barcodes.tsv').write_text('\n'.join(x.obs_names)+'\n')
(a.output/'genes.tsv').write_text('\n'.join(x.var_names)+'\n')
(a.output/'config.json').write_text(json.dumps({'sample':a.sample,'shape':list(x.shape),
 'matrix_orientation':'spot x gene','excluded_missing_patch_barcodes':excluded,
 'counts_sha256':digest(c_path),'expression_sha256':digest(x_path)},indent=2))
print(a.output)
