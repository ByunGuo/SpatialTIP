"""EDM preparation using the revision experiment's expression/gene-dependent rule."""
import argparse, json, hashlib
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
TEST = ['MISC1','MISC2','MISC3','MISC5','MISC6','MISC8','MISC9','MISC10','MISC12']

def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def matrix(adata):
    x = adata.X.toarray() if hasattr(adata.X, 'toarray') else np.asarray(adata.X)
    if not np.isfinite(x).all() or (x < 0).any():
        raise ValueError('Expected finite nonnegative normalized expression')
    if not adata.var_names.is_unique or not adata.obs_names.is_unique:
        raise ValueError('Gene and barcode IDs must be unique')
    return x

def dependent_mask(x, gamma, ratio, seed, beta=1.0):
    """True means held out. Calibrate expected ratio; retain realized ratio separately."""
    row, col = np.nonzero(x)
    if not len(row):
        raise ValueError('No detected values')
    score = -beta * np.log1p(x[row, col]) + gamma[col]
    low, high = -60.0 - score.max(), 60.0 - score.min()
    for _ in range(80):
        alpha = (low + high) / 2
        p = 1 / (1 + np.exp(-np.clip(alpha + score, -700, 700)))
        if p.mean() < ratio:
            low = alpha
        else:
            high = alpha
    selected = np.random.default_rng(seed).random(len(row)) < p
    mask = np.zeros(x.shape, dtype=bool)
    mask[row[selected], col[selected]] = True
    return mask, {'alpha': float(alpha), 'expected_ratio': float(p.mean()),
                  'realized_ratio': float(selected.mean())}

def materialize(folder, output_path=None):
    """Rebuild masked input from frozen mask; no random sampling or refitting."""
    import anndata as ad
    from scipy.sparse import load_npz, csr_matrix
    folder = Path(folder)
    config = json.loads((folder/'config.json').read_text())
    source = Path(config.get("source_path", ROOT/f"hest_data/st_healthy_opus/{config['sample']}.h5ad"))
    if digest(source) != config['input_sha256']:
        raise ValueError('Original expression changed; frozen mask cannot be reused safely')
    a = ad.read_h5ad(source)
    excluded = config['excluded_missing_patch_barcodes']
    if len(excluded) != len(set(excluded)) or not set(excluded).issubset(set(a.obs_names)):
        raise ValueError('Invalid frozen barcode exclusions')
    a = a[~a.obs_names.isin(excluded)].copy()
    x = matrix(a)
    heldout = load_npz(folder/'heldout.npz').toarray()
    if heldout.dtype != bool or heldout.shape != x.shape or int(heldout.sum()) != config['heldout_count']:
        raise ValueError('Frozen heldout mask mismatch')
    if (heldout & (x==0)).any():
        raise ValueError('Frozen mask selects intrinsic zeros')
    masked = np.where(heldout, 0, x)
    output = Path(output_path) if output_path is not None else folder/'input.h5ad'
    if output.exists():
        old = ad.read_h5ad(output)
        if not old.obs_names.equals(a.obs_names) or not old.var_names.equals(a.var_names) or not np.array_equal(matrix(old), masked):
            raise ValueError('Existing masked input differs from frozen mask')
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    a.X = csr_matrix(masked)
    temporary = output.with_name(output.name+'.temporary')
    a.write_h5ad(temporary, compression='gzip')
    temporary.replace(output)
    return output


def fit_training_effects(paths):
    """Original equal-section training-statistic calculation; targets must be excluded."""
    import anndata as ad
    stats, inputs = {}, {}
    for path in paths:
        a = ad.read_h5ad(path); x = matrix(a)
        inputs[str(path)] = digest(path)
        for gene, rate, mean in zip(a.var_names,(x>0).mean(0),x.mean(0)):
            stats.setdefault(gene,[]).append((float(rate),float(mean)))
    genes = sorted(stats)
    values = np.array([np.mean(stats[g],axis=0) for g in genes])
    rate = np.clip(values[:,0],1e-5,1-1e-5)
    features = np.column_stack([np.log(rate/(1-rate)),np.log1p(values[:,1])])
    z = (features-features.mean(0))/np.maximum(features.std(0),1e-8)
    return {'effect':dict(zip(genes,(-0.5*z.sum(1)).tolist())),'input_sha256':inputs}

def ensure_annotations():
    """Assemble existing DLPFC annotations without changing labels."""
    import pandas as pd
    out = ROOT/'experiments/e1/reference_annotations/dlpfc_barcode_level_layer_map.tsv'
    if out.exists(): return out
    rows=[]
    for path in sorted((ROOT/'DLPFC').glob('*/*_truth.txt')):
        table=pd.read_csv(path,sep='\t',header=None,dtype=str)
        if table.shape[1] != 2: raise ValueError(f'Unexpected annotation columns: {path}')
        table.insert(1,'section',path.parent.name);rows.append(table)
    if not rows: raise FileNotFoundError('DLPFC annotation files are required')
    out.parent.mkdir(parents=True,exist_ok=True)
    pd.concat(rows,ignore_index=True).to_csv(out,sep='\t',header=False,index=False)
    return out

def main():
    import anndata as ad
    import h5py
    from scipy.sparse import csr_matrix, save_npz
    p = argparse.ArgumentParser()
    p.add_argument('--sample', required=True)
    p.add_argument('--expression', type=Path, required=True)
    p.add_argument('--patches', type=Path, required=True)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--gene-effects', type=Path, help='Frozen training-derived effect JSON')
    g.add_argument('--training-inputs', type=Path, nargs='+', help='Only the matching pretrained model training input panels')
    p.add_argument('--ratio', type=float, default=.4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    if not 0 < a.ratio < 1: p.error('ratio must be between 0 and 1')
    target = ad.read_h5ad(a.expression); matrix(target)
    with h5py.File(a.patches,'r') as h:
        bars = [b.decode() if isinstance(b,bytes) else str(b) for b in h['barcode'][:].reshape(-1)]
    if len(bars) != len(set(bars)): raise ValueError('Duplicate patch barcodes')
    excluded = target.obs_names[~target.obs_names.isin(bars)].tolist()
    target = target[~target.obs_names.isin(excluded)].copy()
    if a.training_inputs and a.expression.resolve() in [x.resolve() for x in a.training_inputs]:
        raise ValueError('Target must not be used to fit training gene effects')
    fitted = json.loads(a.gene_effects.read_text()) if a.gene_effects else fit_training_effects(a.training_inputs)
    effects = fitted['effect']
    gamma = np.array([effects.get(g,0.) for g in target.var_names])
    held, info = dependent_mask(matrix(target),gamma,a.ratio,a.seed)
    out = a.output or ROOT/f'experiments/e1/healthy_expression_smoke/expression_gene/{a.seed}/{a.sample}_{a.ratio}'
    out.mkdir(parents=True,exist_ok=False)
    save_npz(out/'heldout.npz',csr_matrix(held))
    (out/'gene_effect.json').write_text(json.dumps(fitted,indent=2))
    cfg = dict(sample=a.sample,source_path=str(a.expression.resolve()),input_sha256=digest(a.expression),excluded_missing_patch_barcodes=excluded,heldout_count=int(held.sum()),seed=a.seed,ratio=a.ratio,mechanism='expression_gene',mask_semantics='True=heldout observed nonzero',gene_effect_sha256=digest(out/'gene_effect.json'),patch_sha256=digest(a.patches),**info)
    (out/'config.json').write_text(json.dumps(cfg,indent=2));materialize(out);print(out)
if __name__ == '__main__': main()
