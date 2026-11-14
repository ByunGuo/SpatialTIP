"""Spatial-domain metrics used in the revision analysis."""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, f1_score, accuracy_score

def graph(xy):
    tree = cKDTree(xy)
    nearest = tree.query(xy,k=2)[0][:,1]
    # Immediate physical neighbors; threshold fixed from geometry, independent of labels/output.
    radius = float(np.median(nearest)*1.25)
    pairs = np.array(sorted(tree.query_pairs(radius)),dtype=int).reshape(-1,2)
    return tree, pairs, radius

def scores(labels, truth, edges):
    valid = pd.notna(truth) & ~np.isin(np.asarray(truth,dtype=str),['unknown','Unknown','NA','nan',''])
    e = edges[valid[edges].all(axis=1)]
    actual = truth[e[:,0]] != truth[e[:,1]]
    pred = labels[e[:,0]] != labels[e[:,1]]
    return dict(n_gt=int(valid.sum()),ARI=adjusted_rand_score(truth[valid],labels[valid]),
        NMI=normalized_mutual_info_score(truth[valid],labels[valid]),
        boundary_edge_F1=f1_score(actual,pred,zero_division=0),
        boundary_edge_accuracy=accuracy_score(actual,pred),n_gt_edges=len(e),
        spatial_continuity=float(np.mean(labels[edges[:,0]]==labels[edges[:,1]])))
