"""Metrics over explicit heldout entries; smoothing reads a fixed source."""
import numpy as np

def recovery_metrics(truth, prediction, heldout):
    truth=np.asarray(truth,dtype=np.float64)
    prediction=np.asarray(prediction,dtype=np.float64)
    heldout=np.asarray(heldout)
    if truth.shape != prediction.shape or heldout.shape != truth.shape or heldout.dtype != bool:
        raise ValueError('Aligned matrices and boolean heldout mask required')
    if not heldout.any() or (truth[heldout] <= 0).any():
        raise ValueError('Evaluation requires heldout observed nonzero entries')
    if not np.isfinite(prediction[heldout]).all():
        raise ValueError('Nonfinite predictions on heldout entries')
    error=truth[heldout]-prediction[heldout]
    result={'RMSE':float(np.sqrt(np.mean(error**2))),'MAE':float(np.mean(abs(error))),
            'heldout_count':int(heldout.sum())}
    t=np.where(heldout,truth,0);p=np.where(heldout,prediction,0)
    for name,axis in [('spot',1),('gene',0)]:
        n=heldout.sum(axis=axis,keepdims=True)
        ct=(t-t.sum(axis=axis,keepdims=True)/np.maximum(n,1))*heldout
        cp=(p-p.sum(axis=axis,keepdims=True)/np.maximum(n,1))*heldout
        denom=np.sqrt((ct**2).sum(axis=axis)*(cp**2).sum(axis=axis))
        valid=(n.reshape(-1)>=2)&(denom>0)
        corr=(ct*cp).sum(axis=axis)[valid]/denom[valid]
        corr=corr[np.isfinite(corr)]
        norm=np.sqrt((t**2).sum(axis=axis)*(p**2).sum(axis=axis))
        valid=norm>0
        cosine=(t*p).sum(axis=axis)[valid]/norm[valid]
        cosine=cosine[np.isfinite(cosine)]
        for metric,values in [('PCC',corr),('CSS',cosine)]:
            result[f'{name}_{metric}']=float(np.mean(values)) if len(values) else None
            result[f'{name}_{metric}_valid_count']=len(values)
    return result

def smooth(prediction, neighbors):
    from spatial_smoothing import average_pool
    return average_pool(prediction, neighbors)


def proxy_labels(original_zero, dropout_votes, retained_votes):
    """Each adapter provides 3 aligned boolean votes; ambiguous stays unlabeled."""
    zero=np.asarray(original_zero,dtype=bool)
    drop=np.asarray(dropout_votes)
    keep=np.asarray(retained_votes)
    if drop.shape!=(3,*zero.shape) or keep.shape!=drop.shape:
        raise ValueError('Three aligned reference methods required')
    if drop.dtype!=bool or keep.dtype!=bool or (drop & keep).any():
        raise ValueError('Reference votes must be boolean and mutually exclusive')
    labels=np.full(zero.shape,-1,dtype=np.int8)
    labels[zero & keep.all(0)]=0
    labels[zero & drop.all(0)]=1
    return labels
