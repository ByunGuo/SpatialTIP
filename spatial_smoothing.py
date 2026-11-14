"""SpatialTIP expression assembly and one-pass spatial average pooling."""
import numpy as np
from sklearn.neighbors import NearestNeighbors

def complete_expression(observed, prediction):
    observed = observed.toarray() if hasattr(observed, 'toarray') else np.asarray(observed)
    prediction = prediction.toarray() if hasattr(prediction, 'toarray') else np.asarray(prediction)
    if observed.shape != prediction.shape:
        raise ValueError('Expression and prediction shapes must match')
    return np.where(observed != 0, observed, prediction)

def spatial_neighbors(coordinates, radius=150, n_neighbors=None):
    coordinates = np.asarray(coordinates)
    n = len(coordinates)
    if n == 0: raise ValueError('Empty spatial coordinates')
    if n_neighbors is None:
        if radius <= 0: raise ValueError('radius must be positive')
        rows = NearestNeighbors(radius=radius).fit(coordinates).radius_neighbors(coordinates,return_distance=False)
    else:
        if n_neighbors < 0 or int(n_neighbors) != n_neighbors:
            raise ValueError('n_neighbors must be a nonnegative integer excluding self')
        rows = NearestNeighbors(n_neighbors=min(int(n_neighbors)+1,n)).fit(coordinates).kneighbors(coordinates,return_distance=False)
    graph = {i:{i} for i in range(n)}
    for i,row in enumerate(rows):
        for j in row:
            graph[i].add(int(j));graph[int(j)].add(i)
    return graph

def average_pool(expression, neighbors):
    source = expression.toarray() if hasattr(expression, 'toarray') else np.asarray(expression)
    result = np.empty(source.shape,dtype=np.result_type(source.dtype,np.float32))
    for i in range(len(source)):
        indices = sorted(set(map(int,neighbors.get(i,()))) | {i})
        result[i] = source[indices].mean(axis=0)
    return result

def spatial_smoothing(observed, prediction, coordinates, radius=150, n_neighbors=None):
    return average_pool(complete_expression(observed,prediction),
                        spatial_neighbors(coordinates,radius,n_neighbors))
