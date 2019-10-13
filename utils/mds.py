import numpy as np
from sklearn.manifold import MDS

def mds(dmatrix):
    embedding = MDS(n_components=3)
    return embedding.fit_transform(dmatrix.astype(np.float64))    


