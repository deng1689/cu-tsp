import numpy as np
import scipy

from recon_tester import recover_coords, align
from build3D import MatrixTo3D
from mds import mds
from pdb import set_trace

def get_distance(coords):
    return scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(coords))

def compute_drmsd(a, b):
    return (1.0 / len(a) * np.sum((a - b)**2))**0.5

def recon(D, method):
    if method == 'SDP':
        return recover_coords(D, method='SDP')
    elif method == 'MDS':
        return mds(D)
    elif method == 'FM':
        return MatrixTo3D(D)[0]
    else:
        print("Unsupported method!")

def main():
    N = 50
    methods = ['SDP', 'MDS', 'FM']
    gt_coords = np.random.uniform(size=(N, 3))
    d_matrix = get_distance(gt_coords)
    for method in methods:
        pred_coords = recon(d_matrix, method)
        aligned_coords = align(pred_coords, gt_coords)
        print("[{}] Pre-aligned DRMSD: {}".format(method, compute_drmsd(pred_coords, gt_coords)))
        print("[{}] Post-aligned DRMSD: {}".format(method, compute_drmsd(aligned_coords, gt_coords)))

main()
