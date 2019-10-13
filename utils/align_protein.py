import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Converts the 3D coordinates to a distance matrix
def coords_to_distmat(coords):
    # Input: (n,3) matrix
    #  Output: (n,n) matrix

    n = coords.shape[0]
    dist_mat = np.zeros((n,n))

    for i in range(n):
        coords_i = coords[i]

        delta = coords - coords_i
        delta_squared = delta**2
        l2_squared = np.sum(delta_squared, axis=1)
        l2 = np.sqrt(l2_squared)

        dist_mat[i] = l2

    return dist_mat

# Given an matplotlib axis object, returns it with its scales set to be same
def axisEqual3D(ax):
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:,1] - extents[:,0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)

# Returns a rotation matrix that maps v1 to v2
# NOTE: v1, v2 have to be unit vectors
def get_rot_matrix(v1, v2):
    v = np.cross(v1, v2)
    s = np.linalg.norm(v)
    c = np.dot(v1, v2)
    v_x = np.array([[0.,-v[2],v[1]],
                    [v[2],0,-v[0]],
                    [-v[1],v[0],0.]])
    R = np.eye(3) + v_x + np.dot(v_x, v_x)*(1/(1+c))
    
    return R

def read_pkl_std_coords(datafile):
        with open(datafile, 'rb') as fh:
                data = pickle.load(fh)

        casp_ids = list(data.keys())

        for protein in casp_ids:
                coords = data[protein]['coords']

                unit = coords[0] / np.sqrt(np.sum(coords[0]**2)) # Unit vector of first amino acid
                unit2 = coords[1] / np.sqrt(np.sum(coords[1]**2)) # Unit vector of second amino acid
                R_1 = get_rot_matrix(unit, np.array([0,0,1])) # Get rotation matrix that rotates it to [0,0,1]
                R_2 = get_rot_matrix(unit2, np.array([0,1,0])) # Get rotation matrix that rotates it to [0,1,0]

                offset = coords[0]
                orig_coords = coords - offset # Linear Translation
                std_R1 = np.array([np.matmul(R_1,p) for p in orig_coords]) # Final rotated coordinates
                std_R2 = np.array([np.matmul(R_2,p) for p in std_R1]) #

                data[protein]['coords'] = std_R2

        return data     

