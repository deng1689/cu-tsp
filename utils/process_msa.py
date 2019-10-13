import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from pdb import set_trace

embedding1 = {
'A': [1.8, 0.67, 6, 0, 0, 4], 
'C': [2.5, 0.38, 7, 0, 0, 9],
'D': [-3.5, -1.2, 9, -1, 0, 6],
'E': [-3.5, -0.75, 10, -1, 0, 5],
'F': [2.8, -0.57, 12, 0, 1, 6],
'G': [-0.4, 0.48, 5, 0, 0, 6],
'H': [-3.2, 0.64, 11, 1, 1, 8],
'I': [4.5, 1.9, 9, 0, 0, 4],
'K': [-3.9, -0.57, 10, 1, 0, 5],
'L': [3.8, 1.9, 9, 0, 0, 4],
'M': [1.9, 2.4, 9, 0, 0, 5],
'N': [-3.5, -0.6, 9, 0, 0, 6],
'P': [-1.6, 1.2, 8, 0, 0, 7],
'Q': [-3.5, -0.22, 10, 0, 0, 5],
'R': [-4.5, -2.1, 12, 1, 0, 5],
'S': [0.8, 0.01, 7, 0, 0, 4],
'T': [-0.7, 0.52, 8, 0, 0, 5],
'V': [4.2, 1.5, 8, 0, 0, 4],
'W': [-0.9, 2.6, 15, 0, 1, 11],
'Y': [-0.13, 1.6, 13, 0, 1, 7],
'X': [0, 0, 0, 0, 0, 0]}

embedding2 = {
'A': [0.748392, 0.1432352, -1.388808078, -0.09796, -0.487339717, -0.97169], 
'C': [0.983156, -0.0868934, -0.974238503, -0.09796, -0.487339717, 1.72744], 
'D': [-1.0291, -1.3406976, -0.145099351, -2.05714, -0.487339717, 0.107965],
'E': [-1.0291, -0.9836014, 0.269470224, -2.05714, -0.487339717, -0.43186], 
'F': [1.083768, -0.840763, 1.098609375, -0.09796, 1.949358869, 0.107965],
'G': [0.010564, -0.0075387, -1.803377654, -0.09796, -0.487339717, 0.107965], 
'H': [-0.92849, 0.1194288, 0.6840398, 1.86122, 1.949358869, 1.187615],
'I': [1.653908, 1.119298, -0.145099351, -0.09796, -0.487339717, -0.97169],
'K': [-1.16325, -0.840763, 0.269470224, 1.86122, -0.487339717, -0.43186],
'L': [1.419145, 1.119298, -0.145099351, -0.09796, -0.487339717, -0.97169],
'M': [0.78193, 1.5160715, -0.145099351, -0.09796, -0.487339717, -0.43186],
'N': [-1.0291, -0.8645694, -0.145099351, -0.09796, -0.487339717, 0.107965], 
'P': [-0.39189, 0.5638151, -0.559668927, -0.09796, -0.487339717, 0.64779],
'Q': [-1.0291, -0.5630216, 0.269470224, -0.09796, -0.487339717, -0.43186],
'R': [-1.36448, -2.0548898, 1.098609375, 1.86122, -0.487339717, -0.43186],
'S': [-0.12359, -0.3805058, -0.974238503, -0.09796, -0.487339717, -0.97169],
'T': [-0.09005, 0.0242032, -0.559668927, -0.09796, -0.487339717, -0.43186],
'V': [1.553295, 0.8018792, -0.559668927, -0.09796, -0.487339717, -0.97169],
'W': [-0.15712, 1.6747808, 2.342318102, -0.09796, 1.949358869, 2.80709],
'Y': [0.101116, 0.8812339, 1.513178951, -0.09796, 1.949358869, 0.64779],
'X': [0, 0, 0, 0, 0, 0]}

one_hot = dict()
alpha_size = len(embedding1.keys()) + 1
for i, key in enumerate(embedding1.keys()):
    one_hot[key] = np.zeros(alpha_size)
    one_hot[key][i] = 1
one_hot['X'] = np.zeros(alpha_size)

def process_msa(fp):
    homologs = list()
    with open(fp, 'r') as f:
        buf = list()
        for line in f:
            if line.startswith('>'): 
                if len(buf) != 0:
                    homolog = "".join(buf) 
                    homolog = "".join(x for x in homolog if not x.islower())
                    homologs.append(homolog)
                buf = list()
            else:
                buf.append(line.rstrip())
    pdb = "".join(buf)
    return homologs, pdb

# Equivalent to doing a matrix multiplication of a n x k matrix with a
# k x n matrix, but where each element of the matrix is in the field
# of size embedding dimension.
def get_covariance(homologs, embedding_dict):
    homologs = [list(x) for x in homologs]
    for (i, y) in enumerate(homologs):
        for (j, x) in enumerate(y):
            try:
                homologs[i][j] = embedding_dict[x]
            except:
                homologs[i][j] = embedding_dict['X']
    homologs = np.array(homologs)
    covariance = np.tensordot(homologs.transpose([1, 0, 2]), homologs, 
            axes=([1, 2], [0, 2]))
    return covariance

def main():
    fp = '103L_1_A.a2m'

    homologs, pdb = process_msa(fp)
    cov_onehot = get_covariance(homologs, one_hot)
    cov_embedding1 = get_covariance(homologs, embedding1)
    cov_embedding2 = get_covariance(homologs, embedding2)
    plt.imshow(cov_onehot, cmap='plasma')
    plt.savefig('{}_cov_onehot.png'.format(fp[:-4]))
    plt.imshow(cov_embedding1, cmap='plasma')
    plt.savefig('{}_cov_embedding1.png'.format(fp[:-4]))
    plt.imshow(cov_embedding2, cmap='plasma')
    plt.savefig('{}_cov_embedding2.png'.format(fp[:-4]))

main()
