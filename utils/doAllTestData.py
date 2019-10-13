import numpy as np
import pickle
import Bio.PDB
import cv2
from sklearn.neighbors import NearestNeighbors
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.DSSP import DSSP
from Bio.PDB.Polypeptide import PPBuilder
from Bio.SVDSuperimposer import SVDSuperimposer
from Bio.PDB.QCPSuperimposer import QCPSuperimposer
from scipy.spatial import distance_matrix
from functools import partial
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pdb import set_trace
matplotlib.pyplot.switch_backend('agg')
import pandas as pd

dims = [75, 116, 136, 300, 229, 239]
proteins = ['2n64','5j4a','5fjl','3jb5','5fhy','5jo9']

def get_calpha_distance_matrix(pdb_path):
    structure = Bio.PDB.PDBParser(QUIET=True).get_structure(pdb_path[:-4], pdb_path)
    A = []
    for model in structure:
        for chain in model:
            for res in chain:
                try:
                    coord = res['CA'].get_coord()
                    A.append(np.asarray(coord))
                except:
                    continue
            if A: # first chain not empty
            	D = distance_matrix(A,A)
            	return D
            	
def get_calpha_positions(pdb_path):
    structure = Bio.PDB.PDBParser(QUIET=True).get_structure(pdb_path[:-4], pdb_path)
    A = []
    for model in structure:
        for chain in model:
            for res in chain:
                try:
                    coord = res['CA'].get_coord()
                    A.append(np.asarray(coord))
                except:
                    continue
            if A: # first chain not empty
            	return A

gt = []
for protein in proteins:
    pdb_path = protein + '.pdb'
    A = get_calpha_positions(pdb_path)
    gt.append(A)

import pickle
filename0 = '3dcoordsgt.pkl'
filename1 = '3dcoordsfull.pkl'
#filename1 = 'predicted_3d_coords_wf2232-yg2541-new.pkl'
#filename1 = 'predicted_3d_coords_wf2232-yg2541.pkl'
#filename3 = 'coordinates-submission.pkl'


#pdbs = []
#dcalphas = []
#for pdb in proteins:
#	pdbs.append(pdb)
#	pdb_path = pdb + '.pdb'
#	dcalpha = get_calpha_distance_matrix(pdb_path)
#	dcalphas.append(dcalpha)
#datafile = 'dcalphas-test.pkl'
#with open(datafile, 'wb') as dataoutfile:
#	pickle.dump(dcalphas, dataoutfile, pickle.HIGHEST_PROTOCOL)

#with open('test.pkl', 'rb') as fin:
#    pdbs = pickle.load(fin)[1]
#with open('dcalphas-test.pkl', 'rb') as fin:
#    dcalphas = pickle.load(fin)
    #with open(args.psis_path, 'rb') as fin:
    #    psis = pickle.load(fin)
    #with open(args.phis_path, 'rb') as fin:
    #    phis = pickle.load(fin)

#tb = []
    #for pdb, d, ps, ph in zip(pdbs, dcalphas, psis, phis):
#	for pdb, d in zip(pdbs, dcalphas):
#for pdb, d in zip(pdbs, dcalphas):
#    for i in range(len(d)):
#        for j in range(len(d[i])):
#            tb += [('{}_d_{}_{}'.format(pdb, i + 1, j + 1), d[i][j])]
        #for i in range(len(ps)):
        #    tb += [('{}_psi_{}'.format(pdb, i + 1), ps[i])]
        #for i in range(len(ph)):
        #    tb += [('{}_phi_{}'.format(pdb, i + 1), ph[i])]
#tb = pd.DataFrame(tb, columns=['Id', 'Predicted'])
#tb.to_csv('ground-truth.csv', index=False)
    
with open(filename0, 'rb') as f0:
    data0 = pickle.load(f0) 
gt = []
for i in range(len(proteins)):
    gt.append(data0[i])
    
with open(filename1, 'rb') as f:
    data = pickle.load(f) 
predicted = []
for i in range(len(proteins)):
    predicted.append(data[i])

drmsds = []
scales = []
for i in range(len(proteins)):
    scale_range = np.arange(0.9, 1.1, 0.05)
    best_drmsd = float("inf")
    best_sx = 1
    best_sy = 1
    best_sz = 1
    for sx in scale_range:
        for sy in scale_range:
            for sz in scale_range:
                sup = SVDSuperimposer()
                scaling = np.diag([sx, sy, sz])
                scaled_predicted = np.dot(np.array(predicted[i]), scaling)
                print(predicted[i])
                sup.set(np.array(gt[i]), scaled_predicted)
                sup.run()
                rms = sup.get_rms()
                rot, tran = sup.get_rotran()
                b = sup.get_transformed()
                a = np.array(gt[i])
                drmsd = (1.0 / len(a) * np.sum((a - b)**2))**0.5
                if drmsd < best_drmsd:
                    best_drmsd = drmsd
                    best_sx = sx
                    best_sy = sy
                    best_sz = sz
    scales.append((best_sx, best_sy, best_sz))
    drmsds.append(best_drmsd)
print(drmsds)

for i in range(len(proteins)):
    sup = SVDSuperimposer()
    scaling = np.diag([scales[i][0], scales[i][1], scales[i][2]])
    scaled_predicted = np.dot(np.array(predicted[i]), scaling)
    sup.set(np.array(gt[i]), scaled_predicted)
    sup.run()
    predicted[i] = sup.get_transformed()

def visualize(X, Y, ax):
        plt.draw()

for i in range(len(proteins)):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    gt0 = np.array(gt[i])
    pred0 = np.array(predicted[i])
    plt.cla()
    ax.scatter(gt0[:,0],  gt0[:,1], gt0[:,2], color='red', label='GT')
    ax.scatter(pred0[:,0],  pred0[:,1], pred0[:,2], color='blue', label='Predicted')
    ax.set_title(proteins[i])
    ax.legend(loc='upper left', fontsize='x-large')
    plt.savefig('{}_scatter_{}.png'.format(filename1[:-4], proteins[i]))
