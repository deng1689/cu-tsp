import numpy as np
import Bio.PDB
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.DSSP import DSSP
from Bio.PDB.Polypeptide import PPBuilder
from Bio.SVDSuperimposer import SVDSuperimposer
import pickle
from pdb import set_trace

# Calculate ground-truth 3d coordinates.
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

def compute_drmsd(a, b):
    return (1.0 / len(a) * np.sum((a - b)**2))**0.5

def align(predicted, gt):
    """
    # Grid search through scales for affine alignment.
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
                scaled_predicted = np.dot(np.array(predicted), scaling)
                sup.set(np.array(gt), scaled_predicted)
                sup.run()
                rms = sup.get_rms()
                rot, tran = sup.get_rotran()
                b = sup.get_transformed()
                a = np.array(gt)
                drmsd = compute_drmsd(a, b)
                if drmsd < best_drmsd:
                    best_drmsd = drmsd
                    best_sx = sx
                    best_sy = sy
                    best_sz = sz
    """
    best_sx = 1
    best_sy = 1
    best_sz = 1
    
    # Use best sx, sy, sz to perform final alignment.
    sup = SVDSuperimposer()
    scaling = np.diag([best_sx, best_sy, best_sz])
    scaled_predicted = np.dot(np.array(predicted), scaling)
    sup.set(np.array(gt), scaled_predicted)
    sup.run()
    predicted = sup.get_transformed()
    return predicted

path = 'data/output_dict.pkl'
pred = pickle.load(open(path, 'rb'))
pred = pred['coordinates'][1][:116, :]
A = get_calpha_positions('data/35k/test/5j4a.pdb')
aligned_pred = align(pred, A)
print(compute_drmsd(aligned_pred, np.array(A)))
