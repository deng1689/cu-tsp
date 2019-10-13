import numpy as np
import argparse
import math
import torch
import torch.nn.functional as F
import Bio.PDB
import pickle
import subprocess
import PeptideBuilder
import matplotlib.pyplot as plt

from Bio.PDB import *
from Bio.PDB import PDBParser
from Bio.PDB import Superimposer
from Bio.PDB.Atom import *
from Bio.PDB.Residue import *
from Bio.PDB.Chain import *
from Bio.PDB.Model import *
from Bio.PDB.Structure import *
from Bio.PDB.Vector import *
from Bio.PDB.Entity import *
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.DSSP import DSSP
from Bio.PDB.PDBList import PDBList
from pyrosetta import *
from pyrosetta.rosetta.core.id import AtomID
from pyrosetta.rosetta.utility import vector1_core_id_AtomID, \
        vector1_numeric_xyzVector_double_t
from pyrosetta.rosetta import numeric
init()
from recon_tester import recover_coords, align
from build3D import MatrixTo3D
from pdb import set_trace

def load_data(path, is_pred=False):
    data = pickle.load(open(path, 'rb'))
    pdbs = sorted(list(data.keys()))
    out= dict()
    if not is_pred:
        out['lengths'] = []
        out['chains'] = []
        out['aas'] = []
        out['q8s'] = []
        out['pssms'] = []

    out['pdbs'] = pdbs
    out['dcalphas'] = []
    out['coords'] = []
    out['psis'] = []
    out['phis'] = []
    for p in pdbs:
        if not is_pred:
            out['lengths'].append(data[p]['length'])
            out['chains'].append(data[p]['chain'])
            out['aas'].append(data[p]['aa'])
            out['q8s'].append(data[p]['ss'])
            out['pssms'].append(data[p]['pssm'])
            out['coords'].append(data[p]['coords'])
        if 'model_1' in path:
            out['dcalphas'].append(data[p]['dcalphas'])
        else:
            out['dcalphas'].append(data[p]['dcalpha'])
        #out['coords'].append(data[p]['coords'])
        out['psis'].append(data[p]['psi'])
        out['phis'].append(data[p]['phi'])
    return out

def calc_pairwise_distances(chain_a, chain_b, use_gpu):
    distance_matrix = torch.Tensor(chain_a.size()[0], chain_b.size()[0]).type(torch.float)
    # add small epsilon to avoid boundary issues
    epsilon = 10 ** (-4) * torch.ones(chain_a.size(0), chain_b.size(0))
    if use_gpu:
        distance_matrix = distance_matrix.cuda()
        epsilon = epsilon.cuda()

    for i, row in enumerate(chain_a.split(1)):
        distance_matrix[i] = torch.sum((row.expand_as(chain_b) - chain_b) ** 2, 1).view(1, -1)

    return torch.sqrt(distance_matrix + epsilon)

def calc_drmsd(chain_a, chain_b, use_gpu=False):
    assert len(chain_a) == len(chain_b)
    chain_a = torch.from_numpy(chain_a)
    chain_b = torch.from_numpy(chain_b)
    distance_matrix_a = calc_pairwise_distances(chain_a, chain_a, use_gpu)
    distance_matrix_b = calc_pairwise_distances(chain_b, chain_b, use_gpu)
    return torch.norm(distance_matrix_a - distance_matrix_b, 2) \
            / math.sqrt((len(chain_a) * (len(chain_a) - 1)))

def calc_gdt_ts(a, b):
    N = a.shape[0]
    count1, count2, count4, count8 = 0, 0, 0, 0
    for i in range(N):
        d = np.sum((a[i, :] - b[i, :])**2)**0.5
        if d <= 1:
            count1 += 1
        if d <= 2:
            count2 += 1 
        if d <= 4:
            count4 += 1
        if d <= 8:
            count8 += 1
    gdt_ts = ((count1 + count2 + count4 + count8) / (4*N)) * 100
    return gdt_ts

def set_pdb_coords(pdb, coords, out_path):
    pose = pose_from_pdb(pdb)
    n = coords.shape[0]
    for i in range(1, n + 1):
        x, y, z = coords[i - 1, :]
        out = numeric.xyzVector_double_t(x, y, z)
        pose.residues[i].atom("CA").xyz(out)
    pose.dump_pdb(out_path)

def to_pdb(aa, phi, psi, path):
    phi, psi = phi[1:], psi[:-1]
    struct = PeptideBuilder.make_structure(aa, phi, psi)
    out = Bio.PDB.PDBIO()
    out.set_structure(struct)
    out.save(path)

def print_metrics(p, gt_path, pred_path, a_data, b_data, method):
    gt_pdb_path = '{}_{}.pdb'.format(p, gt_path[:-4])
    pred_pdb_path = '{}_{}_{}.pdb'.format(p, pred_path[:-4], method)
    gt_aa, gt_phi, gt_psi, gt_coords = a_data
    if method == 'torsion':
        b_aa, b_phi, b_psi = b_data
        gt_path = gt_path[:-4]
        pred_path = pred_path[:-4]
        to_pdb(gt_aa, gt_phi, gt_psi, gt_pdb_path)
        to_pdb(b_aa, b_phi, b_psi, pred_pdb_path)
    elif method == 'dcalpha':
        pred_dcalpha = b_data
        #pred_coords = recover_coords(pred_dcalpha, method='SDP') 
        #aligned_coords = align(pred_coords, gt_coords)
        #pred_coords, _, _ = MatrixTo3D(pred_dcalpha) 
        aligned_coords = pred_coords

        to_pdb(gt_aa, gt_phi, gt_psi, gt_pdb_path)
        set_pdb_coords(gt_pdb_path, aligned_coords, pred_pdb_path)
    print("-----------{}-{}------------".format(p, method))
    proc = subprocess.Popen('java -jar TMscore.jar {} {}'.format(gt_pdb_path, pred_pdb_path),
                stdout=subprocess.PIPE, shell=True)
    (out, err) = proc.communicate()
    out = out.decode('utf-8').split('\n')
    for i in range(10, 35):
        print(out[i].rstrip())
    print("---------------------------".format(p))

def torsion_to_3d(aa, phi, psi):
    phi, psi = phi[1:], psi[:-1]
    recon_struct = PeptideBuilder.make_structure(aa, phi, psi)
    atoms = recon_struct.get_atoms()
    coords = list()
    for atom in atoms:
        coords.append(atom.get_coord())
    return np.array(coords)
    
def main(pred_path, gt_path):
    # Change is_pred to True when passing in a valid prediction file!
    pred_data = load_data(pred_path, is_pred=True)
    gt_data = load_data(gt_path, is_pred=False)

    for (i, p) in enumerate(gt_data['pdbs']):
        #i = 2
        #p = gt_data['pdbs'][i]
        # Torsion reconstruction.
        tor_gt_coords = torsion_to_3d(gt_data['aas'][i], 
                        gt_data['phis'][i], 
                        gt_data['psis'][i])
        tor_pred_coords = torsion_to_3d(gt_data['aas'][i], 
                        pred_data['phis'][i], 
                        pred_data['psis'][i])
        pred_dcalpha = pred_data['dcalphas'][i]
        gt_dcalpha = gt_data['dcalphas'][i]
        
        gt_d = gt_data['aas'][i], gt_data['phis'][i], gt_data['psis'][i], gt_data['coords'][i]
        """
        print_metrics(p, gt_path, pred_path, gt_d, pred_dcalpha, method='dcalpha')
        plt.imshow(gt_dcalpha, cmap='plasma')
        plt.savefig('testing_{}_dcalpha.png'.format(p))
        plt.imshow(pred_dcalpha, cmap='plasma')
        plt.savefig('{}_{}_dcalpha.png'.format(pred_path[:-4], p))
        tor_drmsd = calc_drmsd(tor_gt_coords, tor_pred_coords)
        """
        
        tor_pred_data = gt_data['aas'][i], pred_data['phis'][i], pred_data['psis'][i]
        print_metrics(p, gt_path, pred_path, gt_d, tor_pred_data, method='torsion')
        #gt_coords = np.array(gt_data['coords'][i])
        #pred_coords = np.array(pred_data['coords'][i])
        #coord_drmsd = calc_drmsd(gt_coords, pred_coords)
        #coord_gdt_ts = calc_gdt_ts(gt_coords, pred_coords)

        #print("3dcoord, {}, {}, {}".format(p, coord_drmsd, coord_gdt_ts))
        #print("torsion, {}, {}, {}".format(p, tor_drmsd, tor_gdt_ts))

def get_amino_acid_sequences(pdb_path):
    structure = Bio.PDB.PDBParser(QUIET=True).get_structure(pdb_path[:-4], pdb_path)
    ppb = PPBuilder()
    pdb_aas = []
    for pp in ppb.build_peptides(structure): 
        pdb_aa = str(pp.get_sequence())
        pdb_aas.append(pdb_aa)
    return pdb_aas
    
def pdb_to_torsion_and_res(pdb_path):
    structure = Bio.PDB.PDBParser(QUIET=True).get_structure(pdb_path[:-4], pdb_path)
    A = []
    ppb = PPBuilder()
    pdb_aas = []
    model = ppb.build_peptides(structure)
    chain = model[0]
    phi_psi_list = chain.get_phi_psi_list()
    res_list = get_amino_acid_sequences(pdb_path)[0]
    return [x[0] for x in phi_psi_list], [x[1] for x in phi_psi_list], [x[0] for x in res_list]

def torsion_radians_to_degrees_sanitize(phi, psi):
    for i in range(0, len(phi)): # convert to degrees, first is None
        if phi[i] != None:
            phi[i] = phi[i] * (180/np.pi)
        else:
            phi[i] = 0
    for i in range(0, len(psi)): # convert to degrees, last is None
        if psi[i] != None:
            psi[i] = psi[i] * (180/np.pi)
        else:
            psi[i] = 0				
    return phi, psi

def torsion_and_res_to_pdb(phis, psis, ress, pdb_path):
	topo_filename = '{}.topo'.format(pdb_path)
	topo_file = open(topo_filename, 'w')
	for i in range(len(ress)):
		topo_file.write(str(phis[i])+' '+str(psis[i])+' '+str(ress[i])+'\n')
	topo_file.close()
	proc = subprocess.Popen('perl pdb_from_torsions.pl {} > {}'.format(topo_filename, pdb_path), stdout=subprocess.PIPE, shell=True)

def torsion_to_topo_to_pdb(pred_path, gt_path):
    pred_data = load_data(pred_path, is_pred=True)
    gt_data = load_data(gt_path, is_pred=False)
    for (i, p) in enumerate(gt_data['pdbs']):
        gt_filename = 'testing_{}.topo'.format(p)
        pred_filename = '{}_{}.topo'.format(pred_path[:-4], p)
        gt_file = open(gt_filename, 'w')
        pred_file = open(pred_filename, 'w')
        for j in range(len(gt_data['aas'][i])):
            gt_phi = gt_data['phis'][i][j]
            gt_psi = gt_data['psis'][i][j]
            res = gt_data['aas'][i][j]
            pred_phi = pred_data['phis'][i][j]
            pred_psi = pred_data['psis'][i][j]
            gt_file.write(str(gt_phi)+' '+str(gt_psi)+' '+str(res)+'\n')
            pred_file.write(str(pred_phi )+' '+str(pred_psi)+' '+str(res)+'\n')	
        gt_file.close()
        pred_file.close()
        gt_pdb_filename = 'testing_{}.pdb'.format(p)
        pred_pdb_filename = '{}_{}.pdb'.format(pred_path[:-4], p)
        proc = subprocess.Popen('perl pdb_from_torsions.pl {} > {}'.format(gt_filename, gt_pdb_filename), stdout=subprocess.PIPE, shell=True)
        proc = subprocess.Popen('perl pdb_from_torsions.pl {} > {}'.format(pred_filename, pred_pdb_filename), stdout=subprocess.PIPE, shell=True)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("pred_path", help="Path to pickle file for predictions")
    parser.add_argument("gt_path", help="Path to pickle file for ground truth")
    args = parser.parse_args()
    pred_path = args.pred_path
    gt_path = args.gt_path
    
    # pdb -> torsion, res -> pdb
    #in_path = '2n64.pdb'
    #out_path = '2n64_recon.pdb'
    #phi, psi, res = pdb_to_torsion_and_res(in_path)
    #phi, psi = torsion_radians_to_degrees_sanitize(phi, psi)
    #torsion_and_res_to_pdb(phi, psi, res, out_path)

    main(pred_path, gt_path)
