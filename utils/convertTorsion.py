import math
import collections
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
import h5py
from datetime import datetime
import PeptideBuilder
import Bio.PDB
import math
import numpy as np
import time
from numpy import array
import pickle
from PeptideBuilder import Geometry

#!/usr/bin/python

from Bio.PDB import *
from Bio.PDB import PDBParser
from Bio.PDB import Superimposer
from Bio.PDB.Atom import *
from Bio.PDB.Residue import *
from Bio.PDB.Chain import *
from Bio.PDB.Model import *
from Bio.PDB.Structure import *
from Bio.PDB.Vector import *
from Bio.PDB.Entity import*
import math
from PeptideBuilder import Geometry
import PeptideBuilder
import numpy


from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.DSSP import DSSP
from Bio.PDB.PDBList import PDBList

from scipy.spatial.distance import *

from pyrosetta import *
from pyrosetta.rosetta import *
from pyrosetta.toolbox import *
from pyrosetta.toolbox.cleaning import cleanATOM
from pdb import set_trace
#init()
resdict = { 'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', \
        'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L', \
        'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', \
        'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y' }
def get_pdb_from_torsion(aa_seq,phis, psis, save_dir = "/Users/arjunsrivatsa/Desktop/final_reconstruct.pdb"): 
        #phis is of length n, disregard first one
        #psis is of length n, disregard the last one
        #aa_seq is a string of amino acids encoded by capital letters
        phis = phis[1:]
        psis = psis[:-1]
        my_reconstruct = PeptideBuilder.make_structure(aa_seq, phis,psis)
        out = Bio.PDB.PDBIO()
        out.set_structure(my_reconstruct)
        out.save( save_dir)
def get_angles(pdb_path):
        pose = pose_from_pdb(pdb_path)
        r = pose.total_residue()

        omegas = [pose.omega(i) for i in range(1, r+1)]
        phis   = [pose.phi(i)   for i in range(1, r+1)]
        psis   = [pose.psi(i)   for i in range(1, r+1)]

        return omegas, phis, psis
# def get_calpha_distance_matrix(pdb_path):
#       structure = PDBParser(QUIET=True).get_structure(pdb_path[:-4], pdb_path)
#       A = []
#       for model in structure:
#       for chain in model:
#               for res in chain:
#               try:
#                       coord = res['CA'].get_coord()
#                       A.append(np.asarray(coord))
#               except:
#                       continue
#               if A: # first chain not empty
#               D = distance_matrix(A,A)
#               return D
def make_pdb_file(struct, file_nom):
    outfile = PDBIO()
    outfile.set_structure(struct)
    outfile.save(file_nom)
    return file_nom        

def compare_structure(reference, alternate):
    parser=PDBParser()

    ref_struct=parser.get_structure('Reference', reference)
    alt_struct= parser.get_structure("Alternate", alternate)


    ref_model=ref_struct[0]
    ref_chain=ref_model['A']

    alt_model=alt_struct[0]
    alt_chain=alt_model['A']

    ref_atoms=[]
    alt_atoms=[]

    for ref_res in ref_chain:
        if(ref_res.get_resname() in resdict.keys()):
            ref_atoms.append(ref_res['CA'])

    for alt_res in alt_chain:
        if(alt_res.get_resname() in resdict.keys()):
             alt_atoms.append(alt_res['CA'])

    super_imposer= Superimposer()
    super_imposer.set_atoms(ref_atoms, alt_atoms)
    super_imposer.apply(alt_model.get_atoms())

    path = "/Users/arjunsrivatsa/Desktop/ALT_final_reconstruct.pdb"
    make_pdb_file(alt_struct, path)

    full= super_imposer.rms


    return full
def get_rmsd(pdb1, pdb2): 
        pose1 = pose_from_pdb(pdb1)
        pose2 = pose_from_pdb(pdb2)
        return core.scoring.all_atom_rmsd(pose1, pose2)

import operator


def get_pdb_from_torsion(aa_seq,phis, psis, save_path): 
    #phis is of length n, disregard first one
    #psis is of length n, disregard the last one
    #aa_seq is a string of amino acids encoded by capital letters
    phis = phis[1:]
    psis = psis[:-1]
    my_reconstruct = PeptideBuilder.make_structure(aa_seq, phis,psis)
    out = Bio.PDB.PDBIO()
    out.set_structure(my_reconstruct)
    out.save( save_path)

def get_angles(pdb_path):
    pose = pose_from_pdb(pdb_path)
    r = pose.total_residue()

    omegas = [pose.omega(i) for i in range(1, r+1)]
    phis   = [pose.phi(i)   for i in range(1, r+1)]
    psis   = [pose.psi(i)   for i in range(1, r+1)]

    return omegas, phis, psis

def make_pdb_file(struct, file_nom):
    outfile = PDBIO()
    outfile.set_structure(struct)
    outfile.save(file_nom)
    return file_nom        

def compare_structure(reference, alternate):
    parser=PDBParser()

    ref_struct=parser.get_structure('Reference', reference)
    alt_struct= parser.get_structure("Alternate", alternate)


    ref_model=ref_struct[0]
    ref_chain=ref_model['A']

    alt_model=alt_struct[0]
    alt_chain=alt_model['A']

    ref_atoms=[]
    alt_atoms=[]

    for ref_res in ref_chain:
        if(ref_res.get_resname() in resdict.keys()):
            ref_atoms.append(ref_res['CA'])

    for alt_res in alt_chain:
        if(alt_res.get_resname() in resdict.keys()):
             alt_atoms.append(alt_res['CA'])

    super_imposer= Superimposer()
    super_imposer.set_atoms(ref_atoms, alt_atoms)
    super_imposer.apply(alt_model.get_atoms())

    #path = "/Users/arjunsrivatsa/Desktop/ALT_final_reconstruct.pdb"
    #make_pdb_file(alt_struct, path)

    full= super_imposer.rms

    return full

def get_rmsd(pdb1, pdb2): 
    pose1 = pose_from_pdb(pdb1)
    pose2 = pose_from_pdb(pdb2)
    return core.scoring.all_atom_rmsd(pose1, pose2)

def recon_rmsd(gt, aa_seq, phi, psi):
    get_pdb_from_torsion(aa_seq, phi, psi, save_path='tmp_recon.pdb')
    return compare_structure(gt, 'tmp_recon.pdb')

def load_data(path):
    data = pickle.load(open(path, 'rb'))
    pdbs = list(data.keys())
    lengths = []
    chains = []
    aa = []
    ss = []
    dcalphas = []
    coords = []
    psis = []
    phis = []
    pssms = []
    for p in pdbs:
        lengths.append(data[p]['length'])
        chains.append(data[p]['chain'])
        aa.append(data[p]['aa'])
        ss.append(data[p]['ss'])
        dcalphas.append(data[p]['dcalpha'])
        coords.append(data[p]['coords'])
        psis.append(data[p]['psi'])
        phis.append(data[p]['phi'])
        pssms.append(data[p]['pssm'])
    return pdbs, lengths, chains, aa, ss, dcalphas, coords, phis, psis, pssms

def calculate_dihedral_angles(atomic_coords, use_gpu):

    #assert int(atomic_coords.shape[1]) == 9
    #atomic_coords = atomic_coords.contiguous().view(-1,3)

    zero_tensor = torch.tensor(0.0)
    if use_gpu:
        zero_tensor = zero_tensor.cuda()

    dihedral_list = [zero_tensor,zero_tensor]
    dihedral_list.extend(compute_dihedral_list(atomic_coords))
    dihedral_list.append(zero_tensor)
    angles = torch.tensor(dihedral_list).view(-1,3)
    return angles

def compute_dihedral_list(atomic_coords):
    # atomic_coords is -1 x 3
    ba = atomic_coords[1:] - atomic_coords[:-1]
    ba /= ba.norm(dim=1).unsqueeze(1)
    ba_neg = -1 * ba

    n1_vec = torch.cross(ba[:-2], ba_neg[1:-1], dim=1)
    n2_vec = torch.cross(ba_neg[1:-1], ba[2:], dim=1)
    n1_vec /= n1_vec.norm(dim=1).unsqueeze(1)
    n2_vec /= n2_vec.norm(dim=1).unsqueeze(1)

    m1_vec = torch.cross(n1_vec, ba_neg[1:-1], dim=1)

    x = torch.sum(n1_vec*n2_vec,dim=1)
    y = torch.sum(m1_vec*n2_vec,dim=1)

    return torch.atan2(y,x)

def main():
        pdb_dir = 'data/pdb/'
        pdb_prefix = 'pdb'
        pdb_suffix = '.ent'
        
        datafile = 'testing.pkl'
        data = load_data(datafile)
        pdbs, lengths, chains, aas, q8s, dcalphas, coords, phis, psis, pssms = data
        
        # reconstruct from phis, psi
        for index in range(4,5):#len(pdbs)):
                pdb = pdbs[index]
                chain_index = chains[index]
                aa = aas[index]
                phi = phis[index][1:] # omit first None
                psi = psis[index][:-1] # omit last None
                
                coord = coords[index]
                angles = compute_dihedral_list(torch.from_numpy(np.array(coord)))
                set_trace()
                out = Bio.PDB.PDBIO()
                #pdb_path = pdb_dir + pdb_prefix + pdb + pdb_suffix
                #structure = Bio.PDB.PDBParser(QUIET=True).get_structure(pdb_path[:-4], pdb_path)
                #ppb = PPBuilder()
                #model = ppb.build_peptides(structure)
                
                #for chain in model
                
                #for chain in structure.get_chains():
                #       print(chain.get_id())
                #print(structure[0])
                out.set_structure(structure[0]['A'])
                out.save("chain_a.pdb")
                out.set_structure(structure[0]['B'])
                out.save("chain_b.pdb")
                
                print('pdb', pdb)
                #print('aa', aa)
                #print('phi', phi)
                #print('psi', psi)
                #print('coord',coord)
                
                recon_struct = PeptideBuilder.make_structure(aa, phi, psi)
                recon_pdb = Bio.PDB.PDBIO()
                recon_pdb.set_structure(recon_struct)
                recon_pdb.save('reconstruct.pdb')

main()
                
        
    
        
    #pred_path = 'predictions/35k_model_1_nocoords.pkl'
    #data_path = 'data/test.pkl'
    #data = pickle.load(open(data_path, 'rb'))
    #pred = pickle.load(open(pred_path, 'rb'))

    #idx, pid, l, aa_seq, ss_seq, msa = data

    #for (i, pdb) in enumerate(['2n64', '5j4a', '5fjl', '3jb5', '5fhy', '5jo9']):
    #    gt_pdb = 'data/35k/test/{}.clean.pdb'.format(pdb)
    #    _, phi, psi = get_angles(gt_pdb)
    #    rmsd = recon_rmsd(gt_pdb, aa_seq[i], phi, psi)
    #    print("[{}] RMSD: {}".format(pdb, rmsd))
        



# data_loc = '/Users/arjunsrivatsa/Downloads/train_fold_1.pkl'
# datafile = data_loc
# with open(datafile, 'rb') as f:
#       test_f =  pickle.load(f)
# indices, pdbs, length_aas, pdb_aas, q8s, dcalphas, phis, psis, msas = test_f
# idx = 0
# seq = pdb_aas[idx]
# phis = phis[idx]
# psis = psis[idx]
# get_pdb_from_torsion(seq, phis, psis)


#GROUND_TRUTH = '/Users/arjunsrivatsa/Downloads/casp12_split/train_data/5jo9.clean.pdb'
#o,ph,ps = get_angles(GROUND_TRUTH)
#get_pdb_from_torsion(aa[5], ph, ps)
#rmsd = compare_structure(GROUND_TRUTH, "/Users/arjunsrivatsa/Desktop/final_reconstruct.pdb")
#print(rmsd)
#rmsd_2 = get_rmsd("/Users/arjunsrivatsa/Desktop/ALT_final_reconstruct.pdb", GROUND_TRUTH)
#print(rmsd_2)
#IMPORTANT, for phis ignore the first for psis ignore the last in our data format
