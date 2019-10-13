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
from pdb import set_trace

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
init()
resdict = { 'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', \
        'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L', \
        'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', \
        'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y' }

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

def main():
    pred_path = 'predictions/35k_model_1_nocoords.pkl'
    data_path = 'data/test.pkl'
    data = pickle.load(open(data_path, 'rb'))
    pred = pickle.load(open(pred_path, 'rb'))

    idx, pid, l, aa_seq, ss_seq, msa = data

    for (i, pdb) in enumerate(['2n64', '5j4a', '5fjl', '3jb5', '5fhy', '5jo9']):
        gt_pdb = 'data/35k/test/{}.clean.pdb'.format(pdb)
        _, phi, psi = get_angles(gt_pdb)
        phi, psi = pred[pdb]['phi'], pred[pdb]['psi']
        rmsd = recon_rmsd(gt_pdb, aa_seq[i], phi, psi)
        print("[{}] RMSD: {}".format(pdb, rmsd))

if __name__=='__main__':
    main()

