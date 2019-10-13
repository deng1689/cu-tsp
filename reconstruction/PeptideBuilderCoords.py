'''
Modified to build full atom proteins from Calpha's
Iddo Drori, Columbia University
'''


'''This module is part of the PeptideBuilder library,
written by Matthew Z. Tien, Dariya K. Sydykova,
Austin G. Meyer, and Claus O. Wilke.
The PeptideBuilder module contains code to generate 3D
structures of peptides. It requires the Geometry module
(also part of the PeptideBuilder library), which contains
default bond lengths and angles for all amino acids.
This module also requires the Bio.PDB module from
Biopython, for structure manipulation.
This file is provided to you under the MIT License.'''

from __future__ import print_function
from Bio.PDB import *
from Bio.PDB.Atom import *
from Bio.PDB.Residue import *
from Bio.PDB.Chain import *
from Bio.PDB.Model import *
from Bio.PDB.Structure import *
from Bio.PDB.Vector import *
from Bio.PDB.Entity import*
import math, warnings
import numpy

'''This module is part of the PeptideBuilder library,
written by Matthew Z. Tien, Dariya K. Sydykova,
Austin G. Meyer, and Claus O. Wilke.
The Geometry module contains the default geometries of
all 20 amino acids. The main function to be used is the
geometry() function, which returns the default geometry
for the requested amino acid.
This file is provided to you under the MIT License.'''

import random

class Geo():
    '''Geometry base class'''
    def __repr__(self):
        repr = ""
        for var in dir(self):
            if var in self.__dict__: # exclude member functions, only print member variables
                repr += "%s = %s\n" % ( var, self.__dict__[var] )
        return repr


class GlyGeo(Geo):
    '''Geometry of Glycine'''
    def __init__(self):
        self.CA_N_length=1.46
        self.CA_C_length=1.52
        self.N_CA_C_angle=110.8914

        self.C_O_length=1.23
        self.CA_C_O_angle=120.5117
        self.N_CA_C_O_diangle= 180.0

        self.phi=-120
        self.psi_im1=140
        self.omega=180.0
        self.peptide_bond=1.33
        self.CA_C_N_angle =116.642992978143
        self.C_N_CA_angle= 121.382215820277

        self.residue_name= 'G'
    
class AlaGeo(Geo):
    '''Geometry of Alanin'''
    def __init__(self):
        self.CA_N_length=1.46
        self.CA_C_length=1.52
        self.N_CA_C_angle=111.068

        self.C_O_length=1.23
        self.CA_C_O_angle=120.5
        self.N_CA_C_O_diangle=-60.5

        self.phi=-120
        self.psi_im1=140
        self.omega=180.0
        self.peptide_bond=1.33
        self.CA_C_N_angle =116.642992978143
        self.C_N_CA_angle= 121.382215820277
    

        self.CA_CB_length=1.52
        self.C_CA_CB_angle=109.5
        self.N_C_CA_CB_diangle=122.6860

        self.residue_name= 'A'

class SerGeo(Geo):
    '''Geometry of Serine'''
    def __init__(self):
        self.CA_N_length=1.46
        self.CA_C_length=1.52
        self.N_CA_C_angle=111.2812

        self.C_O_length=1.23
        self.CA_C_O_angle=120.5
        self.N_CA_C_O_diangle= -60.0

        self.phi=-120
        self.psi_im1=140
        self.omega=180.0
        self.peptide_bond=1.33
        self.CA_C_N_angle =116.642992978143
        self.C_N_CA_angle= 121.382215820277
    

        self.CA_CB_length=1.52
        self.C_CA_CB_angle=109.5
        self.N_C_CA_CB_diangle=122.6618

        self.CB_OG_length=1.417
        self.CA_CB_OG_angle=110.773
        self.N_CA_CB_OG_diangle=-63.3

        self.residue_name= 'S'

    def inputRotamers(self, rotamers):
        try:
            self.N_CA_CB_OG_diangle=rotamers[0]
        except IndexError:
            print("Input Rotamers List: not long enough")
            self.N_CA_CB_OG_diangle=-63.3
            
        

class CysGeo(Geo):
    '''Geometry of Cystine'''
    def __init__(self):
        self.CA_N_length=1.46
        self.CA_C_length=1.52
        self.N_CA_C_angle= 110.8856      

        self.C_O_length=1.23
        self.CA_C_O_angle=120.5
        self.N_CA_C_O_diangle= -60.0

        self.phi=-120
        self.psi_im1=140
        self.omega=180.0
        self.peptide_bond=1.33
        self.CA_C_N_angle =116.642992978143
        self.C_N_CA_angle= 121.382215820277
    

        self.CA_CB_length=1.52
        self.C_CA_CB_angle=109.5
        self.N_C_CA_CB_diangle=122.5037

        self.CB_SG_length=1.808
        self.CA_CB_SG_angle=113.8169
        self.N_CA_CB_SG_diangle=-62.2

        self.residue_name= 'C'

    def inputRotamers(self, rotamers):
        try:
            self.N_CA_CB_SG_diangle=rotamers[0]
        except IndexError:
            print("Input Rotamers List: not long enough")
            self.N_CA_CB_SG_diangle=-62.2
            
class ValGeo(Geo):
    '''Geometry of Valine'''
    def __init__(self):
        self.CA_N_length=1.46
        self.CA_C_length=1.52
        self.N_CA_C_angle=109.7698
        
        self.C_O_length=1.23
        self.CA_C_O_angle=120.5686
        self.N_CA_C_O_diangle= -60.0

        self.phi=-120
        self.psi_im1=140
        self.omega=180.0
        self.peptide_bond=1.33
        self.CA_C_N_angle =116.642992978143
        self.C_N_CA_angle= 121.382215820277

        self.CA_CB_length=1.52
        self.C_CA_CB_angle=109.5
        self.N_C_CA_CB_diangle=123.2347

        self.CB_CG1_length=1.527
        self.CA_CB_CG1_angle=110.7
        self.N_CA_CB_CG1_diangle=177.2

        self.CB_CG2_length=1.527
        self.CA_CB_CG2_angle=110.4
        self.N_CA_CB_CG2_diangle=-63.3

        self.residue_name= 'V'

    def inputRotamers(self, rotamers):
        try:
            self.N_CA_CB_CG1_diangle=rotamers[0]
            self.N_CA_CB_CG2_diangle=rotamers[1]
        except IndexError:
            print("Input Rotamers List: not long enough")
            self.N_CA_CB_CG1_diangle=177.2
            self.N_CA_CB_CG2_dianlge=-63.3

class IleGeo(Geo):
    '''Geometry of Isoleucine'''
    def __init__(self):
        self.CA_N_length=1.46
        self.CA_C_length=1.52
        self.N_CA_C_angle=109.7202

        self.C_O_length=1.23
        self.CA_C_O_angle=120.5403
        self.N_CA_C_O_diangle= -60.0

        self.phi=-120
        self.psi_im1=140
        self.omega=180.0
        self.peptide_bond=1.33
        self.CA_C_N_angle =116.642992978143
        self.C_N_CA_angle= 121.382215820277

        self.CA_CB_length=1.52
        self.C_CA_CB_angle=109.5
        self.N_C_CA_CB_diangle=123.2347

        self.CB_CG1_length=1.527
        self.CA_CB_CG1_angle=110.7
        self.N_CA_CB_CG1_diangle=59.7

        self.CB_CG2_length=1.527
        self.CA_CB_CG2_angle=110.4
        self.N_CA_CB_CG2_diangle=-61.6

        self.CG1_CD1_length=1.52
        self.CB_CG1_CD1_angle=113.97
        self.CA_CB_CG1_CD1_diangle=169.8

        self.residue_name= 'I'

    def inputRotamers(self, rotamers):
        try:
            self.N_CA_CB_CG1_diangle=rotamers[0]
            self.N_CA_CB_CG2_diangle=rotamers[1]
            self.CA_CB_CG1_CD1_diangle=rotamers[2]
        except IndexError:
            print("Input Rotamers List: not long enough")
            self.N_CA_CB_CG1_diangle=-61.6
            self.N_CA_CB_CG2_diangle=59.7
            self.CA_CB_CG1_CD1_diangle=169.8
            

##    def generateRandomRotamers(self):
##        rotamer_bins=[-60, 60, 180]
##        tempList=[]
##        for i in range(0,3):
##            tempList.append(random.choice(rotamer_bins))
##        self.inputRotamers(tempList)
        

class LeuGeo(Geo):
    '''Geometry of Leucine'''
    def __init__(self):
        self.CA_N_length=1.46
        self.CA_C_length=1.52
        self.N_CA_C_angle=110.8652      

        self.C_O_length=1.23
        self.CA_C_O_angle=120.4647
        self.N_CA_C_O_diangle=120.0

        self.phi=-120
        self.psi_im1=140
        self.omega=180.0
        self.peptide_bond=1.33
        self.CA_C_N_angle =116.642992978143
        self.C_N_CA_angle= 121.382215820277

        self.CA_CB_length=1.52
        self.C_CA_CB_angle=109.5
        self.N_C_CA_CB_diangle=122.4948

        self.CB_CG_length=1.53
        self.CA_CB_CG_angle=116.10
        self.N_CA_CB_CG_diangle=-60.1

        self.CG_CD1_length=1.524
        self.CB_CG_CD1_angle=110.27
        self.CA_CB_CG_CD1_diangle=174.9

        self.CG_CD2_length=1.525
        self.CB_CG_CD2_angle=110.58
        self.CA_CB_CG_CD2_diangle=66.7

        self.residue_name= 'L'

    def inputRotamers(self, rotamers):
        try: 
            self.N_CA_CB_CG_diangle=rotamers[0]
            self.CA_CB_CG_CD1_diangle=rotamers[1]
            self.CA_CB_CG_CD2_diangle=rotamers[2]
        except IndexError:
            print("Input Rotamers List: not long enough")
            self.N_CA_CB_CG_diangle=-60.1
            self.CA_CB_CG_CD1_diangle=174.9
            self.CA_CB_CG_CD2_diangle=66.7
            
            

##    def generateRandomRotamers(self):
##        rotamer_bins=[-60, 60, 180]
##        tempList=[]
##        for i in range(0,3):
##            tempList.append(random.choice(rotamer_bins))
##        self.inputRotamers(tempList)


class ThrGeo(Geo):
    '''Geometry of Threonine'''
    def __init__(self):
        self.CA_N_length=1.46
        self.CA_C_length=1.52
        self.N_CA_C_angle=110.7014

        self.C_O_length=1.23
        self.CA_C_O_angle=120.5359
        self.N_CA_C_O_diangle=120.0

        self.phi=-120
        self.psi_im1=140
        self.omega=180.0
        self.peptide_bond=1.33
        self.CA_C_N_angle =116.642992978143
        self.C_N_CA_angle= 121.382215820277

        self.CA_CB_length=1.52
        self.C_CA_CB_angle=109.5
        self.N_C_CA_CB_diangle=123.0953

        self.CB_OG1_length=1.43
        self.CA_CB_OG1_angle=109.18
        self.N_CA_CB_OG1_diangle=60.0

        self.CB_CG2_length=1.53
        self.CA_CB_CG2_angle=111.13
        self.N_CA_CB_CG2_diangle=-60.3

        self.residue_name= 'T'

    def inputRotamers(self, rotamers):
        try:
            self.N_CA_CB_OG1_diangle=rotamers[0]
            self.N_CA_CB_OG2_diangle=rotamers[1]
        except IndexError:
            print("Input Rotamers List: not long enough")
            self.N_CA_CB_OG1_diangle=-60.3
            self.N_CA_CB_OG2_diangle=60.0

class ArgGeo(Geo):
    '''Geometry of Arginine'''
    def __init__(self):
        self.CA_N_length=1.46
        self.CA_C_length=1.52
        self.N_CA_C_angle=110.98 

        self.C_O_length=1.23
        self.CA_C_O_angle=120.54
        self.N_CA_C_O_diangle=120.0

        self.phi=-120
        self.psi_im1=140
        self.omega=180.0
        self.peptide_bond=1.33
        self.CA_C_N_angle =116.642992978143
        self.C_N_CA_angle= 121.382215820277

        self.CA_CB_length=1.52
        self.C_CA_CB_angle=109.5
        self.N_C_CA_CB_diangle=122.76

        self.CB_CG_length=1.52
        self.CA_CB_CG_angle=113.83
        self.N_CA_CB_CG_diangle=-65.2

        self.CG_CD_length=1.52
        self.CB_CG_CD_angle=111.79
        self.CA_CB_CG_CD_diangle=-179.2

        self.CD_NE_length=1.46
        self.CG_CD_NE_angle=111.68
        self.CB_CG_CD_NE_diangle=-179.3

        self.NE_CZ_length=1.33
        self.CD_NE_CZ_angle=124.79
        self.CG_CD_NE_CZ_diangle=-178.7

        self.CZ_NH1_length=1.33
        self.NE_CZ_NH1_angle=120.64
        self.CD_NE_CZ_NH1_diangle=0.0

        self.CZ_NH2_length=1.33
        self.NE_CZ_NH2_angle=119.63
        self.CD_NE_CZ_NH2_diangle=180.0

        self.residue_name= 'R'

    def inputRotamers(self, rotamers):
        try:
            self.N_CA_CB_CG_diangle=rotamers[0]
            self.CA_CB_CG_CD_diangle=rotamers[1]
            self.CB_CG_CD_NE_diangle=rotamers[2]
            self.CG_CD_NE_CZ_diangle=rotamers[3]
            self.CD_NE_CZ_NH1_diangle=rotamers[4]
            self.CD_NE_CZ_NH2_diangle=rotamers[5]
        except IndexError:
            print("Input Rotamers List: not long enough")
            self.N_CA_CB_CG_diangle=-65.2
            self.CA_CB_CG_CD_diangle=-179.2
            self.CB_CG_CD_NE_diangle=-179.3
            self.CG_CD_NE_CZ_diangle=-178.7
            self.CD_NE_CZ_NH1_diangle=0.0
            self.CD_NE_CZ_NH2_diangle=180.0

    def generateRandomRotamers(self):
        rotamer_bins=[-60, 60, 180]
        tempList=[]
        for i in range(0,6):
            tempList.append(random.choice(rotamer_bins))
        self.inputRotamers(tempList)

class LysGeo(Geo):
    '''Geometry of Lysine'''
    def __init__(self):
        self.CA_N_length=1.46
        self.CA_C_length=1.52
        self.N_CA_C_angle=111.08 

        self.C_O_length=1.23
        self.CA_C_O_angle=120.54
        self.N_CA_C_O_diangle=120.0

        self.phi=-120
        self.psi_im1=140
        self.omega=180.0
        self.peptide_bond=1.33
        self.CA_C_N_angle =116.642992978143
        self.C_N_CA_angle= 121.382215820277

        self.CA_CB_length=1.52
        self.C_CA_CB_angle=109.5
        self.N_C_CA_CB_diangle=122.76

        self.CB_CG_length=1.52
        self.CA_CB_CG_angle=113.83
        self.N_CA_CB_CG_diangle=-64.5

        self.CG_CD_length=1.52
        self.CB_CG_CD_angle=111.79
        self.CA_CB_CG_CD_diangle=-178.1

        self.CD_CE_length=1.46
        self.CG_CD_CE_angle=111.68
        self.CB_CG_CD_CE_diangle=-179.6

        self.CE_NZ_length=1.33
        self.CD_CE_NZ_angle=124.79
        self.CG_CD_CE_NZ_diangle=179.6

        self.residue_name= 'K'

    def inputRotamers(self, rotamers):
        try:
            self.N_CA_CB_CG_diangle=rotamers[0]
            self.CA_CB_CG_CD_diangle=rotamers[1]
            self.CB_CG_CD_CE_diangle=rotamers[2]
            self.CG_CD_CE_NZ_diangle=rotamers[3]
        except IndexError:
            print("Input Rotamers List: not long enough")
            self.N_CA_CB_CG_diangle=-64.5
            self.CA_CB_CG_CD_diangle=-178.1
            self.CB_CG_CD_CE_diangle=-179.6
            self.CG_CD_CE_NZ_diangle=179.6

    def generateRandomRotamers(self):
        rotamer_bins=[-60, 60, 180]
        tempList=[]
        for i in range(0,4):
            tempList.append(random.choice(rotamer_bins))
        self.inputRotamers(tempList)
        
class AspGeo(Geo):
    '''Geometry of Aspartic Acid'''
    def __init__(self):
        self.CA_N_length=1.46
        self.CA_C_length=1.52
        self.N_CA_C_angle=111.03     

        self.C_O_length=1.23
        self.CA_C_O_angle=120.51
        self.N_CA_C_O_diangle=120.0

        self.phi=-120
        self.psi_im1=140
        self.omega=180.0
        self.peptide_bond=1.33
        self.CA_C_N_angle =116.642992978143
        self.C_N_CA_angle= 121.382215820277
    
        self.CA_CB_length=1.52
        self.C_CA_CB_angle=109.5
        self.N_C_CA_CB_diangle=122.82

        self.CB_CG_length=1.52
        self.CA_CB_CG_angle=113.06
        self.N_CA_CB_CG_diangle=-66.4

        self.CG_OD1_length=1.25
        self.CB_CG_OD1_angle=119.22
        self.CA_CB_CG_OD1_diangle=-46.7

        self.CG_OD2_length=1.25
        self.CB_CG_OD2_angle=118.218
        self.CA_CB_CG_OD2_diangle=180+self.CA_CB_CG_OD1_diangle

        self.residue_name= 'D'

    def inputRotamers(self, rotamers):
        try:
            self.N_CA_CB_CG_diangle=rotamers[0]
            self.CA_CB_CG_OD1_diangle=rotamers[1]
            if (self.CA_CB_CG_OD1_diangle > 0):
                self.CA_CB_CG_OD2_diangle=rotamers[1]-180.0
            else:
                self.CA_CB_CG_OD2_diangle=rotamers[1]+180.0
        except IndexError:
            print("Input Rotamers List: not long enough")
            self.N_CA_CB_CG_diangle=-66.4
            self.CA_CB_CG_OD1_diangle=-46.7
            self.CA_CB_CG_OD2_diangle=180+self.CA_CB_CG_OD1_diangle
        

class AsnGeo(Geo):
    '''Geometry of Asparagine'''
    def __init__(self):
        self.CA_N_length=1.46
        self.CA_C_length=1.52
        self.N_CA_C_angle=111.5

        self.C_O_length=1.23
        self.CA_C_O_angle=120.4826
        self.N_CA_C_O_diangle= -60.0

        self.phi=-120
        self.psi_im1=140
        self.omega=180.0
        self.peptide_bond=1.33
        self.CA_C_N_angle =116.642992978143
        self.C_N_CA_angle= 121.382215820277
        
        self.CA_CB_length=1.52
        self.C_CA_CB_angle=109.5
        self.N_C_CA_CB_diangle=123.2254

        self.CB_CG_length=1.52
        self.CA_CB_CG_angle=112.62
        self.N_CA_CB_CG_diangle=-65.5

        self.CG_OD1_length=1.23
        self.CB_CG_OD1_angle=120.85
        self.CA_CB_CG_OD1_diangle=-58.3

        self.CG_ND2_length=1.33
        self.CB_CG_ND2_angle=116.48
        self.CA_CB_CG_ND2_diangle=180.0+self.CA_CB_CG_OD1_diangle

        self.residue_name= 'N'

    def inputRotamers(self, rotamers):
        try:
            self.N_CA_CB_CG_diangle=rotamers[0]
            self.CA_CB_CG_OD1_diangle=rotamers[1]
            if (self.CA_CB_CG_OD1_diangle > 0):
                self.CA_CB_CG_ND2_diangle = rotamers[1]-180.0
            else:
                self.CA_CB_CG_ND2_diangle = rotamers[1]+180.0
        except IndexError:
            print("Input Rotamers List: not long enough")
            self.N_CA_CB_CG_diangle=-65.5
            self.CA_CB_CG_OD1_diangle=-58.3
            self.CA_CB_CG_ND2_diangle=180.0+self.CA_CB_CG_OD1_diangle

class GluGeo(Geo):
    '''Geometry of Glutamic Acid'''
    def __init__(self):
        self.CA_N_length=1.46
        self.CA_C_length=1.52
        self.N_CA_C_angle=111.1703

        self.C_O_length=1.23
        self.CA_C_O_angle=120.511
        self.N_CA_C_O_diangle=120.0

        self.phi=-120
        self.psi_im1=140
        self.omega=180.0
        self.peptide_bond=1.33
        self.CA_C_N_angle =116.642992978143
        self.C_N_CA_angle= 121.382215820277

        self.CA_CB_length=1.52
        self.C_CA_CB_angle=109.5
        self.N_C_CA_CB_diangle=122.8702

        self.CB_CG_length=1.52
        self.CA_CB_CG_angle=113.82
        self.N_CA_CB_CG_diangle=-63.8

        self.CG_CD_length=1.52
        self.CB_CG_CD_angle=113.31
        self.CA_CB_CG_CD_diangle=-179.8

        self.CD_OE1_length=1.25
        self.CG_CD_OE1_angle=119.02
        self.CB_CG_CD_OE1_diangle=-6.2

        self.CD_OE2_length=1.25
        self.CG_CD_OE2_angle=118.08    
        self.CB_CG_CD_OE2_diangle=180.0+self.CB_CG_CD_OE1_diangle

        self.residue_name= 'E'

    def inputRotamers(self, rotamers):
        try:
            self.N_CA_CB_CG_diangle=rotamers[0]
            self.CA_CB_CG_CD_diangle=rotamers[1]
            self.CB_CG_CD_OE1_diangle=rotamers[2]
            if (self.CB_CG_CD_OE1_diangle > 0):
                self.CB_CG_CD_OE2_diangle = rotamers[2]-180.0
            else:
                self.CB_CG_CD_OE2_diangle = rotamers[2]+180.0
        except IndexError:
            print("Input Rotamers List: not long enough")
            self.N_CA_CB_CG_diangle=-63.8
            self.CA_CB_CG_CD_diangle=-179.8
            self.CB_CG_CD_OE1_diangle=-6.2
            self.CB_CG_CD_OE2_diangle=180.0+self.CB_CG_CD_OE1_diangle

    def generateRandomRotamers(self):
        rotamer_bins=[-60, 60, 180]
        tempList=[]
        for i in range(0,3):
            tempList.append(random.choice(rotamer_bins))
        self.inputRotamers(tempList)

class GlnGeo(Geo):
    '''Geometry of Glutamine'''
    def __init__(self):
        self.CA_N_length=1.46
        self.CA_C_length=1.52
        self.N_CA_C_angle=111.0849

        self.C_O_length=1.23
        self.CA_C_O_angle=120.5029
        self.N_CA_C_O_diangle=120.0

        self.phi=-120
        self.psi_im1=140
        self.omega=180.0
        self.peptide_bond=1.33
        self.CA_C_N_angle =116.642992978143
        self.C_N_CA_angle= 121.382215820277

        self.CA_CB_length=1.52
        self.C_CA_CB_angle=109.5
        self.N_C_CA_CB_diangle=122.8134

        self.CB_CG_length=1.52
        self.CA_CB_CG_angle=113.75
        self.N_CA_CB_CG_diangle=-60.2

        self.CG_CD_length=1.52
        self.CB_CG_CD_angle=112.78
        self.CA_CB_CG_CD_diangle=-69.6

        self.CD_OE1_length=1.24
        self.CG_CD_OE1_angle=120.86
        self.CB_CG_CD_OE1_diangle=-50.5

        self.CD_NE2_length=1.33
        self.CG_CD_NE2_angle=116.50
        self.CB_CG_CD_NE2_diangle=180+self.CB_CG_CD_OE1_diangle

        self.residue_name= 'Q'

    def inputRotamers(self, rotamers):
        try:
            self.N_CA_CB_CG_diangle=rotamers[0]
            self.CA_CB_CG_CD_diangle=rotamers[1]
            self.CB_CG_CD_OE1_diangle=rotamers[2]
            if (self.CB_CG_CD_OE1_diangle > 0):
                self.CB_CG_CD_NE2_diangle = rotamers[2]-180.0
            else:
                self.CB_CG_CD_NE2_diangle = rotamers[2]+180.0
        except IndexError:
            print("Input Rotamers List: not long enough")
            self.N_CA_CB_CG_diangle=-60.2
            self.CA_CB_CG_CD_diangle=-69.6
            self.CB_CG_CD_OE1_diangle=-50.5
            self.CB_CG_CD_NE2_diangle=180+self.CB_CG_CD_OE1_diangle
    
    def generateRandomRotamers(self):
        rotamer_bins=[-60, 60, 180]
        tempList=[]
        for i in range(0,3):
            tempList.append(random.choice(rotamer_bins))
        self.inputRotamers(tempList)

class MetGeo(Geo):
    '''Geometry of Methionine'''
    def __init__(self):
        self.CA_N_length=1.46
        self.CA_C_length=1.52
        self.N_CA_C_angle=110.9416

        self.C_O_length=1.23
        self.CA_C_O_angle=120.4816
        self.N_CA_C_O_diangle=120.0

        self.phi=-120
        self.psi_im1=140
        self.omega=180.0
        self.peptide_bond=1.33
        self.CA_C_N_angle =116.642992978143
        self.C_N_CA_angle= 121.382215820277

        self.CA_CB_length=1.52
        self.C_CA_CB_angle=109.5
        self.N_C_CA_CB_diangle=122.6733

        self.CB_CG_length=1.52
        self.CA_CB_CG_angle= 113.68
        self.N_CA_CB_CG_diangle=-64.4

        self.CG_SD_length=1.81
        self.CB_CG_SD_angle=112.69
        self.CA_CB_CG_SD_diangle=-179.6

        self.SD_CE_length=1.79
        self.CG_SD_CE_angle=100.61
        self.CB_CG_SD_CE_diangle=70.1

        self.residue_name= 'M'
    def inputRotamers(self, rotamer):
        try:
            self.N_CA_CB_CG_diangle=rotamer[0]
            self.CA_CB_CG_SD_diangle=rotamer[1]
            self.CB_CG_SD_CE_diangle=rotamer[2]
        except IndexError:
            print("Input Rotamers List: not long enough")
            self.N_CA_CB_CG_diangle=-64.4
            self.CA_CB_CG_SD_diangle=-179.6
            self.CB_CG_SD_CE_diangle=70.1
    
    def generateRandomRotamers(self):
        rotamer_bins=[-60, 60, 180]
        tempList=[]
        for i in range(0,3):
            tempList.append(random.choice(rotamer_bins))
        self.inputRotamers(tempList)

class HisGeo(Geo):
    '''Geometry of Histidine'''
    def __init__(self):
        self.CA_N_length=1.46
        self.CA_C_length=1.52
        self.N_CA_C_angle=111.0859

        self.C_O_length=1.23
        self.CA_C_O_angle=120.4732
        self.N_CA_C_O_diangle=120.0

        self.phi=-120
        self.psi_im1=140
        self.omega=180.0
        self.peptide_bond=1.33
        self.CA_C_N_angle =116.642992978143
        self.C_N_CA_angle= 121.382215820277

        self.CA_CB_length=1.52
        self.C_CA_CB_angle=109.5
        self.N_C_CA_CB_diangle=122.6711

        self.CB_CG_length=1.49
        self.CA_CB_CG_angle=113.74
        self.N_CA_CB_CG_diangle=-63.2
        
        self.CG_ND1_length=1.38
        self.CB_CG_ND1_angle=122.85
        self.CA_CB_CG_ND1_diangle=-75.7          

        self.CG_CD2_length=1.35
        self.CB_CG_CD2_angle=130.61
        self.CA_CB_CG_CD2_diangle=180.0+self.CA_CB_CG_ND1_diangle

        self.ND1_CE1_length=1.32
        self.CG_ND1_CE1_angle=108.5
        self.CB_CG_ND1_CE1_diangle=180.0

        self.CD2_NE2_length=1.35
        self.CG_CD2_NE2_angle=108.5
        self.CB_CG_CD2_NE2_diangle=180.0

        self.residue_name= 'H'

    def inputRotamers(self, rotamers):
        try:
            self.N_CA_CB_CG_diangle=rotamers[0]
            self.CA_CB_CG_ND1_diangle=rotamers[1]
            if (self.CA_CB_CG_ND1_diangle> 0):
                self.CA_CB_CG_CD2_diangle = rotamers[1]-180.0
            else:
                self.CA_CB_CG_CD2_diangle = rotamers[1]+180.0
        except IndexError:
            print("Input Rotamers List: not long enough")
            self.N_CA_CB_CG_diangle=-63.2
            self.CA_CB_CG_ND1_diangle=-75.7
            self.CA_CB_CG_CD2_diangle=180.0+self.CA_CB_CG_ND1_diangle

class ProGeo(Geo):
    '''Geometry of Proline'''
    def __init__(self):
        self.CA_N_length=1.46
        self.CA_C_length=1.52
        self.N_CA_C_angle=112.7499

        self.C_O_length=1.23
        self.CA_C_O_angle=120.2945
        self.N_CA_C_O_diangle=-45.0

        self.phi=-120
        self.psi_im1=140
        self.omega=180.0
        self.peptide_bond=1.33
        self.CA_C_N_angle =116.642992978143
        self.C_N_CA_angle= 121.382215820277

        self.CA_CB_length=1.52
        self.C_CA_CB_angle=109.5
        self.N_C_CA_CB_diangle=115.2975

        self.CB_CG_length=1.49
        self.CA_CB_CG_angle=104.21
        self.N_CA_CB_CG_diangle=29.6

        self.CG_CD_length=1.50
        self.CB_CG_CD_angle=105.03
        self.CA_CB_CG_CD_diangle=-34.8

        self.residue_name= 'P'

class PheGeo(Geo):
    '''Geometry of Phenylalanine'''
    def __init__(self):
        self.CA_N_length=1.46
        self.CA_C_length=1.52
        self.N_CA_C_angle=110.7528

        self.C_O_length=1.23
        self.CA_C_O_angle=120.5316
        self.N_CA_C_O_diangle=120.0

        self.phi=-120
        self.psi_im1=140
        self.omega=180.0
        self.peptide_bond=1.33
        self.CA_C_N_angle =116.642992978143
        self.C_N_CA_angle= 121.382215820277

        self.CA_CB_length=1.52
        self.C_CA_CB_angle=109.5
        self.N_C_CA_CB_diangle=122.6054

        self.CB_CG_length=1.50
        self.CA_CB_CG_angle=113.85
        self.N_CA_CB_CG_diangle=-64.7

        self.CG_CD1_length=1.39
        self.CB_CG_CD1_angle=120.0
        self.CA_CB_CG_CD1_diangle=93.3

        self.CG_CD2_length=1.39
        self.CB_CG_CD2_angle=120.0
        self.CA_CB_CG_CD2_diangle=self.CA_CB_CG_CD1_diangle-180.0

        self.CD1_CE1_length=1.39
        self.CG_CD1_CE1_angle=120.0
        self.CB_CG_CD1_CE1_diangle=180.0

        self.CD2_CE2_length=1.39
        self.CG_CD2_CE2_angle=120.0
        self.CB_CG_CD2_CE2_diangle=180.0

        self.CE1_CZ_length=1.39
        self.CD1_CE1_CZ_angle=120.0
        self.CG_CD1_CE1_CZ_diangle=0.0

        self.residue_name= 'F'

    def inputRotamers(self, rotamers):
        try:
            self.N_CA_CB_CG_diangle=rotamers[0]
            self.CA_CB_CG_CD1_diangle=rotamers[1]
            if (self.CA_CB_CG_CD1_diangle>0):
                self.CA_CB_CG_CD2_diangle = rotamers[1]-180.0
            else:
                self.CA_CB_CG_CD2_diangle = rotamers[1]+180.0
        except IndexError:
            print("Input Rotamers List: not long enough")
            self.N_CA_CB_CG_diangle=-64.7
            self.CA_CB_CG_CD1_diangle=93.3
            self.CA_CB_CG_CD2_diangle=self.CA_CB_CG_CD1_diangle-180.0



class TyrGeo(Geo):
    '''Geometry of Tyrosine'''
    def __init__(self):
        self.CA_N_length=1.46
        self.CA_C_length=1.52
        self.N_CA_C_angle=110.9288

        self.C_O_length=1.23
        self.CA_C_O_angle=120.5434
        self.N_CA_C_O_diangle=120.0

        self.phi=-120
        self.psi_im1=140
        self.omega=180.0
        self.peptide_bond=1.33
        self.CA_C_N_angle =116.642992978143
        self.C_N_CA_angle= 121.382215820277

        self.CA_CB_length=1.52
        self.C_CA_CB_angle=109.5
        self.N_C_CA_CB_diangle=122.6023

        self.CB_CG_length=1.51
        self.CA_CB_CG_angle= 113.8
        self.N_CA_CB_CG_diangle=-64.3

        self.CG_CD1_length=1.39
        self.CB_CG_CD1_angle=120.98
        self.CA_CB_CG_CD1_diangle=93.1

        self.CG_CD2_length=1.39
        self.CB_CG_CD2_angle=120.82
        self.CA_CB_CG_CD2_diangle=self.CA_CB_CG_CD1_diangle+180.0

        self.CD1_CE1_length=1.39
        self.CG_CD1_CE1_angle=120.0
        self.CB_CG_CD1_CE1_diangle=180.0

        self.CD2_CE2_length=1.39
        self.CG_CD2_CE2_angle=120.0
        self.CB_CG_CD2_CE2_diangle=180.0

        self.CE1_CZ_length=1.39
        self.CD1_CE1_CZ_angle=120.0
        self.CG_CD1_CE1_CZ_diangle=0.0

        self.CZ_OH_length=1.39
        self.CE1_CZ_OH_angle=119.78
        self.CD1_CE1_CZ_OH_diangle=180.0

        self.residue_name= 'Y'

    def inputRotamers(self, rotamers):
        try:
            self.N_CA_CB_CG_diangle=rotamers[0]
            self.CA_CB_CG_CD1_diangle=rotamers[1]
            if (self.CA_CB_CG_CD1_diangle>0):
                self.CA_CB_CG_CD2_diangle = rotamers[1]-180.0
            else:
                self.CA_CB_CG_CD2_diangle = rotamers[1]+180.0
        except IndexError:
            print("Input Rotamers List: not long enough")
            self.N_CA_CB_CG_diangle=-64.3
            self.CA_CB_CG_CD1_diangle=93.1
            self.CA_CB_CG_CD2_diangle=self.CA_CB_CG_CD1_diangle+180.0
            

class TrpGeo(Geo):
    '''Geometry of Tryptophan'''
    def __init__(self):
        self.CA_N_length=1.46
        self.CA_C_length=1.52
        self.N_CA_C_angle=110.8914

        self.C_O_length=1.23
        self.CA_C_O_angle=120.5117
        self.N_CA_C_O_diangle=120.0

        self.phi=-120
        self.psi_im1=140
        self.omega=180.0
        self.peptide_bond=1.33
        self.CA_C_N_angle =116.642992978143
        self.C_N_CA_angle= 121.382215820277

        self.CA_CB_length=1.52
        self.C_CA_CB_angle=109.5
        self.N_C_CA_CB_diangle=122.6112

        self.CB_CG_length=1.50
        self.CA_CB_CG_angle=114.10
        self.N_CA_CB_CG_diangle=-66.4

        self.CG_CD1_length=1.37
        self.CB_CG_CD1_angle=127.07
        self.CA_CB_CG_CD1_diangle=96.3

        self.CG_CD2_length=1.43
        self.CB_CG_CD2_angle=126.66
        self.CA_CB_CG_CD2_diangle=self.CA_CB_CG_CD1_diangle-180.0

        self.CD1_NE1_length=1.38
        self.CG_CD1_NE1_angle=108.5
        self.CB_CG_CD1_NE1_diangle=180.0

        self.CD2_CE2_length=1.40
        self.CG_CD2_CE2_angle=108.5
        self.CB_CG_CD2_CE2_diangle=180.0

        self.CD2_CE3_length=1.40
        self.CG_CD2_CE3_angle=133.83
        self.CB_CG_CD2_CE3_diangle=0.0

        self.CE2_CZ2_length=1.40
        self.CD2_CE2_CZ2_angle=120.0
        self.CG_CD2_CE2_CZ2_diangle=180.0

        self.CE3_CZ3_length=1.40
        self.CD2_CE3_CZ3_angle=120.0
        self.CG_CD2_CE3_CZ3_diangle=180.0

        self.CZ2_CH2_length=1.40
        self.CE2_CZ2_CH2_angle=120.0
        self.CD2_CE2_CZ2_CH2_diangle=0.0

        self.residue_name= 'W'

    def inputRotamers(self, rotamers):
        try:
            self.N_CA_CB_CG_diangle=rotamers[0]
            self.CA_CB_CG_CD1_diangle=rotamers[1]
            if (self.CA_CB_CG_CD1_diangle>0):
                self.CA_CB_CG_CD2_diangle = rotamers[1]-180.0
            else:
                self.CA_CB_CG_CD2_diangle = rotamers[1]+180.0
        except IndexError:
            print("Input Rotamers List: not long enough")
            self.N_CA_CB_CG_diangle=-66.4
            self.CA_CB_CG_CD1_diangle=96.3
            self.CA_CB_CG_CD2_diangle=self.CA_CB_CG_CD1_diangle-180.0

def geometry(AA):
    '''Generates the geometry of the requested amino acid.
    The amino acid needs to be specified by its single-letter
    code. If an invalid code is specified, the function
    returns the geometry of Glycine.'''
    if(AA=='G'):
        return GlyGeo()
    elif(AA=='A'):
        return AlaGeo()
    elif(AA=='S'):
        return SerGeo()
    elif(AA=='C'):
        return CysGeo()
    elif(AA=='V'):
        return ValGeo()
    elif(AA=='I'):
        return IleGeo()
    elif(AA=='L'):
        return LeuGeo()
    elif(AA=='T'):
        return ThrGeo()
    elif(AA=='R'):
        return ArgGeo()
    elif(AA=='K'):
        return LysGeo()
    elif(AA=='D'):
        return AspGeo()
    elif(AA=='E'):
        return GluGeo()
    elif(AA=='N'):
        return AsnGeo()
    elif(AA=='Q'):
        return GlnGeo()
    elif(AA=='M'):
        return MetGeo()
    elif(AA=='H'):
        return HisGeo()
    elif(AA=='P'):
        return ProGeo()
    elif(AA=='F'):
        return PheGeo()
    elif(AA=='Y'):
        return TyrGeo()
    elif(AA=='W'):
        return TrpGeo()
    else:
        return GlyGeo()

def get_prop(atm):
    print(atm.get_name())
    print(atm.get_coord())
    print(atm.get_vector())
    print(atm.get_bfactor())
    print(atm.get_anisou())
    print(atm.get_occupancy())
    print(atm.get_altloc())
    print(atm.get_fullname())
    print(atm.get_serial_number())
    print(atm.get_parent())
    print(atm.get_id())
    print(atm.get_full_id())
    print(atm.get_level())

def calculateCoordinates(refA, refB, refC, L, ang, di):
    AV=refA.get_vector()
    BV=refB.get_vector()
    CV=refC.get_vector()
    
    CA=AV-CV
    CB=BV-CV

    ##CA vector
    AX=CA[0]
    AY=CA[1]
    AZ=CA[2]

    ##CB vector
    BX=CB[0]
    BY=CB[1]
    BZ=CB[2]

    ##Plane Parameters
    A=(AY*BZ)-(AZ*BY)
    B=(AZ*BX)-(AX*BZ)
    G=(AX*BY)-(AY*BX)

    ##Dot Product Constant
    F= math.sqrt(BX*BX + BY*BY + BZ*BZ) * L * math.cos(ang*(math.pi/180.0))

    ##Constants
    const=math.sqrt( math.pow((B*BZ-BY*G),2) *(-(F*F)*(A*A+B*B+G*G)+(B*B*(BX*BX+BZ*BZ) + A*A*(BY*BY+BZ*BZ)- (2*A*BX*BZ*G) + (BX*BX+ BY*BY)*G*G - (2*B*BY)*(A*BX+BZ*G))*L*L))
    denom= (B*B)*(BX*BX+BZ*BZ)+ (A*A)*(BY*BY+BZ*BZ) - (2*A*BX*BZ*G) + (BX*BX+BY*BY)*(G*G) - (2*B*BY)*(A*BX+BZ*G)

    X= ((B*B*BX*F)-(A*B*BY*F)+(F*G)*(-A*BZ+BX*G)+const)/denom

    if((B==0 or BZ==0) and (BY==0 or G==0)):
        const1=math.sqrt( G*G*(-A*A*X*X+(B*B+G*G)*(L-X)*(L+X)))
        Y= ((-A*B*X)+const1)/(B*B+G*G)
        Z= -(A*G*G*X+B*const1)/(G*(B*B+G*G))
    else:
        Y= ((A*A*BY*F)*(B*BZ-BY*G)+ G*( -F*math.pow(B*BZ-BY*G,2) + BX*const) - A*( B*B*BX*BZ*F- B*BX*BY*F*G + BZ*const)) / ((B*BZ-BY*G)*denom)
        Z= ((A*A*BZ*F)*(B*BZ-BY*G) + (B*F)*math.pow(B*BZ-BY*G,2) + (A*BX*F*G)*(-B*BZ+BY*G) - B*BX*const + A*BY*const) / ((B*BZ-BY*G)*denom)

    
    #GET THE NEW VECTOR from the orgin
    D=Vector(X, Y, Z) + CV
    with warnings.catch_warnings():
        # ignore inconsequential warning
        warnings.simplefilter("ignore")
        temp=calc_dihedral(AV, BV, CV, D)*(180.0/math.pi)
    
  
    di=di-temp
    rot= rotaxis(math.pi*(di/180.0), CV-BV)
    D=(D-BV).left_multiply(rot)+BV
    
    return D.get_array()

def makeGly(segID, N, CA, C, O, geo):
    '''Creates a Glycine residue'''
    ##Create Residue Data Structure
    res= Residue((' ', segID, ' '), "GLY", '    ')

    res.add(N)
    res.add(CA)
    res.add(C)
    res.add(O)

    ##print(res)
    return res

def makeAla(segID, N, CA, C, O, geo):
    '''Creates an Alanine residue'''
    ##R-Group
    CA_CB_length=geo.CA_CB_length
    C_CA_CB_angle=geo.C_CA_CB_angle
    N_C_CA_CB_diangle=geo.N_C_CA_CB_diangle

    carbon_b= calculateCoordinates(N, C, CA, CA_CB_length, C_CA_CB_angle, N_C_CA_CB_diangle)
    CB= Atom("CB", carbon_b, 0.0 , 1.0, " "," CB", 0,"C")

    ##Create Residue Data Structure
    res = Residue((' ', segID, ' '), "ALA", '    ')
    res.add(N)
    res.add(CA)
    res.add(C)
    res.add(O)
    res.add(CB)
    return res

def makeSer(segID, N, CA, C, O, geo):
    '''Creates a Serine residue'''
    ##R-Group
    CA_CB_length=geo.CA_CB_length
    C_CA_CB_angle=geo.C_CA_CB_angle
    N_C_CA_CB_diangle=geo.N_C_CA_CB_diangle

    CB_OG_length=geo.CB_OG_length
    CA_CB_OG_angle=geo.CA_CB_OG_angle
    N_CA_CB_OG_diangle=geo.N_CA_CB_OG_diangle
    
    carbon_b= calculateCoordinates(N, C, CA, CA_CB_length, C_CA_CB_angle, N_C_CA_CB_diangle)
    CB= Atom("CB", carbon_b, 0.0 , 1.0, " "," CB", 0,"C")
    oxygen_g= calculateCoordinates(N, CA, CB, CB_OG_length, CA_CB_OG_angle, N_CA_CB_OG_diangle)
    OG= Atom("OG", oxygen_g, 0.0, 1.0, " ", " OG", 0, "O")

    ##Create Reside Data Structure
    res= Residue((' ', segID, ' '), "SER", '    ')
    res.add(N)
    res.add(CA)
    res.add(C)
    res.add(O)
    res.add(CB)
    res.add(OG)

    ##print(res)
    return res

def makeCys(segID, N, CA, C, O, geo):
    '''Creates a Cysteine residue'''
    ##R-Group
    CA_CB_length=geo.CA_CB_length
    C_CA_CB_angle=geo.C_CA_CB_angle
    N_C_CA_CB_diangle=geo.N_C_CA_CB_diangle
    
    CB_SG_length= geo.CB_SG_length
    CA_CB_SG_angle= geo.CA_CB_SG_angle
    N_CA_CB_SG_diangle= geo.N_CA_CB_SG_diangle

    carbon_b= calculateCoordinates(N, C, CA, CA_CB_length, C_CA_CB_angle, N_C_CA_CB_diangle)
    CB= Atom("CB", carbon_b, 0.0 , 1.0, " "," CB", 0,"C")
    sulfur_g= calculateCoordinates(N, CA, CB, CB_SG_length, CA_CB_SG_angle, N_CA_CB_SG_diangle)
    SG= Atom("SG", sulfur_g, 0.0, 1.0, " ", " SG", 0, "S")

    ##Create Residue Data Structure
    res= Residue((' ', segID, ' '), "CYS", '    ')
    res.add(N)
    res.add(CA)
    res.add(C)
    res.add(O)
    res.add(CB)
    res.add(SG)
    return res

def makeVal(segID, N, CA, C, O, geo):
    '''Creates a Valine residue'''
    ##R-Group
    CA_CB_length=geo.CA_CB_length
    C_CA_CB_angle=geo.C_CA_CB_angle
    N_C_CA_CB_diangle=geo.N_C_CA_CB_diangle
    
    CB_CG1_length=geo.CB_CG1_length
    CA_CB_CG1_angle=geo.CA_CB_CG1_angle
    N_CA_CB_CG1_diangle=geo.N_CA_CB_CG1_diangle
    
    CB_CG2_length=geo.CB_CG2_length
    CA_CB_CG2_angle=geo.CA_CB_CG2_angle
    N_CA_CB_CG2_diangle=geo.N_CA_CB_CG2_diangle

    carbon_b= calculateCoordinates(N, C, CA, CA_CB_length, C_CA_CB_angle, N_C_CA_CB_diangle)
    CB= Atom("CB", carbon_b, 0.0 , 1.0, " "," CB", 0,"C")
    carbon_g1= calculateCoordinates(N, CA, CB, CB_CG1_length, CA_CB_CG1_angle, N_CA_CB_CG1_diangle)
    CG1= Atom("CG1", carbon_g1, 0.0, 1.0, " ", " CG1", 0, "C")
    carbon_g2= calculateCoordinates(N, CA, CB, CB_CG2_length, CA_CB_CG2_angle, N_CA_CB_CG2_diangle)
    CG2= Atom("CG2", carbon_g2, 0.0, 1.0, " ", " CG2", 0, "C")

    ##Create Residue Data Structure
    res= Residue((' ', segID, ' '), "VAL", '    ')
    res.add(N)
    res.add(CA)
    res.add(C)
    res.add(O)
    res.add(CB)
    res.add(CG1)
    res.add(CG2)
    return res

def makeIle(segID, N, CA, C, O, geo):
    '''Creates an Isoleucine residue'''
    ##R-group
    CA_CB_length=geo.CA_CB_length
    C_CA_CB_angle=geo.C_CA_CB_angle
    N_C_CA_CB_diangle=geo.N_C_CA_CB_diangle
    
    CB_CG1_length=geo.CB_CG1_length
    CA_CB_CG1_angle=geo.CA_CB_CG1_angle
    N_CA_CB_CG1_diangle=geo.N_CA_CB_CG1_diangle 
    
    CB_CG2_length=geo.CB_CG2_length
    CA_CB_CG2_angle=geo.CA_CB_CG2_angle
    N_CA_CB_CG2_diangle= geo.N_CA_CB_CG2_diangle

    CG1_CD1_length= geo.CG1_CD1_length
    CB_CG1_CD1_angle= geo.CB_CG1_CD1_angle
    CA_CB_CG1_CD1_diangle= geo.CA_CB_CG1_CD1_diangle
    
    carbon_b= calculateCoordinates(N, C, CA, CA_CB_length, C_CA_CB_angle, N_C_CA_CB_diangle)
    CB= Atom("CB", carbon_b, 0.0 , 1.0, " "," CB", 0,"C")
    carbon_g1= calculateCoordinates(N, CA, CB, CB_CG1_length, CA_CB_CG1_angle, N_CA_CB_CG1_diangle)
    CG1= Atom("CG1", carbon_g1, 0.0, 1.0, " ", " CG1", 0, "C")
    carbon_g2= calculateCoordinates(N, CA, CB, CB_CG2_length, CA_CB_CG2_angle, N_CA_CB_CG2_diangle)
    CG2= Atom("CG2", carbon_g2, 0.0, 1.0, " ", " CG2", 0, "C")
    carbon_d1= calculateCoordinates(CA, CB, CG1, CG1_CD1_length, CB_CG1_CD1_angle, CA_CB_CG1_CD1_diangle)
    CD1= Atom("CD1", carbon_d1, 0.0, 1.0, " ", " CD1", 0, "C")

    ##Create Residue Data Structure
    res= Residue((' ', segID, ' '), "ILE", '    ')
    res.add(N)
    res.add(CA)
    res.add(C)
    res.add(O)
    res.add(CB)
    res.add(CG1)
    res.add(CG2)
    res.add(CD1)
    return res

def makeLeu(segID, N, CA, C, O, geo):
    '''Creates a Leucine residue'''
    ##R-Group
    CA_CB_length=geo.CA_CB_length
    C_CA_CB_angle=geo.C_CA_CB_angle
    N_C_CA_CB_diangle=geo.N_C_CA_CB_diangle

    CB_CG_length=geo.CB_CG_length
    CA_CB_CG_angle= geo.CA_CB_CG_angle
    N_CA_CB_CG_diangle=geo.N_CA_CB_CG_diangle
    
    CG_CD1_length=geo.CG_CD1_length
    CB_CG_CD1_angle=geo.CB_CG_CD1_angle
    CA_CB_CG_CD1_diangle=geo.CA_CB_CG_CD1_diangle

    CG_CD2_length=geo.CG_CD2_length
    CB_CG_CD2_angle=geo.CB_CG_CD2_angle
    CA_CB_CG_CD2_diangle=geo.CA_CB_CG_CD2_diangle

    carbon_b= calculateCoordinates(N, C, CA, CA_CB_length, C_CA_CB_angle, N_C_CA_CB_diangle)
    CB= Atom("CB", carbon_b, 0.0 , 1.0, " "," CB", 0,"C")
    carbon_g1= calculateCoordinates(N, CA, CB, CB_CG_length, CA_CB_CG_angle, N_CA_CB_CG_diangle)
    CG= Atom("CG", carbon_g1, 0.0, 1.0, " ", " CG", 0, "C")
    carbon_d1= calculateCoordinates(CA, CB, CG, CG_CD1_length, CB_CG_CD1_angle, CA_CB_CG_CD1_diangle)
    CD1= Atom("CD1", carbon_d1, 0.0, 1.0, " ", " CD1", 0, "C")
    carbon_d2= calculateCoordinates(CA, CB, CG, CG_CD2_length, CB_CG_CD2_angle, CA_CB_CG_CD2_diangle)
    CD2= Atom("CD2", carbon_d2, 0.0, 1.0, " ", " CD2", 0, "C")

    ##Create Residue Data Structure
    res= Residue((' ', segID, ' '), "LEU", '    ')
    res.add(N)
    res.add(CA)
    res.add(C)
    res.add(O)
    res.add(CB)
    res.add(CG)
    res.add(CD1)
    res.add(CD2)
    return res
    
def makeThr(segID, N, CA, C, O, geo):
    '''Creates a Threonine residue'''
    ##R-Group
    CA_CB_length=geo.CA_CB_length
    C_CA_CB_angle=geo.C_CA_CB_angle
    N_C_CA_CB_diangle=geo.N_C_CA_CB_diangle
    
    CB_OG1_length=geo.CB_OG1_length
    CA_CB_OG1_angle=geo.CA_CB_OG1_angle
    N_CA_CB_OG1_diangle=geo.N_CA_CB_OG1_diangle 
        
    CB_CG2_length=geo.CB_CG2_length
    CA_CB_CG2_angle=geo.CA_CB_CG2_angle
    N_CA_CB_CG2_diangle= geo.N_CA_CB_CG2_diangle

    carbon_b= calculateCoordinates(N, C, CA, CA_CB_length, C_CA_CB_angle, N_C_CA_CB_diangle)
    CB= Atom("CB", carbon_b, 0.0 , 1.0, " "," CB", 0,"C")
    oxygen_g1= calculateCoordinates(N, CA, CB, CB_OG1_length, CA_CB_OG1_angle, N_CA_CB_OG1_diangle)
    OG1= Atom("OG1", oxygen_g1, 0.0, 1.0, " ", " OG1", 0, "O")
    carbon_g2= calculateCoordinates(N, CA, CB, CB_CG2_length, CA_CB_CG2_angle, N_CA_CB_CG2_diangle)
    CG2= Atom("CG2", carbon_g2, 0.0, 1.0, " ", " CG2", 0, "C")

    ##Create Residue Data Structure
    res= Residue((' ', segID, ' '), "THR", '    ')
    res.add(N)
    res.add(CA)
    res.add(C)
    res.add(O)
    res.add(CB)
    res.add(OG1)
    res.add(CG2)
    return res

def makeArg(segID, N, CA, C, O, geo):
    '''Creates an Arginie residue'''
    ##R-Group
    CA_CB_length=geo.CA_CB_length
    C_CA_CB_angle=geo.C_CA_CB_angle
    N_C_CA_CB_diangle=geo.N_C_CA_CB_diangle

    CB_CG_length=geo.CB_CG_length
    CA_CB_CG_angle= geo.CA_CB_CG_angle
    N_CA_CB_CG_diangle=geo.N_CA_CB_CG_diangle
    
    CG_CD_length=geo.CG_CD_length
    CB_CG_CD_angle=geo.CB_CG_CD_angle
    CA_CB_CG_CD_diangle=geo.CA_CB_CG_CD_diangle
    
    CD_NE_length=geo.CD_NE_length
    CG_CD_NE_angle=geo.CG_CD_NE_angle
    CB_CG_CD_NE_diangle=geo.CB_CG_CD_NE_diangle

    NE_CZ_length=geo.NE_CZ_length
    CD_NE_CZ_angle=geo.CD_NE_CZ_angle
    CG_CD_NE_CZ_diangle=geo.CG_CD_NE_CZ_diangle

    CZ_NH1_length=geo.CZ_NH1_length
    NE_CZ_NH1_angle=geo.NE_CZ_NH1_angle
    CD_NE_CZ_NH1_diangle=geo.CD_NE_CZ_NH1_diangle

    CZ_NH2_length=geo.CZ_NH2_length
    NE_CZ_NH2_angle=geo.NE_CZ_NH2_angle
    CD_NE_CZ_NH2_diangle=geo.CD_NE_CZ_NH2_diangle
    
    carbon_b= calculateCoordinates(N, C, CA, CA_CB_length, C_CA_CB_angle, N_C_CA_CB_diangle)
    CB= Atom("CB", carbon_b, 0.0 , 1.0, " "," CB", 0,"C")
    carbon_g= calculateCoordinates(N, CA, CB, CB_CG_length, CA_CB_CG_angle, N_CA_CB_CG_diangle)
    CG= Atom("CG", carbon_g, 0.0, 1.0, " ", " CG", 0, "C")
    carbon_d= calculateCoordinates(CA, CB, CG, CG_CD_length, CB_CG_CD_angle, CA_CB_CG_CD_diangle)
    CD= Atom("CD", carbon_d, 0.0, 1.0, " ", " CD", 0, "C")
    nitrogen_e= calculateCoordinates(CB, CG, CD, CD_NE_length, CG_CD_NE_angle, CB_CG_CD_NE_diangle)
    NE= Atom("NE", nitrogen_e, 0.0, 1.0, " ", " NE", 0, "N")
    carbon_z= calculateCoordinates(CG, CD, NE, NE_CZ_length, CD_NE_CZ_angle, CG_CD_NE_CZ_diangle)
    CZ= Atom("CZ", carbon_z, 0.0, 1.0, " ", " CZ", 0, "C")
    nitrogen_h1= calculateCoordinates(CD, NE, CZ, CZ_NH1_length, NE_CZ_NH1_angle, CD_NE_CZ_NH1_diangle)
    NH1= Atom("NH1", nitrogen_h1, 0.0, 1.0, " ", " NH1", 0, "N")
    nitrogen_h2= calculateCoordinates(CD, NE, CZ, CZ_NH2_length, NE_CZ_NH2_angle, CD_NE_CZ_NH2_diangle)
    NH2= Atom("NH2", nitrogen_h2, 0.0, 1.0, " ", " NH2", 0, "N")

    ##Create Residue Data Structure
    res= Residue((' ', segID, ' '), "ARG", '    ')
    res.add(N)
    res.add(CA)
    res.add(C)
    res.add(O)
    res.add(CB)
    res.add(CG)
    res.add(CD)
    res.add(NE)
    res.add(CZ)
    res.add(NH1)
    res.add(NH2)
    return res

def makeLys(segID, N, CA, C, O, geo):
    '''Creates a Lysine residue'''
    ##R-Group
    CA_CB_length=geo.CA_CB_length
    C_CA_CB_angle=geo.C_CA_CB_angle
    N_C_CA_CB_diangle=geo.N_C_CA_CB_diangle

    CB_CG_length=geo.CB_CG_length
    CA_CB_CG_angle=geo.CA_CB_CG_angle
    N_CA_CB_CG_diangle=geo.N_CA_CB_CG_diangle

    CG_CD_length=geo.CG_CD_length
    CB_CG_CD_angle=geo.CB_CG_CD_angle
    CA_CB_CG_CD_diangle=geo.CA_CB_CG_CD_diangle

    CD_CE_length=geo.CD_CE_length
    CG_CD_CE_angle=geo.CG_CD_CE_angle
    CB_CG_CD_CE_diangle=geo.CB_CG_CD_CE_diangle

    CE_NZ_length=geo.CE_NZ_length
    CD_CE_NZ_angle=geo.CD_CE_NZ_angle
    CG_CD_CE_NZ_diangle=geo.CG_CD_CE_NZ_diangle
    
    carbon_b= calculateCoordinates(N, C, CA, CA_CB_length, C_CA_CB_angle, N_C_CA_CB_diangle)
    CB= Atom("CB", carbon_b, 0.0 , 1.0, " "," CB", 0,"C")
    carbon_g= calculateCoordinates(N, CA, CB, CB_CG_length, CA_CB_CG_angle, N_CA_CB_CG_diangle)
    CG= Atom("CG", carbon_g, 0.0, 1.0, " ", " CG", 0, "C")
    carbon_d= calculateCoordinates(CA, CB, CG, CG_CD_length, CB_CG_CD_angle, CA_CB_CG_CD_diangle)
    CD= Atom("CD", carbon_d, 0.0, 1.0, " ", " CD", 0, "C")
    carbon_e= calculateCoordinates(CB, CG, CD, CD_CE_length, CG_CD_CE_angle, CB_CG_CD_CE_diangle)
    CE= Atom("CE", carbon_e, 0.0, 1.0, " ", " CE", 0, "C")
    nitrogen_z= calculateCoordinates(CG, CD, CE, CE_NZ_length, CD_CE_NZ_angle, CG_CD_CE_NZ_diangle)
    NZ= Atom("NZ", nitrogen_z, 0.0, 1.0, " ", " NZ", 0, "N")

    ##Create Residue Data Structure
    res= Residue((' ', segID, ' '), "LYS", '    ')
    res.add(N)
    res.add(CA)
    res.add(C)
    res.add(O)
    res.add(CB)
    res.add(CG)
    res.add(CD)
    res.add(CE)
    res.add(NZ)
    return res

def makeAsp(segID, N, CA, C, O, geo):
    '''Creates an Aspartic Acid residue'''
    ##R-Group
    CA_CB_length=geo.CA_CB_length
    C_CA_CB_angle=geo.C_CA_CB_angle
    N_C_CA_CB_diangle=geo.N_C_CA_CB_diangle

    CB_CG_length=geo.CB_CG_length
    CA_CB_CG_angle=geo.CA_CB_CG_angle
    N_CA_CB_CG_diangle=geo.N_CA_CB_CG_diangle

    CG_OD1_length=geo.CG_OD1_length
    CB_CG_OD1_angle=geo.CB_CG_OD1_angle
    CA_CB_CG_OD1_diangle=geo.CA_CB_CG_OD1_diangle

    CG_OD2_length=geo.CG_OD2_length
    CB_CG_OD2_angle=geo.CB_CG_OD2_angle
    CA_CB_CG_OD2_diangle=geo.CA_CB_CG_OD2_diangle

    carbon_b= calculateCoordinates(N, C, CA, CA_CB_length, C_CA_CB_angle, N_C_CA_CB_diangle)
    CB= Atom("CB", carbon_b, 0.0 , 1.0, " "," CB", 0,"C")
    carbon_g= calculateCoordinates(N, CA, CB, CB_CG_length, CA_CB_CG_angle, N_CA_CB_CG_diangle)
    CG= Atom("CG", carbon_g, 0.0, 1.0, " ", " CG", 0, "C")
    oxygen_d1= calculateCoordinates(CA, CB, CG, CG_OD1_length, CB_CG_OD1_angle, CA_CB_CG_OD1_diangle)
    OD1= Atom("OD1", oxygen_d1, 0.0, 1.0, " ", " OD1", 0, "O")
    oxygen_d2= calculateCoordinates(CA, CB, CG, CG_OD2_length, CB_CG_OD2_angle, CA_CB_CG_OD2_diangle)
    OD2= Atom("OD2", oxygen_d2, 0.0, 1.0, " ", " OD2", 0, "O")

    ##Create Residue Data Structure
    res= Residue((' ', segID, ' '), "ASP", '    ')
    res.add(N)
    res.add(CA)
    res.add(C)
    res.add(O)
    res.add(CB)
    res.add(CG) 
    res.add(OD1)
    res.add(OD2)
    return res

def makeAsn(segID,N, CA, C, O, geo):
    '''Creates an Asparagine residue'''
    ##R-Group
    CA_CB_length=geo.CA_CB_length
    C_CA_CB_angle=geo.C_CA_CB_angle
    N_C_CA_CB_diangle=geo.N_C_CA_CB_diangle
    
    CB_CG_length=geo.CB_CG_length
    CA_CB_CG_angle=geo.CA_CB_CG_angle
    N_CA_CB_CG_diangle=geo.N_CA_CB_CG_diangle
    
    CG_OD1_length=geo.CG_OD1_length
    CB_CG_OD1_angle=geo.CB_CG_OD1_angle
    CA_CB_CG_OD1_diangle=geo.CA_CB_CG_OD1_diangle
    
    CG_ND2_length=geo.CG_ND2_length
    CB_CG_ND2_angle=geo.CB_CG_ND2_angle
    CA_CB_CG_ND2_diangle=geo.CA_CB_CG_ND2_diangle

    carbon_b= calculateCoordinates(N, C, CA, CA_CB_length, C_CA_CB_angle, N_C_CA_CB_diangle)
    CB= Atom("CB", carbon_b, 0.0 , 1.0, " "," CB", 0,"C")
    carbon_g= calculateCoordinates(N, CA, CB, CB_CG_length, CA_CB_CG_angle, N_CA_CB_CG_diangle)
    CG= Atom("CG", carbon_g, 0.0, 1.0, " ", " CG", 0, "C")
    oxygen_d1= calculateCoordinates(CA, CB, CG, CG_OD1_length, CB_CG_OD1_angle, CA_CB_CG_OD1_diangle)
    OD1= Atom("OD1", oxygen_d1, 0.0, 1.0, " ", " OD1", 0, "O")
    nitrogen_d2= calculateCoordinates(CA, CB, CG, CG_ND2_length, CB_CG_ND2_angle, CA_CB_CG_ND2_diangle)
    ND2= Atom("ND2", nitrogen_d2, 0.0, 1.0, " ", " ND2", 0, "N")
    res= Residue((' ', segID, ' '), "ASN", '    ')

    ##Create Residue Data Structure
    res.add(N)
    res.add(CA)
    res.add(C)
    res.add(O)
    res.add(CB)
    res.add(CG) 
    res.add(OD1)
    res.add(ND2)
    return res

def makeGlu(segID, N, CA, C, O, geo):
    '''Creates a Glutamic Acid residue'''
    ##R-Group
    CA_CB_length=geo.CA_CB_length
    C_CA_CB_angle = geo.C_CA_CB_angle
    N_C_CA_CB_diangle=geo.N_C_CA_CB_diangle
    
    CB_CG_length=geo.CB_CG_length
    CA_CB_CG_angle=geo.CA_CB_CG_angle
    N_CA_CB_CG_diangle=geo.N_CA_CB_CG_diangle

    CG_CD_length=geo.CG_CD_length
    CB_CG_CD_angle=geo.CB_CG_CD_angle
    CA_CB_CG_CD_diangle=geo.CA_CB_CG_CD_diangle

    CD_OE1_length=geo.CD_OE1_length
    CG_CD_OE1_angle=geo.CG_CD_OE1_angle
    CB_CG_CD_OE1_diangle=geo.CB_CG_CD_OE1_diangle

    CD_OE2_length=geo.CD_OE2_length
    CG_CD_OE2_angle=geo.CG_CD_OE2_angle
    CB_CG_CD_OE2_diangle=geo.CB_CG_CD_OE2_diangle

    carbon_b= calculateCoordinates(N, C, CA, CA_CB_length, C_CA_CB_angle, N_C_CA_CB_diangle)
    CB= Atom("CB", carbon_b, 0.0 , 1.0, " "," CB", 0,"C")
    carbon_g= calculateCoordinates(N, CA, CB, CB_CG_length, CA_CB_CG_angle, N_CA_CB_CG_diangle)
    CG= Atom("CG", carbon_g, 0.0, 1.0, " ", " CG", 0, "C")
    carbon_d= calculateCoordinates(CA, CB, CG, CG_CD_length, CB_CG_CD_angle, CA_CB_CG_CD_diangle)
    CD= Atom("CD", carbon_d, 0.0, 1.0, " ", " CD", 0, "C")
    oxygen_e1= calculateCoordinates(CB, CG, CD, CD_OE1_length, CG_CD_OE1_angle, CB_CG_CD_OE1_diangle)
    OE1= Atom("OE1", oxygen_e1, 0.0, 1.0, " ", " OE1", 0, "O")
    oxygen_e2= calculateCoordinates(CB, CG, CD, CD_OE2_length, CG_CD_OE2_angle, CB_CG_CD_OE2_diangle)
    OE2= Atom("OE2", oxygen_e2, 0.0, 1.0, " ", " OE2", 0, "O")

    ##Create Residue Data Structure
    res= Residue((' ', segID, ' '), "GLU", '    ')
    
    res.add(N)
    res.add(CA)
    res.add(C)
    res.add(O)
    res.add(CB)
    res.add(CG)
    res.add(CD)
    res.add(OE1)
    res.add(OE2)
    return res

def makeGln(segID, N, CA, C, O, geo):
    '''Creates a Glutamine residue'''
    ##R-Group
    CA_CB_length=geo.CA_CB_length
    C_CA_CB_angle=geo.C_CA_CB_angle
    N_C_CA_CB_diangle=geo.N_C_CA_CB_diangle
    
    CB_CG_length=geo.CB_CG_length
    CA_CB_CG_angle=geo.CA_CB_CG_angle
    N_CA_CB_CG_diangle=geo.N_CA_CB_CG_diangle

    CG_CD_length=geo.CG_CD_length
    CB_CG_CD_angle=geo.CB_CG_CD_angle
    CA_CB_CG_CD_diangle=geo.CA_CB_CG_CD_diangle
    
    CD_OE1_length=geo.CD_OE1_length
    CG_CD_OE1_angle=geo.CG_CD_OE1_angle
    CB_CG_CD_OE1_diangle=geo.CB_CG_CD_OE1_diangle
    
    CD_NE2_length=geo.CD_NE2_length
    CG_CD_NE2_angle=geo.CG_CD_NE2_angle
    CB_CG_CD_NE2_diangle=geo.CB_CG_CD_NE2_diangle

    carbon_b= calculateCoordinates(N, C, CA, CA_CB_length, C_CA_CB_angle, N_C_CA_CB_diangle)
    CB= Atom("CB", carbon_b, 0.0 , 1.0, " "," CB", 0,"C")
    carbon_g= calculateCoordinates(N, CA, CB, CB_CG_length, CA_CB_CG_angle, N_CA_CB_CG_diangle)
    CG= Atom("CG", carbon_g, 0.0, 1.0, " ", " CG", 0, "C")
    carbon_d= calculateCoordinates(CA, CB, CG, CG_CD_length, CB_CG_CD_angle, CA_CB_CG_CD_diangle)
    CD= Atom("CD", carbon_d, 0.0, 1.0, " ", " CD", 0, "C")
    oxygen_e1= calculateCoordinates(CB, CG, CD, CD_OE1_length, CG_CD_OE1_angle, CB_CG_CD_OE1_diangle)
    OE1= Atom("OE1", oxygen_e1, 0.0, 1.0, " ", " OE1", 0, "O")
    nitrogen_e2= calculateCoordinates(CB, CG, CD, CD_NE2_length, CG_CD_NE2_angle, CB_CG_CD_NE2_diangle)
    NE2= Atom("NE2", nitrogen_e2, 0.0, 1.0, " ", " NE2", 0, "N")


    ##Create Residue DS
    res= Residue((' ', segID, ' '), "GLN", '    ')
    
    res.add(N)
    res.add(CA)
    res.add(C)
    res.add(O)
    res.add(CB)
    res.add(CG)
    res.add(CD)
    res.add(OE1)
    res.add(NE2)
    return res

def makeMet(segID, N, CA, C, O, geo):
    '''Creates a Methionine residue'''
    ##R-Group
    CA_CB_length=geo.CA_CB_length
    C_CA_CB_angle=geo.C_CA_CB_angle
    N_C_CA_CB_diangle=geo.N_C_CA_CB_diangle

    CB_CG_length=geo.CB_CG_length
    CA_CB_CG_angle=geo.CA_CB_CG_angle
    N_CA_CB_CG_diangle=geo.N_CA_CB_CG_diangle
    
    CG_SD_length=geo.CG_SD_length
    CB_CG_SD_angle=geo.CB_CG_SD_angle
    CA_CB_CG_SD_diangle=geo.CA_CB_CG_SD_diangle
    
    SD_CE_length=geo.SD_CE_length
    CG_SD_CE_angle=geo.CG_SD_CE_angle
    CB_CG_SD_CE_diangle=geo.CB_CG_SD_CE_diangle
    
    carbon_b= calculateCoordinates(N, C, CA, CA_CB_length, C_CA_CB_angle, N_C_CA_CB_diangle)
    CB= Atom("CB", carbon_b, 0.0 , 1.0, " "," CB", 0,"C")
    carbon_g= calculateCoordinates(N, CA, CB, CB_CG_length, CA_CB_CG_angle, N_CA_CB_CG_diangle)
    CG= Atom("CG", carbon_g, 0.0, 1.0, " ", " CG", 0, "C")
    sulfur_d= calculateCoordinates(CA, CB, CG, CG_SD_length, CB_CG_SD_angle, CA_CB_CG_SD_diangle)
    SD= Atom("SD", sulfur_d, 0.0, 1.0, " ", " SD", 0, "S")
    carbon_e= calculateCoordinates(CB, CG, SD, SD_CE_length, CG_SD_CE_angle, CB_CG_SD_CE_diangle)
    CE= Atom("CE", carbon_e, 0.0, 1.0, " ", " CE", 0, "C")

    ##Create Residue Data Structure
    res= Residue((' ', segID, ' '), "MET", '    ')
    res.add(N)
    res.add(CA)
    res.add(C)
    res.add(O)
    res.add(CB)
    res.add(CG)
    res.add(SD)
    res.add(CE)
    return res

def makeHis(segID, N, CA, C, O, geo):
    '''Creates a Histidine residue'''
    ##R-Group
    CA_CB_length=geo.CA_CB_length
    C_CA_CB_angle=geo.C_CA_CB_angle
    N_C_CA_CB_diangle=geo.N_C_CA_CB_diangle

    CB_CG_length=geo.CB_CG_length
    CA_CB_CG_angle=geo.CA_CB_CG_angle
    N_CA_CB_CG_diangle=geo.N_CA_CB_CG_diangle
    
    CG_ND1_length=geo.CG_ND1_length
    CB_CG_ND1_angle=geo.CB_CG_ND1_angle
    CA_CB_CG_ND1_diangle=geo.CA_CB_CG_ND1_diangle
    
    CG_CD2_length=geo.CG_CD2_length
    CB_CG_CD2_angle=geo.CB_CG_CD2_angle
    CA_CB_CG_CD2_diangle=geo.CA_CB_CG_CD2_diangle
    
    ND1_CE1_length=geo.ND1_CE1_length
    CG_ND1_CE1_angle=geo.CG_ND1_CE1_angle
    CB_CG_ND1_CE1_diangle=geo.CB_CG_ND1_CE1_diangle
    
    CD2_NE2_length=geo.CD2_NE2_length
    CG_CD2_NE2_angle=geo.CG_CD2_NE2_angle
    CB_CG_CD2_NE2_diangle=geo.CB_CG_CD2_NE2_diangle
    
    carbon_b= calculateCoordinates(N, C, CA, CA_CB_length, C_CA_CB_angle, N_C_CA_CB_diangle)
    CB= Atom("CB", carbon_b, 0.0 , 1.0, " "," CB", 0,"C")
    carbon_g= calculateCoordinates(N, CA, CB, CB_CG_length, CA_CB_CG_angle, N_CA_CB_CG_diangle)
    CG= Atom("CG", carbon_g, 0.0, 1.0, " ", " CG", 0, "C")
    nitrogen_d1= calculateCoordinates(CA, CB, CG, CG_ND1_length, CB_CG_ND1_angle, CA_CB_CG_ND1_diangle)
    ND1= Atom("ND1", nitrogen_d1, 0.0, 1.0, " ", " ND1", 0, "N")
    carbon_d2= calculateCoordinates(CA, CB, CG, CG_CD2_length, CB_CG_CD2_angle, CA_CB_CG_CD2_diangle)
    CD2= Atom("CD2", carbon_d2, 0.0, 1.0, " ", " CD2", 0, "C")
    carbon_e1= calculateCoordinates(CB, CG, ND1, ND1_CE1_length, CG_ND1_CE1_angle, CB_CG_ND1_CE1_diangle)
    CE1= Atom("CE1", carbon_e1, 0.0, 1.0, " ", " CE1", 0, "C")
    nitrogen_e2= calculateCoordinates(CB, CG, CD2, CD2_NE2_length, CG_CD2_NE2_angle, CB_CG_CD2_NE2_diangle)
    NE2= Atom("NE2", nitrogen_e2, 0.0, 1.0, " ", " NE2", 0, "N")

    ##Create Residue Data Structure
    res= Residue((' ', segID, ' '), "HIS", '    ')
    res.add(N)
    res.add(CA)
    res.add(C)
    res.add(O)
    res.add(CB)
    res.add(CG)
    res.add(ND1)
    res.add(CD2)
    res.add(CE1)
    res.add(NE2)
    return res

def makePro(segID, N, CA, C, O, geo):
    '''Creates a Proline residue'''
    ##R-Group
    CA_CB_length=geo.CA_CB_length
    C_CA_CB_angle=geo.C_CA_CB_angle
    N_C_CA_CB_diangle=geo.N_C_CA_CB_diangle
    
    CB_CG_length=geo.CB_CG_length
    CA_CB_CG_angle=geo.CA_CB_CG_angle
    N_CA_CB_CG_diangle=geo.N_CA_CB_CG_diangle
    
    CG_CD_length=geo.CG_CD_length
    CB_CG_CD_angle=geo.CB_CG_CD_angle
    CA_CB_CG_CD_diangle=geo.CA_CB_CG_CD_diangle
    
    carbon_b= calculateCoordinates(N, C, CA, CA_CB_length, C_CA_CB_angle, N_C_CA_CB_diangle)
    CB= Atom("CB", carbon_b, 0.0 , 1.0, " "," CB", 0,"C")
    carbon_g= calculateCoordinates(N, CA, CB, CB_CG_length, CA_CB_CG_angle, N_CA_CB_CG_diangle)
    CG= Atom("CG", carbon_g, 0.0, 1.0, " ", " CG", 0, "C")
    carbon_d= calculateCoordinates(CA, CB, CG, CG_CD_length, CB_CG_CD_angle, CA_CB_CG_CD_diangle)
    CD= Atom("CD", carbon_d, 0.0, 1.0, " ", " CD", 0, "C")

    ##Create Residue Data Structure
    res= Residue((' ', segID, ' '), "PRO", '    ')
    
    res.add(N)
    res.add(CA)
    res.add(C)
    res.add(O)
    res.add(CB)
    res.add(CG)
    res.add(CD)

    return res

def makePhe(segID, N, CA, C, O, geo):
    '''Creates a Phenylalanine residue'''
    ##R-Group
    CA_CB_length=geo.CA_CB_length
    C_CA_CB_angle=geo.C_CA_CB_angle
    N_C_CA_CB_diangle=geo.N_C_CA_CB_diangle
    
    CB_CG_length=geo.CB_CG_length
    CA_CB_CG_angle=geo.CA_CB_CG_angle
    N_CA_CB_CG_diangle=geo.N_CA_CB_CG_diangle

    CG_CD1_length=geo.CG_CD1_length
    CB_CG_CD1_angle=geo.CB_CG_CD1_angle
    CA_CB_CG_CD1_diangle=geo.CA_CB_CG_CD1_diangle

    CG_CD2_length=geo.CG_CD2_length
    CB_CG_CD2_angle=geo.CB_CG_CD2_angle
    CA_CB_CG_CD2_diangle= geo.CA_CB_CG_CD2_diangle
    
    CD1_CE1_length=geo.CD1_CE1_length
    CG_CD1_CE1_angle=geo.CG_CD1_CE1_angle
    CB_CG_CD1_CE1_diangle=geo.CB_CG_CD1_CE1_diangle

    CD2_CE2_length=geo.CD2_CE2_length
    CG_CD2_CE2_angle=geo.CG_CD2_CE2_angle
    CB_CG_CD2_CE2_diangle=geo.CB_CG_CD2_CE2_diangle

    CE1_CZ_length=geo.CE1_CZ_length
    CD1_CE1_CZ_angle=geo.CD1_CE1_CZ_angle
    CG_CD1_CE1_CZ_diangle=geo.CG_CD1_CE1_CZ_diangle

    carbon_b= calculateCoordinates(N, C, CA, CA_CB_length, C_CA_CB_angle, N_C_CA_CB_diangle)
    CB= Atom("CB", carbon_b, 0.0 , 1.0, " "," CB", 0,"C")
    carbon_g= calculateCoordinates(N, CA, CB, CB_CG_length, CA_CB_CG_angle, N_CA_CB_CG_diangle)
    CG= Atom("CG", carbon_g, 0.0, 1.0, " ", " CG", 0, "C")
    carbon_d1= calculateCoordinates(CA, CB, CG, CG_CD1_length, CB_CG_CD1_angle, CA_CB_CG_CD1_diangle)
    CD1= Atom("CD1", carbon_d1, 0.0, 1.0, " ", " CD1", 0, "C")
    carbon_d2= calculateCoordinates(CA, CB, CG, CG_CD2_length, CB_CG_CD2_angle, CA_CB_CG_CD2_diangle)
    CD2= Atom("CD2", carbon_d2, 0.0, 1.0, " ", " CD2", 0, "C")
    carbon_e1= calculateCoordinates(CB, CG, CD1, CD1_CE1_length, CG_CD1_CE1_angle, CB_CG_CD1_CE1_diangle)
    CE1= Atom("CE1", carbon_e1, 0.0, 1.0, " ", " CE1", 0, "C")
    carbon_e2= calculateCoordinates(CB, CG, CD2, CD2_CE2_length, CG_CD2_CE2_angle, CB_CG_CD2_CE2_diangle)
    CE2= Atom("CE2", carbon_e2, 0.0, 1.0, " ", " CE2", 0, "C")
    carbon_z= calculateCoordinates(CG, CD1, CE1, CE1_CZ_length, CD1_CE1_CZ_angle, CG_CD1_CE1_CZ_diangle)
    CZ= Atom("CZ", carbon_z, 0.0, 1.0, " ", " CZ", 0, "C")

    ##Create Residue Data Structures
    res= Residue((' ', segID, ' '), "PHE", '    ')
    res.add(N)
    res.add(CA)
    res.add(C)
    res.add(O)
    res.add(CB)
    res.add(CG)
    res.add(CD1)
    res.add(CE1)
    res.add(CD2)
    res.add(CE2)
    res.add(CZ)
    return res

def makeTyr(segID, N, CA, C, O, geo):
    '''Creates a Tyrosine residue'''
    ##R-Group
    CA_CB_length=geo.CA_CB_length
    C_CA_CB_angle=geo.C_CA_CB_angle
    N_C_CA_CB_diangle=geo.N_C_CA_CB_diangle
    
    CB_CG_length=geo.CB_CG_length
    CA_CB_CG_angle=geo.CA_CB_CG_angle
    N_CA_CB_CG_diangle=geo.N_CA_CB_CG_diangle

    CG_CD1_length=geo.CG_CD1_length
    CB_CG_CD1_angle=geo.CB_CG_CD1_angle
    CA_CB_CG_CD1_diangle=geo.CA_CB_CG_CD1_diangle
    
    CG_CD2_length=geo.CG_CD2_length
    CB_CG_CD2_angle=geo.CB_CG_CD2_angle
    CA_CB_CG_CD2_diangle=geo.CA_CB_CG_CD2_diangle
    
    CD1_CE1_length=geo.CD1_CE1_length
    CG_CD1_CE1_angle=geo.CG_CD1_CE1_angle
    CB_CG_CD1_CE1_diangle=geo.CB_CG_CD1_CE1_diangle

    CD2_CE2_length=geo.CD2_CE2_length
    CG_CD2_CE2_angle=geo.CG_CD2_CE2_angle
    CB_CG_CD2_CE2_diangle=geo.CB_CG_CD2_CE2_diangle

    CE1_CZ_length=geo.CE1_CZ_length
    CD1_CE1_CZ_angle=geo.CD1_CE1_CZ_angle
    CG_CD1_CE1_CZ_diangle=geo.CG_CD1_CE1_CZ_diangle

    CZ_OH_length=geo.CZ_OH_length
    CE1_CZ_OH_angle=geo.CE1_CZ_OH_angle
    CD1_CE1_CZ_OH_diangle=geo.CD1_CE1_CZ_OH_diangle

    carbon_b= calculateCoordinates(N, C, CA, CA_CB_length, C_CA_CB_angle, N_C_CA_CB_diangle)
    CB= Atom("CB", carbon_b, 0.0 , 1.0, " "," CB", 0,"C")
    carbon_g= calculateCoordinates(N, CA, CB, CB_CG_length, CA_CB_CG_angle, N_CA_CB_CG_diangle)
    CG= Atom("CG", carbon_g, 0.0, 1.0, " ", " CG", 0, "C")
    carbon_d1= calculateCoordinates(CA, CB, CG, CG_CD1_length, CB_CG_CD1_angle, CA_CB_CG_CD1_diangle)
    CD1= Atom("CD1", carbon_d1, 0.0, 1.0, " ", " CD1", 0, "C")
    carbon_d2= calculateCoordinates(CA, CB, CG, CG_CD2_length, CB_CG_CD2_angle, CA_CB_CG_CD2_diangle)
    CD2= Atom("CD2", carbon_d2, 0.0, 1.0, " ", " CD2", 0, "C")
    carbon_e1= calculateCoordinates(CB, CG, CD1, CD1_CE1_length, CG_CD1_CE1_angle, CB_CG_CD1_CE1_diangle)
    CE1= Atom("CE1", carbon_e1, 0.0, 1.0, " ", " CE1", 0, "C")
    carbon_e2= calculateCoordinates(CB, CG, CD2, CD2_CE2_length, CG_CD2_CE2_angle, CB_CG_CD2_CE2_diangle)
    CE2= Atom("CE2", carbon_e2, 0.0, 1.0, " ", " CE2", 0, "C")
    carbon_z= calculateCoordinates(CG, CD1, CE1, CE1_CZ_length, CD1_CE1_CZ_angle, CG_CD1_CE1_CZ_diangle)
    CZ= Atom("CZ", carbon_z, 0.0, 1.0, " ", " CZ", 0, "C")
    oxygen_h= calculateCoordinates(CD1, CE1, CZ, CZ_OH_length, CE1_CZ_OH_angle, CD1_CE1_CZ_OH_diangle)
    OH= Atom("OH", oxygen_h, 0.0, 1.0, " ", " OH", 0, "O")

    ##Create Residue Data S
    res= Residue((' ', segID, ' '), "TYR", '    ')
    res.add(N)
    res.add(CA)
    res.add(C)
    res.add(O)
    res.add(CB)
    res.add(CG)
    res.add(CD1)
    res.add(CE1)
    res.add(CD2)
    res.add(CE2)
    res.add(CZ)
    res.add(OH)
    return res

def makeTrp(segID, N, CA, C, O, geo):
    '''Creates a Tryptophan residue'''
    ##R-Group
    CA_CB_length=geo.CA_CB_length
    C_CA_CB_angle=geo.C_CA_CB_angle
    N_C_CA_CB_diangle=geo.N_C_CA_CB_diangle

    CB_CG_length=geo.CB_CG_length
    CA_CB_CG_angle=geo.CA_CB_CG_angle
    N_CA_CB_CG_diangle=geo.N_CA_CB_CG_diangle

    CG_CD1_length=geo.CG_CD1_length
    CB_CG_CD1_angle=geo.CB_CG_CD1_angle
    CA_CB_CG_CD1_diangle=geo.CA_CB_CG_CD1_diangle

    CG_CD2_length=geo.CG_CD2_length
    CB_CG_CD2_angle=geo.CB_CG_CD2_angle
    CA_CB_CG_CD2_diangle=geo.CA_CB_CG_CD2_diangle
    
    CD1_NE1_length=geo.CD1_NE1_length
    CG_CD1_NE1_angle=geo.CG_CD1_NE1_angle
    CB_CG_CD1_NE1_diangle=geo.CB_CG_CD1_NE1_diangle

    CD2_CE2_length=geo.CD2_CE2_length
    CG_CD2_CE2_angle=geo.CG_CD2_CE2_angle
    CB_CG_CD2_CE2_diangle=geo.CB_CG_CD2_CE2_diangle

    CD2_CE3_length=geo.CD2_CE3_length
    CG_CD2_CE3_angle=geo.CG_CD2_CE3_angle
    CB_CG_CD2_CE3_diangle=geo.CB_CG_CD2_CE3_diangle

    CE2_CZ2_length=geo.CE2_CZ2_length
    CD2_CE2_CZ2_angle=geo.CD2_CE2_CZ2_angle
    CG_CD2_CE2_CZ2_diangle=geo.CG_CD2_CE2_CZ2_diangle

    CE3_CZ3_length=geo.CE3_CZ3_length
    CD2_CE3_CZ3_angle=geo.CD2_CE3_CZ3_angle
    CG_CD2_CE3_CZ3_diangle=geo.CG_CD2_CE3_CZ3_diangle

    CZ2_CH2_length=geo.CZ2_CH2_length
    CE2_CZ2_CH2_angle=geo.CE2_CZ2_CH2_angle
    CD2_CE2_CZ2_CH2_diangle=geo.CD2_CE2_CZ2_CH2_diangle

    carbon_b= calculateCoordinates(N, C, CA, CA_CB_length, C_CA_CB_angle, N_C_CA_CB_diangle)
    CB= Atom("CB", carbon_b, 0.0 , 1.0, " "," CB", 0,"C")
    carbon_g= calculateCoordinates(N, CA, CB, CB_CG_length, CA_CB_CG_angle, N_CA_CB_CG_diangle)
    CG= Atom("CG", carbon_g, 0.0, 1.0, " ", " CG", 0, "C")
    carbon_d1= calculateCoordinates(CA, CB, CG, CG_CD1_length, CB_CG_CD1_angle, CA_CB_CG_CD1_diangle)
    CD1= Atom("CD1", carbon_d1, 0.0, 1.0, " ", " CD1", 0, "C")
    carbon_d2= calculateCoordinates(CA, CB, CG, CG_CD2_length, CB_CG_CD2_angle, CA_CB_CG_CD2_diangle)
    CD2= Atom("CD2", carbon_d2, 0.0, 1.0, " ", " CD2", 0, "C")
    nitrogen_e1= calculateCoordinates(CB, CG, CD1, CD1_NE1_length, CG_CD1_NE1_angle, CB_CG_CD1_NE1_diangle)
    NE1= Atom("NE1", nitrogen_e1, 0.0, 1.0, " ", " NE1", 0, "N")
    carbon_e2= calculateCoordinates(CB, CG, CD2, CD2_CE2_length, CG_CD2_CE2_angle, CB_CG_CD2_CE2_diangle)
    CE2= Atom("CE2", carbon_e2, 0.0, 1.0, " ", " CE2", 0, "C")
    carbon_e3= calculateCoordinates(CB, CG, CD2, CD2_CE3_length, CG_CD2_CE3_angle, CB_CG_CD2_CE3_diangle)
    CE3= Atom("CE3", carbon_e3, 0.0, 1.0, " ", " CE3", 0, "C")

    carbon_z2= calculateCoordinates(CG, CD2, CE2, CE2_CZ2_length, CD2_CE2_CZ2_angle, CG_CD2_CE2_CZ2_diangle)
    CZ2= Atom("CZ2", carbon_z2, 0.0, 1.0, " ", " CZ2", 0, "C")

    carbon_z3= calculateCoordinates(CG, CD2, CE3, CE3_CZ3_length, CD2_CE3_CZ3_angle, CG_CD2_CE3_CZ3_diangle)
    CZ3= Atom("CZ3", carbon_z3, 0.0, 1.0, " ", " CZ3", 0, "C")

    carbon_h2= calculateCoordinates(CD2, CE2, CZ2, CZ2_CH2_length, CE2_CZ2_CH2_angle, CD2_CE2_CZ2_CH2_diangle)
    CH2= Atom("CH2", carbon_h2, 0.0, 1.0, " ", " CH2", 0, "C")
    
    ##Create Residue DS
    res= Residue((' ', segID, ' '), "TRP", '    ')
    res.add(N)
    res.add(CA)
    res.add(C)
    res.add(O)
    res.add(CB)
    res.add(CG)
    res.add(CD1)
    res.add(CD2)

    res.add(NE1)
    res.add(CE2)
    res.add(CE3)

    res.add(CZ2)
    res.add(CZ3)

    res.add(CH2)
    return res

def initialize_res_coord(residue, coord):
    '''Creates a new structure containing a single amino acid. The type and
    geometry of the amino acid are determined by the argument, which has to be
    either a geometry object or a single-letter amino acid code.
    The amino acid will be placed into chain A of model 0.'''
    
    if isinstance( residue, Geo ):
        geo = residue
    else:
        geo=geometry(residue) 
    
    segID=1
    AA= geo.residue_name
    CA_N_length=geo.CA_N_length
    CA_C_length=geo.CA_C_length
    N_CA_C_angle=geo.N_CA_C_angle
    
    CA_coord= coord
    C_coord= numpy.array([CA_C_length,0,0])
    N_coord = numpy.array([CA_N_length*math.cos(N_CA_C_angle*(math.pi/180.0)),CA_N_length*math.sin(N_CA_C_angle*(math.pi/180.0)),0])

    N= Atom("N", N_coord, 0.0 , 1.0, " "," N", 0, "N")
    CA=Atom("CA", CA_coord, 0.0 , 1.0, " "," CA", 0,"C")
    C= Atom("C", C_coord, 0.0, 1.0, " ", " C",0,"C")

    ##Create Carbonyl atom (to be moved later)
    C_O_length=geo.C_O_length
    CA_C_O_angle=geo.CA_C_O_angle
    N_CA_C_O_diangle=geo.N_CA_C_O_diangle
    
    carbonyl=calculateCoordinates(N, CA, C, C_O_length, CA_C_O_angle, N_CA_C_O_diangle)
    O= Atom("O",carbonyl , 0.0 , 1.0, " "," O", 0, "O")

    if(AA=='G'):
        res=makeGly(segID, N, CA, C, O, geo)
    elif(AA=='A'):
        res=makeAla(segID, N, CA, C, O, geo)
    elif(AA=='S'):
        res=makeSer(segID, N, CA, C, O, geo)
    elif(AA=='C'):
        res=makeCys(segID, N, CA, C, O, geo)
    elif(AA=='V'):
        res=makeVal(segID, N, CA, C, O, geo)
    elif(AA=='I'):
        res=makeIle(segID, N, CA, C, O, geo)
    elif(AA=='L'):
        res=makeLeu(segID, N, CA, C, O, geo)
    elif(AA=='T'):
        res=makeThr(segID, N, CA, C, O, geo)
    elif(AA=='R'):
        res=makeArg(segID, N, CA, C, O, geo)
    elif(AA=='K'):
        res=makeLys(segID, N, CA, C, O, geo)
    elif(AA=='D'):
        res=makeAsp(segID, N, CA, C, O, geo)
    elif(AA=='E'):
        res=makeGlu(segID, N, CA, C, O, geo)
    elif(AA=='N'):
        res=makeAsn(segID, N, CA, C, O, geo)
    elif(AA=='Q'):
        res=makeGln(segID, N, CA, C, O, geo)
    elif(AA=='M'):
        res=makeMet(segID, N, CA, C, O, geo)
    elif(AA=='H'):
        res=makeHis(segID, N, CA, C, O, geo)
    elif(AA=='P'):
        res=makePro(segID, N, CA, C, O, geo)
    elif(AA=='F'):
        res=makePhe(segID, N, CA, C, O, geo)
    elif(AA=='Y'):
        res=makeTyr(segID, N, CA, C, O, geo)
    elif(AA=='W'):
        res=makeTrp(segID, N, CA, C, O, geo)
    else:
        res=makeGly(segID, N, CA, C, O, geo)

    cha= Chain('A')
    cha.add(res)
    
    mod= Model(0)
    mod.add(cha)

    struc= Structure('X')
    struc.add(mod)
    return struc
    
def initialize_res(residue):
    '''Creates a new structure containing a single amino acid. The type and
    geometry of the amino acid are determined by the argument, which has to be
    either a geometry object or a single-letter amino acid code.
    The amino acid will be placed into chain A of model 0.'''
    
    if isinstance( residue, Geo ):
        geo = residue
    else:
        geo=geometry(residue) 
    
    segID=1
    AA= geo.residue_name
    CA_N_length=geo.CA_N_length
    CA_C_length=geo.CA_C_length
    N_CA_C_angle=geo.N_CA_C_angle
    
    CA_coord= numpy.array([0.,0.,0.])
    C_coord= numpy.array([CA_C_length,0,0])
    N_coord = numpy.array([CA_N_length*math.cos(N_CA_C_angle*(math.pi/180.0)),CA_N_length*math.sin(N_CA_C_angle*(math.pi/180.0)),0])

    N= Atom("N", N_coord, 0.0 , 1.0, " "," N", 0, "N")
    CA=Atom("CA", CA_coord, 0.0 , 1.0, " "," CA", 0,"C")
    C= Atom("C", C_coord, 0.0, 1.0, " ", " C",0,"C")

    ##Create Carbonyl atom (to be moved later)
    C_O_length=geo.C_O_length
    CA_C_O_angle=geo.CA_C_O_angle
    N_CA_C_O_diangle=geo.N_CA_C_O_diangle
    
    carbonyl=calculateCoordinates(N, CA, C, C_O_length, CA_C_O_angle, N_CA_C_O_diangle)
    O= Atom("O",carbonyl , 0.0 , 1.0, " "," O", 0, "O")

    if(AA=='G'):
        res=makeGly(segID, N, CA, C, O, geo)
    elif(AA=='A'):
        res=makeAla(segID, N, CA, C, O, geo)
    elif(AA=='S'):
        res=makeSer(segID, N, CA, C, O, geo)
    elif(AA=='C'):
        res=makeCys(segID, N, CA, C, O, geo)
    elif(AA=='V'):
        res=makeVal(segID, N, CA, C, O, geo)
    elif(AA=='I'):
        res=makeIle(segID, N, CA, C, O, geo)
    elif(AA=='L'):
        res=makeLeu(segID, N, CA, C, O, geo)
    elif(AA=='T'):
        res=makeThr(segID, N, CA, C, O, geo)
    elif(AA=='R'):
        res=makeArg(segID, N, CA, C, O, geo)
    elif(AA=='K'):
        res=makeLys(segID, N, CA, C, O, geo)
    elif(AA=='D'):
        res=makeAsp(segID, N, CA, C, O, geo)
    elif(AA=='E'):
        res=makeGlu(segID, N, CA, C, O, geo)
    elif(AA=='N'):
        res=makeAsn(segID, N, CA, C, O, geo)
    elif(AA=='Q'):
        res=makeGln(segID, N, CA, C, O, geo)
    elif(AA=='M'):
        res=makeMet(segID, N, CA, C, O, geo)
    elif(AA=='H'):
        res=makeHis(segID, N, CA, C, O, geo)
    elif(AA=='P'):
        res=makePro(segID, N, CA, C, O, geo)
    elif(AA=='F'):
        res=makePhe(segID, N, CA, C, O, geo)
    elif(AA=='Y'):
        res=makeTyr(segID, N, CA, C, O, geo)
    elif(AA=='W'):
        res=makeTrp(segID, N, CA, C, O, geo)
    else:
        res=makeGly(segID, N, CA, C, O, geo)

    cha= Chain('A')
    cha.add(res)
    
    mod= Model(0)
    mod.add(cha)

    struc= Structure('X')
    struc.add(mod)
    return struc


def getReferenceResidue(structure):
    '''Returns the last residue of chain A model 0 of the given structure.
    
    This function is a helper function that should not normally be called
    directly.'''

    # If the following line doesn't work we're in trouble.
    # Likely initialize_res() wasn't called.
    resRef = structure[0]['A'].child_list[-1]
    
    # If the residue is not an amino acid we're in trouble.
    # Likely somebody is trying to append residues to an existing
    # structure that has non-amino-acid molecules in the chain.
    assert is_aa(resRef)
        
    return resRef

def add_residue_from_geo_coord(structure, geo, coord):
    '''Adds a residue to chain A model 0 of the given structure, and
    returns the new structure. The residue to be added is determined by
    the geometry object given as second argument.
    
    This function is a helper function and should not normally be called
    directly. Call add_residue() instead.'''
    resRef= getReferenceResidue(structure)
    AA=geo.residue_name
    segID= resRef.get_id()[1]
    segID+=1

    ##geometry to bring together residue
    peptide_bond=geo.peptide_bond
    CA_C_N_angle=geo.CA_C_N_angle
    C_N_CA_angle=geo.C_N_CA_angle

    ##Backbone Coordinages
    N_CA_C_angle=geo.N_CA_C_angle
    CA_N_length=geo.CA_N_length
    CA_C_length=geo.CA_C_length
    phi= geo.phi
    psi_im1=geo.psi_im1
    omega=geo.omega

    N_coord=calculateCoordinates(resRef['N'], resRef['CA'], resRef['C'], peptide_bond, CA_C_N_angle, psi_im1)
    N= Atom("N", N_coord, 0.0 , 1.0, " "," N", 0, "N")
    
    CA_coord=Vector(coord[0], coord[1], coord[2]).get_array()#calculateCoordinates(resRef['CA'], resRef['C'], N, CA_N_length, C_N_CA_angle, omega)
    CA=Atom("CA", CA_coord, 0.0 , 1.0, " "," CA", 0,"C")

    C_coord=calculateCoordinates(resRef['C'], N, CA, CA_C_length, N_CA_C_angle, phi)
    C= Atom("C", C_coord, 0.0, 1.0, " ", " C",0,"C")

    ##Create Carbonyl atom (to be moved later)
    C_O_length=geo.C_O_length
    CA_C_O_angle=geo.CA_C_O_angle
    N_CA_C_O_diangle=geo.N_CA_C_O_diangle

    carbonyl=calculateCoordinates(N, CA, C, C_O_length, CA_C_O_angle, N_CA_C_O_diangle)
    O= Atom("O",carbonyl , 0.0 , 1.0, " "," O", 0, "O")
    
    if(AA=='G'):
        res=makeGly(segID, N, CA, C, O, geo)
    elif(AA=='A'):
        res=makeAla(segID, N, CA, C, O, geo)
    elif(AA=='S'):
        res=makeSer(segID, N, CA, C, O, geo)
    elif(AA=='C'):
        res=makeCys(segID, N, CA, C, O, geo)
    elif(AA=='V'):
        res=makeVal(segID, N, CA, C, O, geo)
    elif(AA=='I'):
        res=makeIle(segID, N, CA, C, O, geo)
    elif(AA=='L'):
        res=makeLeu(segID, N, CA, C, O, geo)
    elif(AA=='T'):
        res=makeThr(segID, N, CA, C, O, geo)
    elif(AA=='R'):
        res=makeArg(segID, N, CA, C, O, geo)
    elif(AA=='K'):
        res=makeLys(segID, N, CA, C, O, geo)
    elif(AA=='D'):
        res=makeAsp(segID, N, CA, C, O, geo)
    elif(AA=='E'):
        res=makeGlu(segID, N, CA, C, O, geo)
    elif(AA=='N'):
        res=makeAsn(segID, N, CA, C, O, geo)
    elif(AA=='Q'):
        res=makeGln(segID, N, CA, C, O, geo)
    elif(AA=='M'):
        res=makeMet(segID, N, CA, C, O, geo)
    elif(AA=='H'):
        res=makeHis(segID, N, CA, C, O, geo)
    elif(AA=='P'):
        res=makePro(segID, N, CA, C, O, geo)
    elif(AA=='F'):
        res=makePhe(segID, N, CA, C, O, geo)
    elif(AA=='Y'):
        res=makeTyr(segID, N, CA, C, O, geo)
    elif(AA=='W'):
        res=makeTrp(segID, N, CA, C, O, geo)
    else:
        res=makeGly(segID, N, CA, C, O, geo)
        
    resRef['O'].set_coord(calculateCoordinates(res['N'], resRef['CA'], resRef['C'], C_O_length, CA_C_O_angle, 180.0))

    ghost= Atom("N", calculateCoordinates(res['N'], res['CA'], res['C'], peptide_bond, CA_C_N_angle, psi_im1), 0.0 , 0.0, " ","N", 0, "N")
    res['O'].set_coord(calculateCoordinates( res['N'], res['CA'], res['C'], C_O_length, CA_C_O_angle, 180.0))

    structure[0]['A'].add(res)
    return structure
    
def add_residue_from_geo(structure, geo):
    '''Adds a residue to chain A model 0 of the given structure, and
    returns the new structure. The residue to be added is determined by
    the geometry object given as second argument.
    
    This function is a helper function and should not normally be called
    directly. Call add_residue() instead.'''
    resRef= getReferenceResidue(structure)
    AA=geo.residue_name
    segID= resRef.get_id()[1]
    segID+=1

    ##geometry to bring together residue
    peptide_bond=geo.peptide_bond
    CA_C_N_angle=geo.CA_C_N_angle
    C_N_CA_angle=geo.C_N_CA_angle

    ##Backbone Coordinages
    N_CA_C_angle=geo.N_CA_C_angle
    CA_N_length=geo.CA_N_length
    CA_C_length=geo.CA_C_length
    phi= geo.phi
    psi_im1=geo.psi_im1
    omega=geo.omega

    N_coord=calculateCoordinates(resRef['N'], resRef['CA'], resRef['C'], peptide_bond, CA_C_N_angle, psi_im1)
    N= Atom("N", N_coord, 0.0 , 1.0, " "," N", 0, "N")

    CA_coord=calculateCoordinates(resRef['CA'], resRef['C'], N, CA_N_length, C_N_CA_angle, omega)
    CA=Atom("CA", CA_coord, 0.0 , 1.0, " "," CA", 0,"C")

    C_coord=calculateCoordinates(resRef['C'], N, CA, CA_C_length, N_CA_C_angle, phi)
    C= Atom("C", C_coord, 0.0, 1.0, " ", " C",0,"C")

    ##Create Carbonyl atom (to be moved later)
    C_O_length=geo.C_O_length
    CA_C_O_angle=geo.CA_C_O_angle
    N_CA_C_O_diangle=geo.N_CA_C_O_diangle

    carbonyl=calculateCoordinates(N, CA, C, C_O_length, CA_C_O_angle, N_CA_C_O_diangle)
    O= Atom("O",carbonyl , 0.0 , 1.0, " "," O", 0, "O")
    
    if(AA=='G'):
        res=makeGly(segID, N, CA, C, O, geo)
    elif(AA=='A'):
        res=makeAla(segID, N, CA, C, O, geo)
    elif(AA=='S'):
        res=makeSer(segID, N, CA, C, O, geo)
    elif(AA=='C'):
        res=makeCys(segID, N, CA, C, O, geo)
    elif(AA=='V'):
        res=makeVal(segID, N, CA, C, O, geo)
    elif(AA=='I'):
        res=makeIle(segID, N, CA, C, O, geo)
    elif(AA=='L'):
        res=makeLeu(segID, N, CA, C, O, geo)
    elif(AA=='T'):
        res=makeThr(segID, N, CA, C, O, geo)
    elif(AA=='R'):
        res=makeArg(segID, N, CA, C, O, geo)
    elif(AA=='K'):
        res=makeLys(segID, N, CA, C, O, geo)
    elif(AA=='D'):
        res=makeAsp(segID, N, CA, C, O, geo)
    elif(AA=='E'):
        res=makeGlu(segID, N, CA, C, O, geo)
    elif(AA=='N'):
        res=makeAsn(segID, N, CA, C, O, geo)
    elif(AA=='Q'):
        res=makeGln(segID, N, CA, C, O, geo)
    elif(AA=='M'):
        res=makeMet(segID, N, CA, C, O, geo)
    elif(AA=='H'):
        res=makeHis(segID, N, CA, C, O, geo)
    elif(AA=='P'):
        res=makePro(segID, N, CA, C, O, geo)
    elif(AA=='F'):
        res=makePhe(segID, N, CA, C, O, geo)
    elif(AA=='Y'):
        res=makeTyr(segID, N, CA, C, O, geo)
    elif(AA=='W'):
        res=makeTrp(segID, N, CA, C, O, geo)
    else:
        res=makeGly(segID, N, CA, C, O, geo)
        
    resRef['O'].set_coord(calculateCoordinates(res['N'], resRef['CA'], resRef['C'], C_O_length, CA_C_O_angle, 180.0))

    ghost= Atom("N", calculateCoordinates(res['N'], res['CA'], res['C'], peptide_bond, CA_C_N_angle, psi_im1), 0.0 , 0.0, " ","N", 0, "N")
    res['O'].set_coord(calculateCoordinates( res['N'], res['CA'], res['C'], C_O_length, CA_C_O_angle, 180.0))

    structure[0]['A'].add(res)
    return structure


def make_extended_structure(AA_chain):
    '''Place a sequence of amino acids into a peptide in the extended
    conformation. The argument AA_chain holds the sequence of amino
    acids to be used.'''
    geo = geometry(AA_chain[0])
    struc=initialize_res(geo)
    
    for i in range(1,len(AA_chain)): 
        AA = AA_chain[i]
        geo = geometry(AA)
        add_residue(struc, geo)

    return struc

    
def add_residue(structure, residue, phi=-120, psi_im1=140, omega=-370):
    '''Adds a residue to chain A model 0 of the given structure, and
    returns the new structure. The residue to be added can be specified
    in two ways: either as a geometry object (in which case
    the remaining arguments phi, psi_im1, and omega are ignored) or as a
    single-letter amino-acid code. In the latter case, the optional
    arguments phi, psi_im1, and omega specify the corresponding backbone
    angles.
    
    When omega is specified, it needs to be a value greater than or equal
    to -360. Values below -360 are ignored.''' 
    
    if isinstance( residue, Geo ):
        geo = residue
    else:
        geo=geometry(residue) 
        geo.phi=phi
        geo.psi_im1=psi_im1
        if omega>-361:
            geo.omega=omega
    
    add_residue_from_geo(structure, geo)
    return structure
    
def add_residue_coord(structure, residue, coord, phi=-120, psi_im1=140, omega=-370):
    '''Adds a residue to chain A model 0 of the given structure, and
    returns the new structure. The residue to be added can be specified
    in two ways: either as a geometry object (in which case
    the remaining arguments phi, psi_im1, and omega are ignored) or as a
    single-letter amino-acid code. In the latter case, the optional
    arguments phi, psi_im1, and omega specify the corresponding backbone
    angles.
    
    When omega is specified, it needs to be a value greater than or equal
    to -360. Values below -360 are ignored.''' 
    
    if True:
        geo = residue
    else:
        geo=geometry(residue) 
        geo.phi=phi
        geo.psi_im1=psi_im1
        if omega>-361:
            geo.omega=omega
    
    add_residue_from_geo_coord(structure, geo, coord)
    return structure
    
def make_structure(AA_chain,phi,psi_im1,omega=[]):
    '''Place a sequence of amino acids into a peptide with specified
    backbone dihedral angles. The argument AA_chain holds the
    sequence of amino acids to be used. The arguments phi and psi_im1 hold
    lists of backbone angles, one for each amino acid, *starting from
    the second amino acid in the chain*. The argument 
    omega (optional) holds a list of omega angles, also starting from
    the second amino acid in the chain.'''
    geo = geometry(AA_chain[0])
    struc=initialize_res(geo)

    if len(omega)==0:
        for i in range(1,len(AA_chain)): 
            AA = AA_chain[i]
            add_residue(struc, AA, phi[i-1], psi_im1[i-1])
    else:
        for i in range(1,len(AA_chain)): 
            AA = AA_chain[i]
            add_residue(struc, AA, phi[i-1], psi_im1[i-1], omega[i-1])

    return struc

def make_structure_from_geos(geos):
    '''Creates a structure out of a list of geometry objects.'''
    model_structure=initialize_res(geos[0])
    for i in range(1,len(geos)):
        model_structure=add_residue(model_structure, geos[i])

    return model_structure    
    
def make_structure_from_geos_coords(geos, X):
    '''Creates a structure out of a list of geometry objects and coordinates.'''
    model_structure=initialize_res_coord(geos[0], X[0])
    for i in range(1,len(geos)):
        model_structure=add_residue_coord(model_structure, geos[i], X[i])

    return model_structure
