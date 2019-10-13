'''
    fullatom.py
	
    Iddo Drori
    Columbia University
	
    Build full atom protein from Calpha's backbone
	
'''

import PeptideBuilderCoords

def fullatom(pdb, aa, X):
    geos = []
    for a in aa:
        geos.append(Geometry.geometry(a))
    structure = PeptideBuilderCoords.make_structure_from_geos_coords(geos, X) 
    out = Bio.PDB.PDBIO()
    out.set_structure(structure)
    out.save(pdb+'.pdb')
