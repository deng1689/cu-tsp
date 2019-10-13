import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import Bio.PDB

from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.DSSP import DSSP
from Bio.PDB.Polypeptide import PPBuilder
from Bio.SVDSuperimposer import SVDSuperimposer
from scipy.spatial import distance_matrix
from pdb import set_trace

dims = [75, 116, 136, 300, 229, 239]
proteins = ['2n64', '5j4a', '5fjl', '3jb5', '5fhy', '5jo9']
LEARNING_RATE = 1e-4
PRECISION = 1e-1
MAX_ITER = 200

LAMBDA = 1
RHO = 100

def load_data(path):
    global dims, proteins

    D = []
    for dim in dims:
        d = np.zeros((dim, dim))
        D.append(d)

    testfile = open(path, 'r')
    next(testfile)
    for line in testfile:
        line = line.split(',')
        key = line[0].split('_')
        protein = key[0]
        type = key[1]
        if type == 'd':
            i = int(key[2]) - 1
            j = int(key[3]) - 1
            value = float(line[1])
            p = proteins.index(protein)
            D[p][i, j] = value
    return D

def recover_coords_sdp(D):
    n = D.shape[0]
    G = cp.Variable((n, n), symmetric=True)
    eta = cp.Variable((n, n))
    lam = 1

    constraints = [G >> 0] 
    constraints += [eta[i, j] >= 0 for i in range(n) for j in range(n)]
    reg = lam * cp.norm(eta, p=1)
    Gii = cp.diag(G)
    hstacked = cp.hstack([cp.reshape(Gii, (n, 1)) for i in range(n)])
    vstacked = cp.vstack([cp.reshape(Gii, (1, n)) for i in range(n)])
    obj = reg + 0.5 * cp.sum((hstacked + vstacked - 2 * G + eta - D**2)**2)
    #obj = 0.5 * cp.sum((hstacked + vstacked - 2 * G - D**2))
    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve(verbose=True)
    #print("The optimal value is", prob.value)
    #print("A solution G is: {}".format(G.value))
    u, s, v = np.linalg.svd(G.value)
    u, s = np.real(u), np.real(s)
    X = np.dot(u, np.diag(np.sqrt(s)))[:, :3]
    return X

'''
    The objective we want to minimize
    input:
        x   : ndarray with shape (2*m*m, )
            1st m*m is flattened G (Gram matrix)
            2nd m*m is flattened s (slack variable)
        args : tuple
            args[0] is m (number of residues)
            args[1] is D (distance matrix)
            args[2] is Z (same shape)
            args[3] is U (same shape)
            args[4] is lam (coefficient for slack)
            args[5] is rho (coefficient for augmented lagrangian)
    output:
        scalar
'''
def obj(x, args):
    m, D, Z, U, lam, rho = args
    G = x[:m*m].reshape(m, m)
    S = x[m*m:].reshape(m, m)

    F1 = lam * np.linalg.norm(S, 1)

    Gii = np.diag(G)
    F2 = Gii.reshape(m, 1) + Gii.reshape(1, m) - 2 * G + S - np.square(D)
    F2 = 0.5 * np.square(F2).sum()
    F3 = 0.5 * rho * (np.linalg.norm(G-Z+U, 2) ** 2)

    return F1+F2+F3

'''
    Compute gradient of objective w.r.t. G, S
    input:
        same as obj
    output:
        grads : ndarray with shape (2*m*m, ))
            derivative w.r.t. G, S
'''
def gradient(x, args):
    m, D, Z, U, lam, rho = args
    G = x[:m*m].reshape(m, m)
    S = x[m*m:].reshape(m, m)

    Gii = np.diag(G)
    dLOSS = Gii.reshape(m,1) + Gii.reshape(1,m) - 2*G + S - D**2

    # compute derivatives w.r.t G
    dG = np.copy(dLOSS)
    np.fill_diagonal(dG, 0) # need to zero diagonal elements diagonal elements
    dGii = dG.sum(0) + dG.sum(1)
    #dG = -2*dG + np.diag(dGii) + rho*(G-Z+U) # incorrect gradient
    
    dG = -2*dG + np.diag(dGii) # corrected gradient
    np.fill_diagonal(dG, 0)
    dG += rho*(G-Z+U)
    
    # compute derivatives w.r.t S
    dS = lam * np.sign(S) + dLOSS
    
    return np.r_[dG, dS].ravel()

# gradient descent minimizing objective
def vanilla_descent(x, args, beta):
    it, step = 0, 10
    alpha = 0.0008 # set small learning rate with many iterations
    while step > PRECISION and it < MAX_ITER:
        it +=1
        x_ = np.copy(x)
        g = gradient(x, args)
        x = x - alpha * g
        step = np.abs(obj(x, args) - obj(x_, args))
    return x

'''
    gradient descent using momentum
    minimizing objective
'''
def momentum_descent(x, args, beta):
    it, step = 0, 10
    alpha = LEARNING_RATE
    beta = 0.9
    m = np.zeros_like(x)
    while step > PRECISION and it < MAX_ITER:
        it +=1
        x_ = np.copy(x)
        g = gradient(x, args)
        m = beta * m + alpha * g
        x = x - m
        step = np.abs(obj(x, args) - obj(x_, args))
    return x
 

'''
    find gram matrix by admm
'''
def get_gram(x, descent_type, args):
    it, step = 0, 100
    while step > .1:
        it += 1
        x_ = np.copy(x)

        # minimize loss w.r.t. G, S for current Z, U
        m, D, Z, U, lam, rho = args
        if descent_type == 'vanilla':
            x = vanilla_descent(x, args, 0.9)
        elif descent_type == 'momentum':
            x = momentum_descent(x, args, 0.9)
        G = x[:m*m].reshape(m, m)
        S = x[m*m:].reshape(m, m)

        # project G+U onto p.s.d. cone
        l, v = np.linalg.eig(G+U)
        idx = l.argsort()[::-1] # sort eigenvalues
        l = l[idx]
        l = (l > 0) * l # projecting
        v = v[:, idx]

        # update U, V
        Z = np.dot(np.dot(v, np.diag(l)), v.T) # update Z
        U = U+G-Z # update U

        # set new args
        F_ = obj(x_, args)
        args = (m, D, Z, U, LAMBDA, RHO)
        F = obj(x, args)
        step = np.abs(F-F_)

        #print('it #' + str(it) + ' step: ' + str(step))
    return G

def recover_coords_admm(D, descent_type):
    # initializations
    m = D.shape[0]

    Z = np.ones_like(D)
    U = np.ones_like(D)
    G = np.ones_like(D)
    S = np.ones_like(D)

    # initial settings
    x = np.r_[G, S].ravel()
    args = (m, D, Z, U, LAMBDA, RHO)

    # get gram
    G = get_gram(x, descent_type, args)

    u, s, v = np.linalg.svd(G)
    u, s = np.real(u), np.real(s)
    X = np.dot(u, np.diag(np.sqrt(s)))[:, :3]

    return X

def recover_coords(D, method='SDP'):
    if method == 'SDP':
        return recover_coords_sdp(D)
    elif 'ADMM' in method:
        descent_type = method[5:]
        return recover_coords_admm(D, descent_type)

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

def visualize(X, A, method, protein):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    gt0 = np.array(A)
    pred0 = np.array(X)
    plt.cla()
    ax.scatter(gt0[:,0],  gt0[:,1], gt0[:,2], color='red', label='GT')
    ax.scatter(pred0[:,0],  pred0[:,1], pred0[:,2], color='blue', label='Predicted')
    ax.set_title(protein)
    ax.legend(loc='upper left', fontsize='x-large')
    plt.savefig('{}_scatter_{}.png'.format(protein, method))   

def compute_gdt_ts(a, b):
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
    
def main():
    global dims, proteins
    method = 'SDP'
    #method = 'ADMM-vanilla'
    #method = 'ADMM-momentum'

    for (i, protein) in enumerate(proteins):
        pdb_path = protein + '.pdb'
        Din = get_calpha_distance_matrix(pdb_path)
        X = recover_coords(Din, method=method)
        
        # Also load corresponding ground truth protein
        A = get_calpha_positions(pdb_path)

        aligned_X = align(X, A)
        print("[{}] {} - DRMSD: {}. GDT-TS: {}".format(protein, method,
                compute_drmsd(aligned_X, np.array(A)), 
                compute_gdt_ts(aligned_X, np.array(A))))
        visualize(aligned_X, A, method, protein)

main()
