import numpy as np
import scipy as sp
import pandas as pd
from scipy.spatial.distance import pdist, squareform
import time
import pickle
import matplotlib.pyplot as plt

LEARNING_RATE = 1e-4
PRECISION = 1e-1
MAX_ITER = 200

LAMBDA = 1
RHO = 100

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
def descent(x, args, beta):
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
def descent__(x, args, beta):
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
    hypergradient descent
    minimizing objective
'''
def descent2(x, args, beta):
    it, step = 0, 10
    alpha = 0.01
    g_prev = np.zeros(x.shape)
    while step > PRECISION and it < MAX_ITER:
        it +=1
        x_ = np.copy(x)
        g = gradient(x, args)
        alpha = alpha - beta * g * g_prev
        x = x - alpha * g
        step = np.abs(obj(x, args) - obj(x_, args))
        g_prev = g
    return x
    



'''
    gradient descent using adam
'''
def descent_(x, args, beta):
    it, step = 0, 10
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    beta1 = 0.9
    beta2 = 0.9
    alpha = 0.03
    eps = 0.00001
    while step > PRECISION and it < MAX_ITER:
        it +=1
        x_ = np.copy(x)
        g = gradient(x, args)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * (g * g)
        #v = v - (1 - beta2) * np.sign(v - (g * g)) * (g * g)
        x = x - alpha * m / (np.sqrt(v) + eps)
        step = np.abs(obj(x, args) - obj(x_, args))
    return x


'''
    find gram matrix by admm
'''
def get_gram(x, args):
    it, step = 0, 100
    while step > .1:
        it += 1
        x_ = np.copy(x)

        # minimize loss w.r.t. G, S for current Z, U
        m, D, Z, U, lam, rho = args
        x = descent(x, args, 0.9)
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

def rmsd(X, X_):
    return np.sqrt(np.sum(np.square(X-X_)) / len(X))

''' recover 3D coordinates from Gram matrix '''
def recover_coords(D):
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
    G = get_gram(x, args)

    u, s, v = np.linalg.svd(G)
    u, s = np.real(u), np.real(s)
    X = np.dot(u, np.diag(np.sqrt(s)))[:, :3]

    return G, X

#if __name__ == '__main__':
#    ''' Loading distance matrices '''
    
dims = [75, 116, 136, 300, 229, 239]
proteins = ['2n64','5j4a','5fjl','3jb5','5fhy','5jo9']
D = []
for dim in dims:
	d = np.zeros((dim,dim))
	D.append(d)
	
path = 'ground-truth.csv'
print(path)
testfile = open(path,'r')
next(testfile)
for line in testfile:
	line = line.split(',')
	key = line[0].split('_')
	protein = key[0]
	type = key[1]
	if type =='d':
		i = int(key[2])-1
		j = int(key[3])-1
		value = float(line[1])
		p = proteins.index(protein)
		D[p][i,j] = value
    # matrices = np.load('../../tsp-data/train_output.npz')
    
    #lens = np.zeros(10, np.int32)
    #errs = np.zeros(10, np.float32)
    #times = np.zeros(10, np.float32)
    #i, count = 0, 0
    #while count < 6:
    #    D = D[count]#matrices['arr_'+str(i)]
    #    i+=1
    #    if len(D)>300:
    #        continue
    #    start = time.time()
    #    G, X = recover_coords(D)
    #    D_ = squareform(pdist(X))
    #    rmse = np.sqrt(np.mean(np.square(D-D_)))
    #    end = time.time()

    #    lens[count] = len(D)
    #    errs[count] = rmse
    #    times[count] = end - start

    #    print count, len(D), rmse, end-start

    #    count+=1

    # np.savetxt('lens.csv', lens, delimiter=',')
    # np.savetxt('errors.csv', errs, delimiter=',')
    # np.savetxt('times.csv', times, delimiter=',')

    #x = np.loadtxt('lens.csv')
    #y = np.loadtxt('times.csv')
    #plt.scatter(x,y)
    #plt.xlabel('sequence length')
    #plt.ylabel('time (s)')
    #plt.title('ADMM runtime')
    #plt.show()

Xall = []
for i in range(6):
	print(i+1)
	Din = D[i]#matrices['arr_15']
	G, X = recover_coords(Din)
	D_ = squareform(pdist(X))
	#print np.sqrt(np.mean(np.square(Din-D_)))
	print(X)
	Xall.append(X)
coordsfile = '3dcoords-ground-truth.pkl'
with open(coordsfile, 'wb') as coordsoutfile:
	pickle.dump(Xall, coordsoutfile, pickle.HIGHEST_PROTOCOL)

