'''
Code for recovering 3D points from distance matrices
Uses Data/train_output.npz (4554 matrices from competition data)
Log of RMSE of each matrix is in file '3D.txt'
Most of the log(RMSE) is smaller than -5

Exceptions:
ID      log(RMSE)   RMSE
623     -4.85959    0.00775
1079     1.26613    3.54710
1523    -4.82079    0.00806


Only uses 4 x n values from the n x n distance matrix.
'''
import numpy as np

eps = 1e-8

def readData():
    Y = []
    file = np.load('Data/train_output.npz')
    for key in file:
        Y.append(file[key])
    n = len(Y)
    return n, Y

def sign(x:float):
    if x > eps:
        return 1
    if x < -eps:
        return -1
    return 0

class Point:
    def __init__(self,x:float,y:float,z:float):
        self.x = x
        self.y = y
        self.z = z
        
    def __str__(self):
        return '('+str(self.x)+','+str(self.y)+','+str(self.z)+')'
        
    def __add__(self,other):
        return Point(self.x+other.x,self.y+other.y,self.z+other.z)
        
    def __sub__(self,other):
        return Point(self.x-other.x,self.y-other.y,self.z-other.z)
        
    def __mul__(self,other):
        if type(other) == Point:
            # dot product
            return self.x*other.x + self.y*other.y + self.z*other.z
        
        return Point(self.x*other,self.y*other,self.z*other)
        
    def __matmul__(self,other):
        '''cross product @'''
        return Point(self.y*other.z-self.z*other.y,self.z*other.x-self.x*other.z,self.x*other.y-self.y*other.x)
        
    def length(self):
        return np.sqrt(self*self)
    
    def unit(self):
        l = self.length()
        return Point(self.x/l,self.y/l,self.z/l)

def to_numpy(x):
    out = np.zeros((len(x), 3)) 
    for (i, p) in enumerate(x):
        out[i] = np.array([p.x, p.y, p.z])
    return out

def Find4thPoint(OA:Point,OB:Point,DisCO:float,DisCA:float,DisCB:float):
    # Let OC = k1*OA + k2*OB + k3 * OA@OB      (@: cross product)
    # solves k1, k2 and k3
    
    # s1 * k1 + t1 * k2 = c1
    # s2 * k2 + t2 * k2 = c2
    s1 = 2 * OA.length()**2
    t1 = 2 * (OA*OB)
    c1 = DisCO**2 - DisCA**2 + OA.length()**2
    s2 = 2 * (OA*OB)
    t2 = 2 * OB.length()**2
    c2 = DisCO**2 - DisCB**2 + OB.length()**2
    
    if sign(s1*t2-s2*t1) != 0:
        k1 = (c1*t2-c2*t1)/(s1*t2-s2*t1)
        k2 = (s1*c2-s2*c1)/(s1*t2-s2*t1)
    else:
        k1 = 0
        k2 = 0
    
    _k3_temp = DisCO**2 - 2*k1*k2*(OA*OB)-k1**2 * OA.length()**2 - k2**2 * OB.length()**2
    if sign(_k3_temp) > 0:
        # k3 has two solutions, positive and negative
        k3 = np.sqrt(_k3_temp) / (OA@OB).length()
    else:
        k3 = 0
    
    # two possible solutions
    OC1 = OA * k1 + OB * k2 + (OA@OB) * k3
    OC2 = OA * k1 + OB * k2 - (OA@OB) * k3
    
    Error = np.sqrt((OC1.length() - DisCO)**2) + \
            np.sqrt(((OC1-OA).length() - DisCA)**2) + \
            np.sqrt(((OC1-OB).length() - DisCB)**2)
    return OC1, OC2, Error

def MatrixTo3D(y):
    '''
    Given distance matrix y
    Return Points, distance matrix from Points, and log of RMSE for two distance matrices
    
    Put the first 3 points on XY plane.
    For any other point, there are only two solutions(only the sign of the z coordinate is different) 
    given the distance to the first three points.
    So let the 4th point have a positive z axis.
    For the 5th point and later, use the distance to the first 3 points to find the two solution.
    And use the distance to the 4th point to determine the sign of the z coordinate.
    '''
    length = y.shape[0]
    O = Point(0,0,0)
    A = Point(y[0,1],0,0)
    _x = 0.5*(y[0,1]+(y[0,2]**2-y[1,2]**2)/y[0,1])
    _y = np.sqrt(y[0,2]**2-_x**2)
    B = Point(_x,_y,0)
    
    if sign(_y) == 0:
        print('OAB on same line')
        return None
    Points = [O,A,B]
    for k in range(3,length):
        C, RevC, Error = Find4thPoint(A,B,y[0,k],y[1,k],y[2,k])
        if k >= 4:
            Dis_C = (C-Points[3]).length()
            Dis_RevC = (RevC - Points[3]).length()
            if abs(Dis_C-y[k,3]) > abs(Dis_RevC-y[k,3]):
                C = RevC
        Points.append(C)
    
    M = np.zeros((length,length))
    for i in range(length):
        for j in range(i+1,length):
            M[i,j] = M[j,i] = (Points[i] - Points[j]).length()
    return to_numpy(Points), M, np.log(np.sqrt(np.mean((M-y)**2)))

if __name__ == '__main__':
    n, Y = readData()
    f = open('3D.txt','w')
    f.close()
    
    for k in range(n):
        Ps, M, Error = MatrixTo3D(Y[k])
        
        f = open('3D.txt','a')
        f.write('{0},{1}\r\n'.format(k,Error))
        f.close()
        
        if Error > -5:
            print(k,Error,np.exp(Error))







