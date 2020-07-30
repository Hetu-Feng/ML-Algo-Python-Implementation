


import numpy as np
import cvxopt

data = np.loadtxt('linsep.txt', dtype = 'float', delimiter =",")
x=data[:,0:2]
y=data[:,2]

def linearSVM(x,y):
    
    sample_num, feature_num = x.shape

    # min (1/2)x^T.Px + q^T.x  /  Gx <= h & Ax = b
    # max 1^T.a - (1/2).a^T.x^T.X.a  / a>=0 & y^T.a = 0
    # project my SVM variable to cvxopt's quadratic programming solver
    # p = x transpose dot product x
    xTx = np.zeros((sample_num, sample_num))
    for i in range(sample_num):
        for j in range(sample_num):
            xTx[i,j] = np.dot(x[i],x[j])
    P = cvxopt.matrix(np.outer(y,y) * xTx)
    # -1(1,N)
    q = cvxopt.matrix(np.ones(sample_num) * -1)
    # G = -1N.x.N -1(N,N)
    G = cvxopt.matrix(np.diag(np.ones(sample_num) * -1))
    # h=0(1,N)
    h = cvxopt.matrix(np.zeros(sample_num))
    # y^T
    A = cvxopt.matrix(y, (1,sample_num))
    #b=0
    b = cvxopt.matrix(0.0)


    #alpha
    a = np.ravel(cvxopt.solvers.qp(P, q, G, h, A, b)['x'])
    sv_index = np.where(a>0.00001)[0]  # non-sero support vectors index [27,83,87]

    #support vectors
    SV = x[sv_index] 

    #weights
    weights = np.zeros((1,feature_num))[0]
    for i in range(len(a)):
        weights += np.array(a[i] * y[i] * x[i])

    # b
    index = sv_index[0]
    b = y[index] - np.dot(weights, x[index])

    # slope
    slope = -weights[0]/weights[1]
    intercept = -b/weights[1]
    
    return SV, slope, intercept

sv, sl, inte = linearSVM(x,y)
print('support vectors are:\n', sv)
print('equation of the line is: {}x + {}'.format(sl,inte))
