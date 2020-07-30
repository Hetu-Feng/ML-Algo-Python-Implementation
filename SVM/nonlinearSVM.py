

import numpy as np
import cvxopt

def polynomial_kernal(x1,x2): #for dimension 2
    return np.array([1, x1**2, x2**2, np.sqrt(2)*x1, np.sqrt(2)*x2, np.sqrt(2)*x1*x2])

def nolinsep(x,y):

    sample_num, feature_num = x.shape
    k = np.zeros((sample_num, 6))

    #apply kernal function to data point using polynomial kernal function
    for i in range(0, sample_num):
        k[i] = polynomial_kernal(x[i][0], x[i][1])
    zTz = np.zeros((sample_num, sample_num))
    for i in range(sample_num):
        for j in range(sample_num):
            zTz[i][j] = np.dot(k[i], k[j])
    P = cvxopt.matrix(np.outer(y,y)*zTz)
    q = cvxopt.matrix(np.ones(sample_num)*-1)
    G = cvxopt.matrix(np.diag(np.ones(sample_num) * -1))
    h = cvxopt.matrix(np.zeros(sample_num))
    A = cvxopt.matrix(y, (1,(sample_num)))
    b = cvxopt.matrix(0.0)
    a = np.ravel(cvxopt.solvers.qp(P, q, G, h, A, b)['x'])
    sv_index = np.where(a>0.00001)[0]  # non-sero support vectors index [ 5, 36, 51, 55, 94, 95]

    #weights
    weights = np.zeros(6)
    for i in sv_index:
        weights += np.array(a[i] * y[i] * k[i])

    # b
    index = sv_index[0]
    b = y[index] - np.dot(weights, k[index])


    print('support vectors are:\n', a[sv_index])
    print('intercepts are :', b )
    print('weights are :\n', weights)


if __name__ =='__main__':
    data = np.loadtxt('nonlinsep.txt', dtype = 'float', delimiter =",")
    x=data[:,0:2]
    y=data[:,2]
    nolinsep(x,y)




