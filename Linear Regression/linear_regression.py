


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data = np.loadtxt('linear-regression.txt',delimiter=',')
X = data[:,0:2]
Y = data[:,2].reshape(-1,1)

samples_num, features_num = X.shape

x_0 = np.ones(samples_num).reshape(samples_num,1)
X = np.concatenate((x_0,X), axis =1).T  # 3x2000

X = np.matrix(X)
Y = np.matrix(Y)

w = np.linalg.inv(X * X.T) * X * Y

print('Linear Regression coefiicients:\n', w)