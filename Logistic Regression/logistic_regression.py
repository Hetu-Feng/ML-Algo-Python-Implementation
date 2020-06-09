





import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('classification.txt', dtype = 'float',delimiter=',')
X = data[:,:3]
Y = data[:,4].reshape(-1,1)
Y[Y<0] = 0

samples_num, features_num = X.shape

w = np.random.random(size=features_num+1) #1x4 array
x_0 = np.ones(samples_num).reshape(samples_num,1)
X =np.concatenate((x_0,X), axis =1)

learning_rate = 0.01

def hypothesis(w, X):
    h = 1/(1+np.exp(-np.dot(w,X.T)))
    return h

def gradient(h, Y, samples_num):
    h = h.reshape(-1,1)
    diff = h-Y
    gradient = np.dot(X.T,diff) /samples_num
    return gradient

iteration = 0
while iteration <7000:
    h = hypothesis(w, X)
    g = gradient(h, Y, samples_num)
    
    w = w - learning_rate * g.T
    
    iteration+=1

prediction = h.copy()
prediction[prediction<0.5] = 0
prediction[prediction>=0.5] =1

accurate_prediction = len(prediction.T == Y)
accuracy = accurate_prediction/samples_num

print('Logistic Regression\n', 'coefficients:', w, '\nAccuracy:', accuracy)