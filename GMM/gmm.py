



import numpy as np
import pandas as pd
import math


K = 3
error = 0.1

X = np.loadtxt('clusters.txt', delimiter = ',')


# create random weights for each data point for each cluster
def random_ric(k):
    r = np.random.choice([1,2,3], size = 3)
    return [r[0]/r.sum(), r[1]/r.sum(), r[2]/r.sum()]

def show_result(para,result):
    print(para + ': ')
    for i in result:
        print(i)


class gaussian:
    def __init__(self,ric,X, K, threshold):
        self.X = X # N*d matrix
        self.d = X.shape[1] #d
        self.N = X.shape[0] #X
        self.C = K
        
        self.ric = ric # 150 * 3
        
        self.threshold = threshold
        
        self.mu = [] # a list of dx1 vector u
        self.pi = [] # a list of number pi
        self.sigma = [] # a list of dxd matrix
        
    def get_params(self):
        # find out pi_c, mu_c, sigma_c
        return self.mu, self.pi, self.sigma
    
    def clean_param(self):
        self.mu.clear()
        self.pi.clear()
        self.sigma.clear()
    
    def calculate_param(self):
        for c in range(self.C):
            mu_c = np.zeros(self.d)        
            sigma_c = np.zeros((self.d, self.d))
            
            Nc = self.ric[:,c].sum() # a number            
            pi_c = Nc/self.N # a number
            
            for i in range(self.N):
                mu_c += self.ric[i][c] * self.X[i]
            mu_c = mu_c/Nc # d
            
            for i in range(self.N):
                sigma_c += self.ric[i][c]* np.dot((self.X[i]- mu_c).reshape(2,1),
                                                  (self.X[i]-mu_c).reshape(2,1).T)
            sigma_c = sigma_c/Nc
            
            self.mu.append(mu_c.reshape(2,1)) # a list of d*1 array
            self.pi.append(pi_c) # a list of number
            self.sigma.append(sigma_c) # a list of d*d matrix
            
    def gaussian_function(self,i,c): #gaussian distribution of data point i in a cluster c
        mu = self.mu[c] # d * 1
        d = self.X[i].reshape(2,1) - mu
        cov = self.sigma[c]
        cov_inv = np.linalg.inv(cov) # d*d
        const = 1 / np.sqrt( (np.pi*2)**self.d * np.linalg.det(cov) )
        power =np.dot(d.T, cov_inv)
        power = np.dot(power, d)
        G = const * np.exp( -0.5 * power )
        return G[0][0]
    
    def calculate_ric(self):
        new_ric = np.zeros((self.N, self.C))
        for i in range(self.N):
            denominator = 0
            for c in range(self.C):
                denominator+= self.pi[c] * self.gaussian_function(i, c)
            for c in range(self.C):
                numerator = self.pi[c] * self.gaussian_function(i,c)
                new_ric[i][c]= numerator / denominator
        return new_ric
    
    def run(self):
        convergent = False
        count = 1000
        while not convergent and count > 0:
#            count -= 1
            self.calculate_param()
            new_ric = self.calculate_ric()
            mean_ric = np.abs(self.ric-new_ric)
            
            if mean_ric.all() <= self.threshold:
                convergent = True
#                print(count)
                return self.get_params()
            else:
                self.clean_param()
                self.ric = new_ric



ric = np.zeros((X.shape[0],K))
for i in range(len(ric)):
    ric[i] = np.array(random_ric(3))

g = gaussian(ric, X, K, 0.000001)
MU, PI, SIGMA = g.run()

show_result('MEAN',MU)
show_result('AMPLITUDE',PI)
show_result('SIGMA',SIGMA)









