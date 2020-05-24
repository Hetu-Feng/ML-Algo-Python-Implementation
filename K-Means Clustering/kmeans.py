



import numpy as np
import pandas as pd
import math



K = 3
error = 0.1

X = np.loadtxt('clusters.txt', delimiter = ',')


# define a class k-means
class Kmean():
    
    def __init__(self, X, K, error):
        self.X = X
        self.K = K
        self.error = error
        self.clusters = {}
        
    def initial_centroid(self):
        idx = np.random.randint(self.X.shape[0], size =3)
        return X[idx, :] 
    
    def distance(self, p1, p2):
        return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
    
    def calculate_centroid(self, data):
        return [data.transpose()[0].mean(), data.transpose()[1].mean()]
    
    def get_clusters(self):
        return clusters
    

    # to run, iteratively calculate distance and centroids til convergence happens
    def run(self):
        
        convergent = False
        centroids = self.initial_centroid()
        dist = {}
        clusters = {'c1':np.array([centroids[0]]), 'c2':np.array([centroids[1]]), 'c3':np.array([centroids[2]])}
        
        while not convergent:
            for p in self.X:
                dist['c1'] = self.distance(p, centroids[0])
                dist['c2'] = self.distance(p, centroids[1])
                dist['c3'] = self.distance(p, centroids[2])
                min_dist = min(dist.values())
                min_c = [key for key, value in dist.items() if value == min_dist][0]
                clusters[min_c] =  np.append(clusters[min_c], [p], axis = 0)

            new_centroids = np.array([self.calculate_centroid(clusters['c1']),
                                      self.calculate_centroid(clusters['c2']),
                                      self.calculate_centroid(clusters['c3'])])
            
            if np.abs(new_centroids-centroids).mean() <= error:
                convergent = True
            else:
                centroids = new_centroids
                clusters = {'c1':np.array([centroids[0]]), 'c2':np.array([centroids[1]]), 'c3':np.array([centroids[2]])}
        
        self.clusters = clusters
        
        return centroids



km = Kmean(X, K, error)
print(km.run())



