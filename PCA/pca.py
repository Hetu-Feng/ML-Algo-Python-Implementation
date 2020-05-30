



import numpy as np
from numpy import linalg as linalg

k = 2
data = np.loadtxt('pca-data.txt')

# step1 normalize data
mean = np.mean(data, axis = 0)
normalized_data = data -mean  # 6000 x 3


#get covariance matrix
normalized_data_transposed = normalized_data.transpose()  # 3 x 6000
covariance = np.cov(normalized_data_transposed)    # 3 x 3

#get eigen values and eigen vectors
eigenvalues, eigenvectors = np.linalg.eig(covariance)

#sort and truncate
indices = np.argsort(-eigenvalues)
eigenvalues = eigenvalues[indices]
eigenvectors = eigenvectors[indices]   # 3 x 3

feature_vectors = eigenvectors[:,range(k)] #3 x 2

# get principal coponents data
z = np.matmul(normalized_data,feature_vectors)

print("Directoions of 2 components\n", feature_vectors.T)

