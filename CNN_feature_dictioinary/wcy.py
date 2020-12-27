import numpy as np
from sklearn.decomposition import PCA
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans

# PCA reduce the dimension based on the layer and the patch size
def RPCA(patch_size, datas):
	datas = datas.detach().numpy()
	#if layer == 'conv5_x':
		#if patch_size == 16:
			#n_dimension = 2217

		#elif patch_size == 32:
			#n_dimension = 3083

		#elif patch_size == 64:
			#n_dimension = 2633

		#else:
			#n_dimension = 1585

	#else:
	if patch_size == 16:
		n_dimension = 151

	elif patch_size == 32:
		n_dimension = 203

	elif patch_size == 64:
		n_dimension = 175

	else:
		n_dimension = 111

	pca = PCA(n_components = n_dimension)
	reduced_data_pca = pca.fit_transform(datas)
	return reduced_data_pca



# standardize the data reducted
def standardize(X): 
    m, n = X.shape 
    for j in range(n):
        features = X[:, j]
        meanVal = features.mean(axis=0)  
        std = features.std(axis=0) 
        if std != 0: 
            X[:, j] = (features - meanVal) / std
        else:
            X[:, j] = 0
    return X

# k_means to generate the dictionary
def clustering(centroids, datas):
	kmeans = KMeans(n_clusters = centroids, random_state = 0).fit(datas)
	code_dictionary = kmeans.cluster_centers_
	return code_dictionary
