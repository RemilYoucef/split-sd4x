import numpy as np
import pandas as pd
import scipy
import math


def cal_covn(data_num, num_size, n) :
    
    cov_matrix = np.cov(data_num.T)
    cov_matrix[:num_size, :num_size] = cov_matrix[:num_size, :num_size] / n
    cov_matrix[:num_size, num_size:] = cov_matrix[:num_size, num_size:] / math.sqrt(n)
    cov_matrix[num_size:, :num_size] = cov_matrix[num_size:, :num_size] / math.sqrt(n)
    cov_matrix[num_size:, num_size:] = cov_matrix[num_size:, num_size:] / 1

    return cov_matrix


def generate_all_neighbors(data, data_compressed, n_neigh, numerical_cols, numerical_cols_compressed, categ_unique, categ_unique_compressed,n_var, model):
    
    list_neighs = []
    num_size = numerical_cols.size
    num_size_compressed = numerical_cols_compressed.size

    n = np.size(data, 0)
    covn = cal_covn(data, num_size, n_var)
    covn_compressed = cal_covn(data_compressed, num_size_compressed, n_var)

    base = np.zeros(data.shape[1])
    neighbors_base = np.random.multivariate_normal(base, covn, n_neigh)

    base_compressed = np.zeros(data_compressed.shape[1])
    neighbors_base_compressed = np.random.multivariate_normal(base_compressed, covn_compressed, n_neigh)

    for i in range(n):
        neighbors = neighbors_base + data[i]
        neighbors_compressed = neighbors_base_compressed + data_compressed[i]

        # for original neighbors 
        j = num_size
        for l in categ_unique :
            if l > 2 :
                neighbors[:,j:j+l] = (np.absolute(1-neighbors[:,j:j+l]).min(axis=1, keepdims=1) == np.absolute(1-neighbors[:,j:j+l])).astype(int)
                j = j + l

            else : #boolen
                neighbors[:,j] = neighbors[:,j].round()
                neighbors[:,j][neighbors[:,j] <= 0] = 0
                neighbors[:,j][neighbors[:,j] >= 1] = 1
                j = j + 1

        # for compressed neighbors
        k = num_size_compressed
        for l in categ_unique_compressed :
            if l > 2 :
                neighbors_compressed[:,k:k+l] = (np.absolute(1-neighbors_compressed[:,k:k+l]).min(axis=1, keepdims=1) == np.absolute(1-neighbors_compressed[:,k:k+l])).astype(int)
                k = k + l

            else : #boolen
                neighbors_compressed[:,k] = neighbors_compressed[:,k].round()
                neighbors_compressed[:,k][neighbors_compressed[:,k] <= 0] = 0
                neighbors_compressed[:,k][neighbors_compressed[:,k] >= 1] = 1
                k = k + 1

        neighbors[neighbors < 0] = 0
        neighbors_compressed [neighbors_compressed < 0] = 0
        target = model.predict(neighbors)
        list_neighs.append((neighbors_compressed, target))

    return list_neighs