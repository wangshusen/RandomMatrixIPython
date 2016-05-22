# define the RBF kernel function
import numpy as np

def rbf(matrixX1, matrixX2, sigma):
    n1 = matrixX1.shape[0]
    n2 = matrixX2.shape[0]
    K = np.dot(matrixX1, matrixX2.T)
    rowNormX1 = np.sum(np.square(matrixX1), 1) / 2
    rowNormX2 = np.sum(np.square(matrixX2), 1) / 2
    K = K - rowNormX1.reshape(n1, 1)
    K = K - rowNormX2.reshape(1, n2)
    K = K / (sigma**2)
    return np.exp(K)

