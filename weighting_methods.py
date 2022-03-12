import numpy as np
from correlations import *
from normalizations import *

# entropy weighting
def entropy_weighting(X):
    # normalization for profit criteria
    criteria_type = np.ones(np.shape(X)[1])
    pij = sum_normalization(X, criteria_type)
    pij = np.abs(pij)
    m, n = np.shape(pij)

    H = np.zeros((m, n))
    for j in range(n):
        for i in range(m):
            if pij[i, j] != 0:
                H[i, j] = pij[i, j] * np.log(pij[i, j])

    h = np.sum(H, axis = 0) * (-1 * ((np.log(m)) ** (-1)))
    d = 1 - h
    w = d / (np.sum(d))
    return w