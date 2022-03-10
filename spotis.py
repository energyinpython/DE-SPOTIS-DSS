import numpy as np
from mcdm_method import MCDM_method

class SPOTIS(MCDM_method):
    def __init__(self):
        pass

    def __call__(self, matrix, weights, types, bounds):
        SPOTIS._verify_input_data(matrix, weights, types)
        return SPOTIS._spotis(matrix, weights, types, bounds)

    @staticmethod
    def _spotis(matrix, weights, types, bounds):
        isp = np.zeros(matrix.shape[1])

        #ideal solution point
        isp[types == 1] = bounds[1, types == 1]
        isp[types == -1] = bounds[0, types == -1]

        norm_matrix = np.abs(matrix - isp) / np.abs(bounds[1, :] - bounds[0, :])
        D = np.sum(weights * norm_matrix, axis = 1)
        return D
