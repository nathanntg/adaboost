import numpy as np


def to_column_matrix(arr_or_mat):
    if len(arr_or_mat.shape) == 1:
        arr_or_mat.shape = [len(arr_or_mat), 1]
    elif np.shape(arr_or_mat)[0] == 1:
        return arr_or_mat.T
    return arr_or_mat
