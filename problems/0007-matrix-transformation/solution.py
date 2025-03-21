import numpy as np

def check_invertible(matrix: list[list[int|float]]) -> bool:
    matrix = np.array(matrix)

    if matrix.shape[0] == matrix.shape[1]:
        if np.linalg.det(matrix) == 0:
            return False
        else:
            return True
    else:
        return False


def transform_matrix(A: list[list[int|float]], T: list[list[int|float]], S: list[list[int|float]]) -> list[list[int|float]]:

    # check if T & S are invertible
    if check_invertible(T) and check_invertible(S):
        A = np.matrix(A)
        T = np.matrix(T)
        S = np.matrix(S)
        
        transformed_matrix = np.linalg.inv(T) * A * S
    else:
        return -1

    return transformed_matrix


