def inverse_2x2(matrix: list[list[float]]) -> list[list[float]]:

    det = matrix[0][0]*matrix[1][1] - matrix[0][1]*matrix[1][0]

    if det == 0:
        return None
    else:
        inverse = []
        inverse.append([(1/det)*matrix[1][1], -(1/det)*matrix[0][1]])
        inverse.append([-(1/det)*matrix[1][0], (1/det)*matrix[0][0]])

    return inverse