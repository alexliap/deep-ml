def scalar_multiply(matrix: list[list[int|float]], scalar: int|float) -> list[list[int|float]]:

    for i in range(len(matrix)):
        matrix[i] = [value * scalar for value in matrix[i]]

    return matrix