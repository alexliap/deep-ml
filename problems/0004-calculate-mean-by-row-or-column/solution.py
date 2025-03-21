def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:

    means = []
    num_rows = len(matrix)
    num_columns = len(matrix[0])

    if mode == 'row':
        for row in matrix:
            means.append(sum(row)/len(row))

    elif mode == 'column':
        for i in range(num_columns):
            col_sum = 0
            for row in matrix:
                col_sum += row[i]

            means.append(col_sum/num_rows)

    return means