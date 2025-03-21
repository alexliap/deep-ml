def calculate_eigenvalues(matrix: list[list[float|int]]) -> list[float]:
    # the equation is of the form: ax^2+bx+c=0
    # so let's calculat all the variables a, b and c
    a = 1

    b = -(matrix[0][0] + matrix[1][1])

    c = matrix[0][0]*matrix[1][1] - matrix[1][0]*matrix[0][1]

    eigenvalues = []

    x1 = (-b + (b**2 - 4*a*c)**(0.5)) / (2 * a)
    x2 = (-b - (b**2 - 4*a*c)**(0.5)) / (2 * a)

    eigenvalues.append(x1)
    eigenvalues.append(x2)

    return eigenvalues