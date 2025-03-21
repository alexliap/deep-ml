def matrixmul(a:list[list[int|float]],
              b:list[list[int|float]])-> list[list[int|float]]:

    if len(a[0]) != len(b):
        return -1
    else:
        c = [[], [], []]
        # get the i-th row in a
        for i in range(len(a)):
            row_a = a[i]
            # get the j-th column from b
            for j in range(len(b[0])):
                column_b = [row[j] for row in b]

                value = 0
                for k in range(len(row_a)):
                    value += row_a[k] * column_b[k]
                
                c[i].append(value)

    return c
