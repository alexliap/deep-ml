import numpy as np

def numerical_gradient_check(f, x, analytical_grad, epsilon=1e-7):
    c_f_d = []

    for i in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()

        x_plus[i] += epsilon
        x_minus[i] -= epsilon

        c_f_d.append(
            (f(x_plus) - f(x_minus)) / (2 * epsilon)
        )

    numerical_grad = np.array(c_f_d)

    relative_error = (
        np.linalg.norm(numerical_grad - analytical_grad)
        / (np.linalg.norm(numerical_grad) + np.linalg.norm(analytical_grad))
    )

    return numerical_grad, relative_error