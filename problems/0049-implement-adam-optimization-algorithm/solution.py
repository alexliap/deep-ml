import numpy as np

def adam_optimizer(f, grad, x0, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, num_iterations=10):
    # Note: `f` is accepted only for interface parity with the PyTorch/Tinygrad
    # variants, which derive gradients from it via autograd. This version uses
    # `grad` only — the objective value `f` is never evaluated.
    moment = np.zeros_like(x0)
    sec_moment = np.zeros_like(x0)
    for i in range(1, num_iterations + 1):
        g_t = grad(x0)

        moment = beta1*moment + (1-beta1)*g_t
        corr_moment = moment / (1 - beta1**i)
        
        sec_moment = beta2*sec_moment + (1-beta2)*(g_t**2)
        corr_sec_moment = sec_moment / (1 - beta2**i)
        
        x0 = x0 - learning_rate * corr_moment / (np.sqrt(corr_sec_moment) + epsilon)

    return x0