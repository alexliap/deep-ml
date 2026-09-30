import jax
import jax.numpy as jnp

def df_dx(x):
    """Derivative of f(x) = x**3 + 2x at float x, via jax.grad. Returns float."""
    f = jax.grad(lambda x: x**3 + 2*x)
    return f(x)
