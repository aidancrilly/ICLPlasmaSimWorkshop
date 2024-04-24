import jax
from jax import numpy as jnp
from jaxtyping import Array, Float, Int
from typing import Any, Callable, Mapping, Optional

# From https://github.com/google/jax/issues/14101

def legendre_recurrence(
    x: Float[Array, "m"], n_max: Int[Array, ""]
) -> Float[Array, "n m"]:
    """
    Compute the Legendre polynomials up to degree n_max at a given point or array of points x.

    The function employs the recurrence relation for Legendre polynomials. The Legendre polynomials
    are orthogonal on the interval [-1,1] and are used in a wide array of scientific and mathematical applications.
    This function returns a series of Legendre polynomials evaluated at the point(s) x, up to the degree n_max.

    Args:
        n_max (int): The highest degree of Legendre polynomial to compute. Must be a non-negative integer.
        x (jnp.ndarray): The point(s) at which the Legendre polynomials are to be evaluated. Can be a single
                        point (float) or an array of points.

    Returns:
        jnp.ndarray: A sequence of Legendre polynomial values of shape (n_max+1,) + x.shape, evaluated at point(s) x.
                    The i-th entry of the output array corresponds to the Legendre polynomial of degree i.

    Notes:
        The first two Legendre polynomials are initialized as P_0(x) = 1 and P_1(x) = x. The subsequent polynomials
        are computed using the recurrence relation:
        P_{n+1}(x) = ((2n + 1) * x * P_n(x) - n * P_{n-1}(x)) / (n + 1).
    """

    p_init = jnp.zeros((2,) + x.shape)
    p_init = p_init.at[0].set(1.0)  # Set the 0th degree Legendre polynomial
    p_init = p_init.at[1].set(x)  # Set the 1st degree Legendre polynomial

    def body_fun(carry, _):
        i, (p_im1, p_i) = carry
        p_ip1 = ((2 * i + 1) * x * p_i - i * p_im1) / (i + 1)

        return ((i + 1).astype(int), (p_i, p_ip1)), p_ip1

    (_, (_, _)), p_n = jax.lax.scan(
        f=body_fun, init=(1, (p_init[0], p_init[1])), xs=(None), length=(n_max - 1)
    )
    p_n = jnp.concatenate((p_init, p_n), axis=0)

    return p_n