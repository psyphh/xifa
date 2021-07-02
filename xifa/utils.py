import jax
import jax.numpy as jnp
from jax import jit, vmap


@jit
def cal_multinomial_logpmf(k, p):
    multinomial_logpmf = jax.scipy.special.xlogy(k, p).sum(axis=-1)
    return multinomial_logpmf


@jit
def cal_normal_logpdf(x, mean, cov):
    l = jax.lax.linalg.cholesky(jnp.linalg.inv(cov))
    z = (x - mean) @ l
    normal_logpdf = -0.5 * (z.shape[-1] * jnp.log(2 * jnp.pi) + jnp.square(z).sum(-1)) + jnp.sum(jnp.log(l.diagonal()))
    return normal_logpdf


@jit
def cal_p12(y, freq):
    y_freq = y * freq[..., None, None]
    f1 = jnp.sum(y_freq, axis=0) + 1.
    p1 = f1 / jnp.sum(f1, axis=1)[..., None]
    f2 = vmap(lambda x_i: y_freq.T @ x_i, in_axes=(1,))(y)
    p2 = f2 / jnp.sum(f2, axis=(1, 3))[:, None, :, None]
    return p1, p2
