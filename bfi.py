
import jax
import jax.numpy as jnp
from xifa import GRM

jax.config.update("jax_enable_x64", True)

data = jnp.load("datasets/bfi.npy")
n_items = 300
n_factors = 30
q = int(n_items / n_factors)
pattern = {"labda": {m: set(range(m * q, (m + 1) * q)) for m in range(n_factors)}}
key = jax.random.PRNGKey(1)
grm = GRM(data, n_factors,
          pattern=pattern,
          weight=None,
          init_frac=None,
          key=None,
          verbose=None)

grm.fit(lr=1.,
        max_iter=1000,
        discard_iter=200,
        tol=10 ** (-4),
        window_size=3,
        verbose=True)
