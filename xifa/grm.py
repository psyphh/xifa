import jax
import jax.numpy as jnp
from jax import jit

from .base import Ordinal


class GRM(Ordinal):
    def __init__(self,
                 data, n_factors,
                 pattern=None,
                 weight=None,
                 init_frac=None,
                 key=None,
                 verbose=None):
        super().__init__(
            data, n_factors,
            pattern, weight,
            init_frac, key)
        self.init_crf()
        self.init_masks()
        self.init_params()
        self.init_eta()
        if isinstance(verbose, type(None)):
            verbose = True
        if verbose:
            self.init_print()

    def init_crf(self):
        @jit
        def crf(eta, params):
            tau = -(eta @ params["labda"].T)[..., None] + params["nu"]
            cr_prob = jnp.diff(
                jax.scipy.special.expit(tau),
                axis=-1)
            return cr_prob

        self.crf = crf

    def init_masks(self):
        super().init_masks()
        self.masks["nu"] = jnp.hstack(
            [jnp.full(
                shape=(self.info["n_items"], 1),
                fill_value=0.,
                dtype=self.info["dtype"]),
                jnp.full(
                    shape=(self.info["n_items"],
                           self.info["n_cats"] - 1),
                    fill_value=1.,
                    dtype=self.info["dtype"]),
                jnp.full(
                    shape=(self.info["n_items"], 1),
                    fill_value=0.,
                    dtype=self.info["dtype"])])

    def init_params(self):
        def init_nu(p1):
            n_items = p1.shape[0]
            nu = jnp.hstack(
                [jnp.full((n_items, 1), -jnp.inf),
                 jax.scipy.special.logit(
                     jnp.cumsum(p1, axis=1)[:, :-1]),
                 jnp.full((n_items, 1), jnp.inf)])
            return nu

        super().init_params()
        self.params["nu"] = init_nu(
            self.stats["p1"])
        del self.stats
