import jax
import jax.numpy as jnp
from jax import jit

from .base import Ordinal




class GPCM(Ordinal):
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
            n_cats = params["intercept"].shape[1]
            logit = (eta @ params["loading"].T)[..., None] * jnp.arange(n_cats) + jnp.cumsum(params["intercept"], axis=1)
            cr_prob = jnp.exp(logit)
            cr_prob = cr_prob / jnp.sum(cr_prob, axis=2)[..., None]
            return cr_prob

        self.crf = crf

    def init_masks(self):
        super().init_masks()
        self.masks["intercept"] = jnp.hstack(
            [jnp.full(
                shape=(self.info["n_items"], 1),
                fill_value=0.,
                dtype=self.info["dtype"]),
                jnp.full(
                    shape=(self.info["n_items"],
                           self.info["n_cats"] - 1),
                    fill_value=1.,
                    dtype=self.info["dtype"])])

    def init_params(self):
        def init_intercept(p1):
            n_items = p1.shape[0]
            intercept = jax.scipy.special.logit(
                jnp.cumsum(p1, axis=1)[:, :-1])
            intercept = jnp.hstack(
                [jnp.full((n_items, 1), 0.),
                 intercept[:, 0:1],
                 jnp.diff(intercept, axis=1)])
            return intercept

        super().init_params()
        self.params["intercept"] = init_intercept(
            self.stats["p1"])
        del self.stats
