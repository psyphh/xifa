import jax
import jax.numpy as jnp
from jax import jit

from .base import Ordinal


class GPCM(Ordinal):
    def __init__(
            self,
            data, n_factors,
            patterns=None,
            freq=None,
            init_frac=None,
            verbose=None,
            key=None):
        super().__init__(
            data=data,
            n_factors=n_factors,
            patterns=patterns,
            freq=freq,
            init_frac=init_frac,
            verbose=verbose,
            key=key)
        self.init_crf()
        self.init_masks()
        self.init_params()
        self.init_eta()
        self.print_init()

    def fit(self,
            lr=1.,
            max_iters=500,
            stem_iters=200,
            tol=10 ** (-4),
            window=3,
            chains=1,
            warm_up=5,
            jump_std="default",
            jump_change=.01,
            target_rate=.23,
            gain_decay=1.,
            corr_update="gd",
            batch_size=None,
            batch_shuffle=None,
            verbose=None,
            key=None,
            params=None,
            masks=None):
        super().fit(lr=lr,
                    max_iters=max_iters,
                    stem_iters=stem_iters,
                    tol=tol,
                    window=window,
                    chains=chains,
                    warm_up=warm_up,
                    jump_std=jump_std,
                    jump_change=jump_change,
                    target_rate=target_rate,
                    gain_decay=gain_decay,
                    corr_update=corr_update,
                    batch_size=batch_size,
                    batch_shuffle=batch_shuffle,
                    verbose=verbose,
                    key=key,
                    params=params,
                    masks=masks)

    def init_crf(self):
        @jit
        def crf(eta, params):
            n_cats = params["intercept"].shape[1]
            logit = (eta @ params["loading"].T)[..., None] * jnp.arange(n_cats) + jnp.cumsum(params["intercept"],
                                                                                             axis=1)
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
