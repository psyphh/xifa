"""
Generalized Partial Credit Models
"""

import jax
import jax.numpy as jnp
from jax import jit

from .base import Ordinal




class GPCM(Ordinal):
    """A Class for Fitting Generalized Partial Credit Models (GPCMs)

     Attributes:
         y (jax.numpy.ndarray):
             A 3D array with shape `(n_cases, n_items, max_cats)` to represent one-hot encoding data for IFA.
         freq (jax.numpy.ndarray):
             A 1D array with shape `(n_cases,)` to represent frequencies for rows of data.
         verbose (bool):
             A `bool` to specify whether model information should be printed after successful initialization.
         key (numpy.ndarray-like):
             A pseudorandom number generator (PRNG) key for random number generation.
         info (list):
             A `list` to represent model information.
         crf (funtion):
             A `function` to calculate category responses.
         params (list):
             A `list` for model parameter matrices.
             `params` includes three parameter arrays:
             (1) `params[`intercept`]` is an intercept matrix with shape `(n_items, max_cats + 1)`;
             (2) `params[`loading`]` is a loading matrix with shape `(n_items, n_factors)`;
             (3) `params[`corr`]` is a correlation matrix with shape `(n_factors, n_factors)`.
         masks (list):
             A `list` for model parameter masks.
             `masks` includes three arrays with only 0/1 value:
             (1) `masks['intercept']` is a mask matrix for intercept matrix;
             (2) `masks['loading']` is a mask matrix for loading matrix;
             (3) `masks['corr']` is a mask matrix for correlation matrix.
             If an element in `masks` is zero, its corresponding element in `params` will not be updated in estimation.
         trace (list):
             A `list` to record the fitting history made by `fit()` method.
             `trace` includes eight elements:
             (1) `trace['accept_rate']` is a `list` storing acceptance rates in Metropolis-Hasting sampling.
             (2) `trace['closs']` is a `list` storing complete data loss values (negative complete data log-likelihood).
             (3) `trace['change_params']` is a float for the maximal change in parameters.
             (4) `trace['jump_std']` is a `list` storing values of jumping standard deviations used in MCMC sampling.
             (5) `trace['n_iters']` is a `float` for the number of iterations.
             (6) `trace['is_converged']` is a `bool` to indicate whether the algorithm is converged.
             (7) `trace['is_nan']` is a `bool` to indicate whether the algorithm results in NaN values.
             (8) `trace['fit_time']` is a `float` for the fitting time.
             `trace` is only available after using `fit()` method.
         eta (jax.numpy.ndarray):
             A 2D array with shape `(n_cases, n_factors)` for predicted `eta` values .
             The values of `eta` elements are calculated by averaging values of Metropolis-Hasting samples across `n_chains` specified in `fit()`.
             Therefore, it is not appropriate to directly use `eta` when `n_chains` is small.
             `eta` is only available after using `fit()` method.
         aparams (list):
             A `list` for averaged model parameter matrices.
             The averaging is with respect to all iterations in the stochastic approximation stage.
             `aparams` includes three parameter arrays:
             (1) `aparams['intercept']` is an intercept matrix with shape `(n_items, max_cats + 1)`;
             (2) `aparams['loading']` is a loading matrix with shape `(n_items, n_factors)`;
             (3) `aparams['corr']` is a correlation matrix with shape `(n_factors, n_factors)`.
             `aparams` is only available after using `fit()` method.
     """
    def __init__(
            self,
            data,
            n_factors,
            patterns=None,
            freq=None,
            init_frac=None,
            verbose=None,
            key=None):

        """__init__ Method for GPCM Class

        Args:
            data (numpy.ndarray-like):
                A float 2D array with shape `(n_cases, n_items)` to specify data for IFA.
                Its element must be a floating point with value from 0 to `max_cats` - 1,
                where `max_cats` is the maximal number of categories among items.
                A missing value must be represented by `jax.numpy.nan` or `numpy.nan`.
            n_factors (int):
                An `int` to specify the number of factors in IFA.
            patterns (list, optional):
                A `list` to specify patterns for confirmatory analysis.
            freq (numpy.ndarray-like, optional):
                A 1D array with shape `(n_cases,)` to specify frequencies for rows of data.
                By default, `freq` is a 1D array of ones with shape `(n_cases,)`.
            init_frac (float, optional):
                A `float` between 0 and 1 to specify the percentage of cases for calculating sample statistics.
                The sample statistics is useful for initializing parameter values.
                `init_frac` is only required when data is too big with respect to the available memory.
                By default, `init_frac` is `1.0` (i.e., using all cases for computing sample statistics).
            verbose (bool, optional):
                A `bool` to specify whether model information should be printed after successful initialization.
                By default, `verbose` is `True`.
            key (numpy.ndarray-like, optional):
                A pseudorandom number generator (PRNG) key for random number generation.
                It can be generated by using `jax.random.PRNGKey(seed)`, where seed is an integer.
                In the initialization stage, `key` is only used if `init_frac` is specified.
                However, `key` will be largely used in `fit()` method (`key` can be also specified as an argument of `fit()`).
                By default, `key` is set by `jax.random.PRNGKey(0)`.
        """
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
            lr=None,
            max_iters=None,
            stem_steps=None,
            warmup_steps=None,
            tol=None,
            window_size=None,
            n_chains=None,
            n_warmups=None,
            jump_std=None,
            jump_change=None,
            target_rate=None,
            gain_decay=None,
            corr_update=None,
            batch_size=None,
            verbose=None,
            key=None,
            params=None,
            masks=None):
        """Fit Method for GPCM Class

       Args:
            lr (float, optional):
                A `float` for learning rate (or step size) for gradient descent.
                By default, `lr` is `1.0`.
            max_iters (int, optional):
                An `int` to specify the maximal number of iterations for the fitting process, that is,
                the sum of expectation-maximization (StEM) steps and stochastic approximation (SA) steps.
                By default, `max_iters` is `700`.
            stem_steps (int, optional):
                An `int` for the number of steps for the stochastic expectation-maximization (StEM) algorithm.
                StEM is used in the first stage of fitting.
                Note that `stem_steps` is also counted in `max_iters`.
                By default, `stem_steps` is `200`.
            warmup_steps (int, optional):
                An `int` for the number of steps for the warmup steps when intializing MCMC sampling.
                By default, `warmup_steps` is `150`.
            tol (float, optional):
                A `float` to specify the tolerance for checking the convergence of stochastic approximation (SA) stage.
                SA is the second stage of fitting.
                The SA stage stops when the maximal value of changes in parameters is smaller than `tol` within window size defined in `window_size`.
                By default, `tol` is `10**(-4)`.
            window_size (int, optional):
                An `int` for the window size to check convergence.
                By default, `window_size` is `3`.
            n_chains (int, optional):
                An `int` to specify how many independent chains are established in MCMC sampling.
                The last sample of each chain will be used to construct the complete-data likelihood.
                In other words, the complete-data likelihood is approximated by `n_chains` samples for each case.
                By default, `n_chains` is `1`.
            n_warmups (int, optional):
                An `int` to specify the number of warmup iterations when conducting MCMC in each iteration.
                In other words, the `n_warmups+1`(th) sample is considered as a valid MCMC sample.
                By default, `n_warmups` is `5`.
            jump_std (float, optional):
                A `float` for the jumping standard deviation for Metropolis-Hasting sampling.
                By default, `jump_std` is set as `2.4/sqrt(n_factors)`.
            jump_change (float, optional):
                A `float` to specify the change value for adaptive `jump_std`.
                Setting `jump_change=0` implies a fixed `jump_std`.
                By default, `jump_change` is `.01`.
            target_rate (float, optional):
                A `float` to specify an optimal value of acceptance rate for Metropolis-Hasting sampling.
                By default, `target_rate` is `(5-min(n_factors,5))/4*0.44+(min(n_factors,5)-1)/4*.23`.
            gain_decay (float, optional):
                A `float` to specify the decay level of gain in stochastic approximation update.
                The gain is calculated by `gain = 1 / ((n_iters - stem_steps)**gain_decay)`,
                where `n_iters` is the number of current iteration.
                By default, `gain_decay` is `1.0`.
            corr_update (str, optional):
                A `str` to specify the method for updating correlation matrix of latent factors.
                Its value must be one of `['gd', 'gd_bls', 'empirical']`.
                Here, `gd` is for gradient descent,
                `gd_bls` is for gradient descent with backtracking line search,
                and `empirical` is for empirical covariance calculated by MCMC samples.
                By default, `corr_update` is `gd`.
            batch_size (int, optional):
                An `int` to specify the batch size if mini-batch stochastic gradient descent is used.
                If `batch_size=n_cases`, it results in usual gradient descent method.
                By default, `batch_size` is `n_cases`.
            verbose (bool, optional):
                A `bool` to specify whether fitting summary and progress bar should be printed after successful initialization.
                By default, `verbose` is the value specified in `__init__()`.
            key (numpy.ndarray-like, optional):
                A pseudorandom number generator (PRNG) key for random number generation.
                It can be generated by using `jax.random.PRNGKey(seed)`, where seed is an integer.
                By default, `key`  is the value yielded by `__init__()`.
        """
        super().fit(lr=lr,
                    max_iters=max_iters,
                    stem_steps=stem_steps,
                    warmup_steps=warmup_steps,
                    tol=tol,
                    window_size=window_size,
                    n_chains=n_chains,
                    n_warmups=n_warmups,
                    jump_std=jump_std,
                    jump_change=jump_change,
                    target_rate=target_rate,
                    gain_decay=gain_decay,
                    corr_update=corr_update,
                    batch_size=batch_size,
                    verbose=verbose,
                    key=key,
                    params=params,
                    masks=masks)

    def init_crf(self):
        @jit
        def crf(eta, params):
            max_cats = params["intercept"].shape[1]
            logit = (eta @ params["loading"].T)[..., None] * jnp.arange(max_cats) + jnp.cumsum(params["intercept"], axis=1)
            cr_prob = jnp.exp(logit)
            cr_prob = cr_prob / jnp.sum(cr_prob, axis=-1)[..., None]
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
                        self.info["max_cats"] - 1),
                 fill_value=1.,
                 dtype=self.info["dtype"])])
        if self.info["cats"] == "unequal":
            row_idx = []
            col_idx = []
            for i, n_cats in enumerate(self.info["ns_cats"]):
                for j in range(n_cats, self.info["max_cats"]):
                    row_idx.append(i)
                    col_idx.append(j)
            self.masks["intercept"] = jax.ops.index_update(
                x=self.masks["intercept"],
                idx=(row_idx, col_idx),
                y=0.)

    def init_params(self):
        def init_intercept(p1):
            n_items = p1.shape[0]
            intercept = jax.scipy.special.logit(
                jnp.cumsum(p1, axis=1)[:, :-1])
            intercept = jnp.hstack(
                [jnp.full(
                    shape=(n_items, 1),
                    fill_value=0.,
                    dtype=self.info["dtype"]),
                 intercept[:, 0:1],
                 jnp.diff(intercept, axis=1)])
            return intercept

        super().init_params()
        self.params["intercept"] = init_intercept(
            self.stats["p1"])
        del self.stats
        if self.info["cats"] == "unequal":
            row_idx = []
            col_idx = []
            for i, n_cats in enumerate(self.info["ns_cats"]):
                for j in range(n_cats, self.info["max_cats"]):
                    row_idx.append(i)
                    col_idx.append(j)
            self.params["intercept"] = jax.ops.index_update(
                x=self.params["intercept"],
                idx=(row_idx, col_idx),
                y=-jnp.inf)