import jax
import jax.numpy as jnp

from .mhrm import fit_mhrm, conduct_mcmc
from .utils import cal_p12


class Base():
    def __init__(self,
                 data,
                 n_factors,
                 patterns=None,
                 freq=None,
                 init_frac=None,
                 verbose=None,
                 key=None):
        if not isinstance(data, type(jnp.array)):
            data = jnp.array(data)
        n_cases = data.shape[0]
        n_items = data.shape[1]
        ns_cats = list(jnp.nanmax(data, axis=0).astype("int") + 1)
        max_cats = max(ns_cats)
        dtype = data.dtype
        y = jax.nn.one_hot(
            x=data,
            num_classes=max_cats,
            dtype=dtype)
        del data
        if min(ns_cats) == max_cats:
            cats = "equal"
        else:
            cats = "unequal"
        if isinstance(patterns, type(None)):
            analysis = "exploratory"
            patterns = {"loading": None, "corr": None}
        else:
            analysis = "confirmatory"
            if "loading" not in patterns:
                patterns["loading"] = None
            if "corr" not in patterns:
                patterns["corr"] = None
        if isinstance(freq, type(None)):
            freq = jnp.full(
                shape=(n_cases,),
                fill_value=1.,
                dtype=dtype)
        else:
            if not isinstance(freq, type(jnp.array)):
                freq = jnp.array(
                    freq, dtype=dtype)
        if isinstance(verbose, type(None)):
            verbose = True
        if isinstance(key, type(None)):
            key = jax.random.PRNGKey(0)
        if isinstance(init_frac, type(None)):
            p1, p2 = cal_p12(y, freq)
        else:
            init_idx = jax.random.choice(
                key=key,
                a=n_cases,
                shape=(int(n_cases * init_frac),),
                replace=False)
            p1, p2 = cal_p12(
                y[init_idx, ...],
                freq[init_idx, ...])
        stats = {"p1": p1, "p2": p2}
        info = {"n_cases": n_cases,
                "n_items": n_items,
                "n_factors": n_factors,
                "ns_cats": ns_cats,
                "max_cats": max_cats,
                "analysis": analysis,
                "patterns": patterns,
                "cats": cats,
                "dtype": dtype}
        self.y, self.freq = y, freq
        self.info = info
        self.stats = stats
        self.key, self.verbose = key, verbose

    def print_init(self):
        if self.verbose:
            print("A", self.__class__.__name__,
                  "Object is Initialized for",
                  self.info["analysis"].capitalize(),
                  "Analysis.")
            print(" + Number of Cases: %.0f" % (self.info["n_cases"]))
            print(" + Number of Items: %.0f" % (self.info["n_items"]))
            print(" + Number of Factors: %.0f" % (self.info["n_factors"]))
            if self.info["cats"] == "equal":
                print(" + Number of Categories: %.0f" % (self.info["max_cats"]))
            else:
                print(" + Number of Categories: %.0f-%.0f" % (min(self.info["ns_cats"]), self.info["max_cats"]))


class Ordinal(Base):
    def init_masks(self):
        self.masks = {}
        if self.info["analysis"] == "exploratory":
            self.masks["loading"] = jnp.ones(
                (self.info["n_items"],
                 self.info["n_factors"]),
                dtype=self.info["dtype"])
        else:
            if isinstance(
                    self.info["patterns"]["loading"], type(None)):
                self.masks["loading"] = jnp.ones(
                    (self.info["n_items"],
                     self.info["n_factors"]),
                    dtype=self.info["dtype"])
            else:
                row_idx = []
                col_idx = []
                for key, values in self.info["patterns"]["loading"].items():
                    for value in values:
                        row_idx.append(value)
                        col_idx.append(key)
                self.masks["loading"] = jnp.zeros(
                        (self.info["n_items"],
                         self.info["n_factors"]),
                        dtype=self.info["dtype"])
                self.masks["loading"] = self.masks[
                    "loading"].at[(row_idx, col_idx)].set(1.)

        if self.info["analysis"] == "exploratory":
            self.masks["corr"] = jnp.zeros(
                (self.info["n_factors"],
                 self.info["n_factors"]),
                dtype=self.info["dtype"])
        else:
            if isinstance(
                    self.info["patterns"]["corr"], type(None)):
                self.masks["corr"] = jnp.ones(
                    (self.info["n_factors"],
                     self.info["n_factors"]),
                    dtype=self.info["dtype"]) - jnp.eye(
                    self.info["n_factors"],
                    dtype=self.info["dtype"])
            else:
                row_idx = []
                col_idx = []
                for key, values in self.info["patterns"]["corr"].items():
                    for value in values:
                        if (key != value):
                            row_idx.append(value)
                            col_idx.append(key)
                            col_idx.append(value)
                            row_idx.append(key)
                self.masks["corr"] = jnp.zeros(
                        (self.info["n_factors"],
                         self.info["n_factors"]),
                        dtype=self.info["dtype"])
                self.masks["corr"] = self.masks[
                    "corr"].at[(row_idx, col_idx)].set(1.)


    def init_params(self):
        self.params = {}

        def init_loading(p1, p2, n_factors):
            max_cats = p1.shape[1]
            k = jnp.arange(max_cats)
            m1 = jnp.sum(p1 * k, axis=1)
            m2 = jnp.sum(p2 * jnp.outer(k, k)[None, :, None, :], axis=(1, 3))
            s = m2 - jnp.outer(m1, m1)
            d = jnp.sqrt(jnp.diagonal(s))
            r = s * jnp.outer(d, d)
            eigval, eigvec = jnp.linalg.eigh(r)
            loading = eigvec[:, -n_factors:]
            return loading

        if self.info["analysis"] == "exploratory":
            self.params["loading"] = init_loading(
                self.stats["p1"],
                self.stats["p2"],
                self.info["n_factors"])
        else:
            self.params["loading"] = jax.random.uniform(
                key=self.key,
                shape=self.masks["loading"].shape,
                dtype=self.info["dtype"],
                minval=0.5,
                maxval=1.0) * self.masks["loading"]
        self.params["corr"] = jnp.eye(
            self.info["n_factors"],
            dtype=self.info["dtype"])

    def init_eta(self):
        eta = self.y.argmax(axis=-1) @ self.params["loading"]
        eta = (eta - eta.mean(axis=0)) / eta.std(axis=0)
        self.eta = eta

    def fit(
            self,
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
        if isinstance(lr, type(None)):
            lr = 1.
        if isinstance(max_iters, type(None)):
            max_iters = 700
        if isinstance(stem_steps, type(None)):
            stem_steps = 200
        if isinstance(warmup_steps, type(None)):
            warmup_steps = 150
        if isinstance(tol, type(None)):
            tol = 10 ** (-4)
        if isinstance(window_size, type(None)):
            window_size = 3
        if isinstance(n_chains, type(None)):
            n_chains = 1
        if isinstance(n_warmups, type(None)):
            n_warmups = 5
        if isinstance(jump_std, type(None)):
            jump_std = 2.4 / jnp.sqrt(self.info["n_factors"])
        if isinstance(jump_change, type(None)):
            jump_change = .01
        if isinstance(target_rate, type(None)):
            target_rate = (5 - min(self.info["n_factors"], 5)) / 4 * 0.44 + (min(self.info["n_factors"], 5) - 1) / 4 * .23
        if isinstance(gain_decay, type(None)):
            gain_decay = 1.0
        if isinstance(corr_update, type(None)):
            corr_update = "gd"
        if isinstance(verbose, type(None)):
            verbose = self.verbose
        if isinstance(key, type(None)):
            key = self.key
        if isinstance(params, type(None)):
            params = self.params
        if isinstance(masks, type(None)):
            masks = self.masks
        y, freq = self.y, self.freq
        eta = self.eta
        crf = self.crf
        if verbose:
            print("Fitting Process is Started.")
        params, aparams, eta, trace = fit_mhrm(
            lr=lr,
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
            masks=masks,
            y=y,
            eta=eta,
            freq=freq,
            crf=crf)
        self.params = params
        self.aparams = aparams
        self.eta = eta
        self.trace = trace
        if verbose:
            if self.trace["is_nan"]:
                print("`NaN` Occurs after %.0f Iterations (%.2f sec)." % (
                    self.trace["n_iters"], self.trace["fit_time"]))
                print("Possible Solutions Include:")
                print("+ Try Other `key`.")
                print("+ Try Smaller `lr`.")
                print("+ Set `corr_update='empirical'`.")
            else:
                if self.trace["is_converged"]:
                    print("Converged after %.0f Iterations (%.2f sec)." % (
                        self.trace["n_iters"], self.trace["fit_time"]))
                else:
                    print("Not Converged after %.0f Iterations (%.2f sec)." % (
                        self.trace["n_iters"], self.trace["fit_time"]))
                    print("Possible Solutions Include:")
                    print("+ Try Larger `max_iters`.")
                    print("+ Try Larger `n_chains`.")
        return self

    def transform(
            self,
            data=None,
            warmup_steps=None,
            n_chains=None,
            jump_std=None,
            batch_size=None,
            verbose=None,
            key=None):
        params = self.params
        crf = self.crf
        if isinstance(data, type(None)):
            y, freq, eta = self.y, self.freq, self.eta
            n_cases = y.shape[0]
        else:
            if not isinstance(data, type(jnp.array)):
                data = jnp.array(data)
            y = jax.nn.one_hot(
                x=data,
                num_classes=self.info["max_cats"],
                dtype=self.info["dtype"])
            n_cases = y.shape[0]
            del data
            freq = jnp.full(
                shape=(n_cases,),
                fill_value=1.,
                dtype=self.info["dtype"])
            eta = y.argmax(axis=-1) @ params["loading"]
            eta = (eta - eta.mean(axis=0)) / eta.std(axis=0)
        if isinstance(warmup_steps, type(None)):
            warmup_steps = 20
        if isinstance(n_chains, type(None)):
            n_chains = 100
        if isinstance(jump_std, type(None)):
            jump_std = self.trace["jump_std"][-1]
        if isinstance(verbose, type(None)):
            verbose = self.verbose
        if isinstance(key, type(None)):
            key = self.key
        key, subkey = jax.random.split(key)
        eta3d = jnp.repeat(eta[None, ...], n_chains, axis=0)
        jump_change, target_rate = 0, 0
        if isinstance(batch_size, type(None)):
            eta3d, accept_rate, jump_std = conduct_mcmc(
                subkey, warmup_steps,
                jump_std, jump_change, target_rate,
                eta3d, y, freq, params, crf)
        else:
            n_batches = int(jnp.ceil(n_cases / batch_size))
            batch_slices = [
                slice(batch_size * i, min(batch_size * (i + 1), n_cases), 1) for i in range(n_batches)]
            sum_freq = freq.sum()
            accept_rate = jnp.zeros(())
            key, subkey = jax.random.split(key)
            for batch_slice in batch_slices:
                y_batch, eta3d_batch, freq_batch = y[batch_slice, ...], eta3d[:, batch_slice, :], freq[batch_slice]
                key, subkey = jax.random.split(key)
                eta3d_batch, accept_rate_batch, jump_std = conduct_mcmc(
                    subkey, warmup_steps,
                    jump_std, jump_change, target_rate,
                    eta3d_batch, y_batch, freq_batch, params, crf)
                accept_rate = accept_rate + (freq_batch.sum() / sum_freq) * accept_rate_batch
                eta3d = eta3d.at[:, batch_slice, :].set(eta3d_batch)

        eta = jnp.mean(eta3d, axis=0)
        if verbose:
            print("Data are Transformed to Factor Scores.")
            print("+ Number of Chains: %.0f" % (n_chains))
            print("+ Number of Warmup Steps: %.0f" % (warmup_steps))
            print("+ Accept Rate: %.3f" % (accept_rate))
        return eta

    def loglik(self,
               n_points=None,
               batch_size=None,
               individual=None,
               verbose=None,
               key=None):
        if isinstance(n_points, type(None)):
            n_points = 5000
        if isinstance(individual, type(None)):
            individual = False
        if isinstance(verbose, type(None)):
            verbose = self.verbose
        if isinstance(key, type(None)):
            key = self.key
        n_cases, n_factors = self.info["n_cases"], self.info["n_factors"]
        y, freq = self.y, self.freq
        params = self.params
        crf = self.crf
        points = jax.random.multivariate_normal(
            key=key,
            mean=jnp.zeros(
                shape=(n_factors,)),
            cov=params["corr"],
            shape=(n_points,))
        if isinstance(batch_size, type(None)):
            loglik_i = jnp.log(
                    jnp.mean(
                        jnp.prod(
                            crf(points, params) ** y[:, jnp.newaxis, ...],
                            axis=(-1, -2)),
                        axis=-1))
            loglik = jnp.sum(
                 loglik_i * freq) / freq.sum()
        else:
            n_batches = int(jnp.ceil(n_cases / batch_size))
            batch_slices = [
                slice(batch_size * i, min(batch_size * (i + 1), n_cases), 1) for i in range(n_batches)]
            sum_freq = freq.sum()
            loglik_i = []
            loglik = jnp.zeros(())
            for batch_slice in batch_slices:
                y_batch, freq_batch = y[batch_slice, ...], freq[batch_slice]
                sum_freq_batch = freq_batch.sum()
                loglik_i_batch = jnp.log(
                        jnp.mean(
                            jnp.prod(
                                crf(points, params) ** y_batch[:, jnp.newaxis, ...],
                                axis=(-1, -2)),
                            axis=-1))
                loglik_i.append(loglik_i_batch)
                loglik_batch = jnp.sum(
                    loglik_i_batch * freq_batch) / sum_freq_batch
                loglik = loglik + (sum_freq_batch / sum_freq) * loglik_batch
        if verbose:
            print("Log-Likelihood is Calculated (%.0f Quadrature Points)." % (n_points) )
        if individual:
            loglik_i = jnp.hstack(loglik_i)
            return loglik, loglik_i
        else:
            return loglik