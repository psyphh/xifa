import jax
import jax.numpy as jnp

from .mhrm import fit_mhrm
from .utils import cal_p12


class Base():
    def __init__(self,
                 data, n_factors,
                 patterns=None,
                 weight=None,
                 init_frac=None,
                 verbose=None,
                 key=None):
        if not isinstance(data, type(jnp.array)):
            data = jnp.array(data)
        n_cases = data.shape[0]
        n_items = data.shape[1]
        n_cats = int(jnp.nanmax(data) + 1)
        dtype = data.dtype
        y = jax.nn.one_hot(
            data, n_cats, dtype=dtype)
        del data
        if isinstance(patterns, type(None)):
            analysis = "exploratory"
            pattern = {"loading": None, "corr": None}
        else:
            analysis = "confirmatory"
            if "loading" not in patterns:
                patterns["loading"] = None
            if "corr" not in patterns:
                patterns["corr"] = None
        if isinstance(weight, type(None)):
            w = jnp.full(
                shape=(n_cases,),
                fill_value=1.,
                dtype=dtype)
        else:
            if not isinstance(weight, type(jnp.array)):
                w = jnp.array(
                    weight, dtype=dtype)
            else:
                w = weight
        del weight
        if isinstance(verbose, type(None)):
            verbose = True
        if isinstance(key, type(None)):
            key = jax.random.PRNGKey(0)
        if isinstance(init_frac, type(None)):
            p1, p2 = cal_p12(y, w)
        else:
            key, subkey = jax.random.split(key)
            init_idx = jax.random.choice(
                key=subkey,
                a=n_cases,
                shape=(int(n_cases * init_frac),),
                replace=False)
            p1, p2 = cal_p12(
                y[init_idx, ...],
                w[init_idx, ...])
        stats = {"p1": p1, "p2": p2}
        info = {"n_cases": n_cases,
                "n_items": n_items,
                "n_cats": n_cats,
                "n_factors": n_factors,
                "analysis": analysis,
                "patterns": patterns,
                "dtype": dtype}
        self.y, self.w = y, w
        self.info = info
        self.stats = stats
        self.key, self.verbose = key, verbose

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
        if jump_std == "default":
            jump_std = 2.4 / jnp.sqrt(self.info["n_factors"])
        if isinstance(verbose, type(None)):
            verbose = self.verbose
        if isinstance(key, type(None)):
            key = self.key
        if isinstance(params, type(None)):
            params = self.params
        if isinstance(masks, type(None)):
            masks = self.masks
        y, w = self.y, self.w
        eta = self.eta
        crf = self.crf
        params, aparams, eta, trace = fit_mhrm(
            lr=lr,
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
            masks=masks,
            y=y,
            eta=eta,
            w=w,
            crf=crf)
        self.params = params
        self.aparams = aparams
        self.eta = eta
        self.trace = trace
        if verbose:
            if self.trace["n_iters"] < max_iters:
                print("Converged after %.0f Iterations (%3.2f sec)." % (
                    self.trace["n_iters"], self.trace["fit_time"]))
            else:
                print("Not Converged after %.0f Iterations (%3.2f sec)." % (
                    self.trace["n_iters"], self.trace["fit_time"]))
        return self

    def print_init(self):
        if self.verbose:
            print("A", self.__class__.__name__,
                  "Object is Initialized for",
                  self.info["analysis"].capitalize(),
                  "Analysis.")
            print(" + Number of Cases: %.0f" % (self.info["n_cases"]))
            print(" + Number of Items: %.0f" % (self.info["n_items"]))
            print(" + Number of Factors: %.0f" % (self.info["n_factors"]))
            print(" + Number of Categories: %.0f" % (self.info["n_cats"]))


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
                self.masks["loading"] = jax.ops.index_update(
                    jnp.zeros(
                        (self.info["n_items"],
                         self.info["n_factors"]),
                        dtype=self.info["dtype"]),
                    (row_idx, col_idx), 1.)
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
                self.masks["corr"] = jax.ops.index_update(
                    jnp.zeros(
                        (self.info["n_factors"],
                         self.info["n_factors"]),
                        dtype=self.info["dtype"]),
                    (row_idx, col_idx), 1.)

    def init_params(self):
        self.params = {}

        def init_loading(p1, p2, n_factors):
            n_cats = p1.shape[1]
            k = jnp.arange(n_cats)
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
                self.key, self.masks["loading"].shape) * self.masks["loading"]
        self.params["corr"] = jnp.eye(
            self.info["n_factors"],
            dtype=self.info["dtype"])

    def init_eta(self):
        eta = self.y.argmax(axis=-1) @ self.params["loading"]
        eta = (eta - eta.mean(axis=0)) / eta.std(axis=0)
        self.eta = eta
