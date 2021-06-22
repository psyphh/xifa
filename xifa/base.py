import jax
import jax.numpy as jnp

from .mhrm import fit_mhrm
from .utils import cal_p12




class Base():
    def __init__(self,
                 data, n_factors,
                 pattern=None,
                 weight=None,
                 init_frac=None,
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
        if isinstance(key, type(None)):
            key = jax.random.PRNGKey(0)
        if isinstance(init_frac, type(None)):
            p1, p2 = cal_p12(y, w)
        else:
            key, subkey = jax.random.split(key)
            init_idx = jax.random.choice(
                subkey, a=n_cases,
                shape=(int(n_cases * init_frac),),
                replace=False)
            p1, p2 = cal_p12(y[init_idx, ...], w[init_idx, ...])
        stats = {"p1": p1, "p2": p2}
        info = {"n_cases": n_cases, "n_factors": n_factors,
                "n_items": n_items, "n_cats": n_cats,
                "dtype": dtype, "pattern": pattern}
        self.y, self.w = y, w
        self.info = info
        self.stats = stats
        self.key = key

    def fit(self,
            lr=1.,
            max_iter=500,
            discard_iter=200,
            tol=10 ** (-4),
            window_size=3,
            chains=1,
            warm_up=5,
            jump_scale="default",
            adaptive_jump=True,
            target_rate=.23,
            sa_power=1.,
            cor_update="gd",
            verbose=True,
            key=None,
            batch_size=None,
            batch_shuffle=True,
            params=None,
            masks=None):
        if jump_scale == "default":
            jump_scale = jnp.sqrt(2.4 / (self.info["n_factors"] ** .5))
        if isinstance(key, type(None)):
            key = self.key
        if isinstance(params, type(None)):
            params = self.params
        if isinstance(masks, type(None)):
            masks = self.masks
        y, w = self.y, self.w
        eta = self.eta
        crf = self.crf
        params, aparams, eta3d, eta, trace = fit_mhrm(
            lr, max_iter, discard_iter,
            tol, window_size, chains, warm_up,
            jump_scale, adaptive_jump,
            target_rate, sa_power, cor_update,
            verbose, key,
            batch_size, batch_shuffle,
            params, masks,
            y, eta, w, crf)
        self.params = params
        self.aparams = aparams
        self.eta3d = eta3d
        self.eta = eta
        self.trace = trace
        if verbose:
            if self.trace["iter"] < max_iter:
                print("Converged after %.0f Iterations (%3.2f sec)." % (
                    self.trace["iter"], self.trace["time"]))
            else:
                print("Not Converged after %.0f Iterations (%3.2f sec)." % (
                    self.trace["iter"], self.trace["time"]))
        return self


    def init_print(self):
        print("A", self.__class__.__name__, "Object is Initialized Successfully.")
        print(" + Number of Cases: %.0f" % (self.info["n_cases"]))
        print(" + Number of Items: %.0f" % (self.info["n_items"]))
        print(" + Number of Factors: %.0f" % (self.info["n_factors"]))
        print(" + Number of Categories: %.0f" % (self.info["n_cats"]))


class Ordinal(Base):
    def init_masks(self):
        self.masks = {}
        if not isinstance(
                self.info["pattern"], type(None)):
            if "labda" not in self.info["pattern"]:
                self.info["pattern"]["labda"] = None
            if "phi" not in self.info["pattern"]:
                self.info["pattern"]["phi"] = None
        if isinstance(
                self.info["pattern"], type(None)):
            self.masks["labda"] = jnp.ones(
                (self.info["n_items"],
                 self.info["n_factors"]),
                dtype=self.info["dtype"])
        else:
            if isinstance(
                    self.info["pattern"]["labda"], type(None)):
                self.masks["labda"] = jnp.ones(
                    (self.info["n_items"],
                     self.info["n_factors"]),
                    dtype=self.info["dtype"])
            else:
                row_idx = []
                col_idx = []
                for key, values in self.info["pattern"]["labda"].items():
                    for value in values:
                        row_idx.append(value)
                        col_idx.append(key)
                self.masks["labda"] = jax.ops.index_update(
                    jnp.zeros(
                        (self.info["n_items"],
                         self.info["n_factors"]),
                        dtype=self.info["dtype"]),
                    (row_idx, col_idx), 1.)
        if isinstance(
                self.info["pattern"], type(None)):
            self.masks["phi"] = jnp.zeros(
                (self.info["n_factors"],
                 self.info["n_factors"]),
                dtype=self.info["dtype"])
        else:
            if isinstance(
                    self.info["pattern"]["phi"], type(None)):
                self.masks["phi"] = jnp.ones(
                    (self.info["n_factors"],
                     self.info["n_factors"]),
                    dtype=self.info["dtype"]) - jnp.eye(
                    self.info["n_factors"],
                    dtype=self.info["dtype"])
            else:
                row_idx = []
                col_idx = []
                for key, values in self.info["pattern"]["phi"].items():
                    for value in values:
                        if (key != value):
                            row_idx.append(value)
                            col_idx.append(key)
                            col_idx.append(value)
                            row_idx.append(key)
                self.masks["phi"] = jax.ops.index_update(
                    jnp.zeros(
                        (self.info["n_factors"],
                         self.info["n_factors"]),
                        dtype=self.info["dtype"]),
                    (row_idx, col_idx), 1.)

    def init_params(self):
        self.params = {}
        def init_labda(p1, p2, n_factors):
            n_cats = p1.shape[1]
            k = jnp.arange(n_cats)
            m1 = jnp.sum(p1 * k, axis=1)
            m2 = jnp.sum(p2 * jnp.outer(k, k)[None, :, None, :], axis=(1, 3))
            s = m2 - jnp.outer(m1, m1)
            d = jnp.sqrt(jnp.diagonal(s))
            r = s * jnp.outer(d, d)
            eigval, eigvec = jnp.linalg.eigh(r)
            labda = eigvec[:, -n_factors:]
            return labda

        if isinstance(
                self.info["pattern"], type(None)):
            self.params["labda"] = init_labda(
                self.stats["p1"],
                self.stats["p2"],
                self.info["n_factors"])
        else:
            self.params["labda"] = jax.random.uniform(
                self.key, self.masks["labda"].shape) * self.masks["labda"]
        self.params["phi"] = jnp.eye(
            self.info["n_factors"],
            dtype=self.info["dtype"])

    def init_eta(self):
        eta = self.y.argmax(axis=-1) @ self.params["labda"]
        eta = (eta - eta.mean(axis=0)) / eta.std(axis=0)
        self.eta = eta
