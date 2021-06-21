import time
from functools import partial

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from tqdm.auto import tqdm

from .utils import cal_multinomial_logpmf


@partial(jit, static_argnums=(4,))
def cal_closs2d_i(params, y, eta2d, w, crf):
    closs_i = cal_floss2d_i(params, y, eta2d, w, crf) + cal_bloss2d_i(params, eta2d, w)
    return closs_i


@partial(jit, static_argnums=(4,))
def cal_closs2d(params, y, eta2d, w, crf):
    closs = cal_closs2d_i(params, y, eta2d, w, crf).sum() / w.sum()
    return closs


@partial(jit, static_argnums=(4,))
def cal_dcloss2d(params, y, eta2d, w, crf):
    dcloss = grad(cal_closs2d, argnums=0)(params, y, eta2d, w, crf)
    return dcloss


@partial(jit, static_argnums=(4,))
def cal_dcloss2d_i(params, y, eta2d, w, crf):
    dcloss_i = jax.jacfwd(cal_closs2d_i, argnums=0)(params, y, eta2d, w, crf)
    return dcloss_i


@partial(jit, static_argnums=(4,))
def cal_d2closs2d(params, y, eta2d, w, crf):
    return jax.jacfwd(cal_dcloss2d, argnums=0)(params, y, eta2d, w, crf)


@partial(jit, static_argnums=(4,))
def cal_closs3d_i(params, y, eta3d, w, crf):
    closs_i = vmap(cal_closs2d_i,
                   in_axes=(None, None, 0, None, None))(
        params, y, eta3d, w, crf)
    return closs_i


@partial(jit, static_argnums=(4,))
def cal_closs3d(params, y, eta3d, w, crf):
    closs = cal_closs3d_i(params, y, eta3d, w, crf).mean(axis=0).sum() / w.sum()
    return closs


@partial(jit, static_argnums=(4,))
def cal_dcloss3d(params, y, eta3d, w, crf):
    dcloss = grad(cal_closs3d, argnums=0)(params, y, eta3d, w, crf)
    return dcloss


@partial(jit, static_argnums=(4,))
def cal_dcloss3d_i(params, y, eta3d, w, crf):
    dcloss_i = vmap(cal_dcloss2d_i,
                    in_axes=(None, None, 0, None, None))(
        params, y, eta3d, w, crf)
    return dcloss_i


@partial(jit, static_argnums=(4,))
def cal_d2closs3d(params, y, eta3d, w, crf):
    d2closs = jax.jacfwd(cal_dcloss3d, argnums=0)(params, y, eta3d, w, crf)
    return d2closs


@partial(jit, static_argnums=(4,))
def cal_floss2d_i(params, y, eta2d, w, crf):
    floss_i = -jnp.sum(
        cal_multinomial_logpmf(
            k=y,
            p=crf(eta2d, params)),
        axis=-1) * w
    return floss_i


@jit
def cal_bloss2d_i(params, eta2d, w):
    n_factors = eta2d.shape[-1]
    bloss_i = - jax.scipy.stats.multivariate_normal.logpdf(
        x=eta2d,
        mean=jnp.zeros(n_factors),
        cov=params["phi"]) * w
    return bloss_i


@jit
def cal_bloss3d_i(params, eta3d, w):
    bloss_i = vmap(cal_bloss2d_i,
                   in_axes=(None, 0, None))(
        params, eta3d, w)
    return bloss_i


@jit
def cal_bloss3d(params, eta3d, w):
    bloss = cal_bloss3d_i(params, eta3d, w).mean(axis=0).sum() / w.sum()
    return bloss


def adjust_jump_scale(jump_scale, accept_rate, target_rate):
    if accept_rate > (target_rate + .01):
        jump_scale = jump_scale + .01
    elif accept_rate < (target_rate - .01):
        jump_scale = jump_scale - .01
    else:
        pass
    return jump_scale


def set_timers(max_iter, discard_iter):
    timer1 = tqdm(
        total=discard_iter,
        disable=False,
        position=0)
    timer2 = tqdm(
        total=max_iter - discard_iter,
        disable=False,
        position=1)
    return timer1, timer2


def update_timers(timer1, timer2, iter_, discard_iter, trace):
    if iter_ <= discard_iter:
        timer1.set_postfix(
            {"Accept Rate": jnp.round(trace["accept_rate"][-1], 2),
             'Loss': jnp.round(trace["closs"][-1], 2)})
        timer1.set_description("Stage 1")
        timer1.update(1)
    else:
        timer2.set_postfix(
            {"Accept Rate": jnp.round(trace["accept_rate"][-1], 2),
             'Loss': jnp.round(trace["closs"][-1], 2)})
        timer2.set_description("Stage 2")
        timer2.update(1)


@partial(jit, static_argnums=(7,))
def conduct_mcmc(key, warm_up, jump_scale,
                 eta, y, params, w, crf):
    accept_rate = 0
    n_factors = eta.shape[-1]

    def sample_eta(i, value):
        key, eta, accept_rate = value
        key, subkey = jax.random.split(key)
        eta_new = jax.random.multivariate_normal(
            key=subkey,
            mean=eta,
            cov=jnp.diag(jnp.repeat(jump_scale ** 2, n_factors)),
            shape=eta.shape[:-1])
        ratio = jnp.exp(
            - cal_closs3d_i(params, y, eta_new, w, crf) + cal_closs3d_i(params, y, eta, w, crf))
        key, subkey = jax.random.split(key)
        accept = jax.random.bernoulli(
            key=subkey,
            p=jnp.minimum(ratio, 1))
        eta = jnp.where(
            jnp.repeat(
                accept[..., None],
                n_factors,
                axis=-1),
            x=eta_new,
            y=eta)
        accept_rate = accept.mean()
        return key, eta, accept_rate

    key, eta, accept_rate = jax.lax.fori_loop(
        0, warm_up + 1, sample_eta, (key, eta, accept_rate))
    return eta, accept_rate


def cal_dparams_batch(batch_slices, params, y, eta3d, w, crf):
    sum_w = w.sum()
    dparams = {key: jnp.zeros(value.shape) for key, value in params.items()}
    for batch_slice in batch_slices:
        y_batch, eta3d_batch, w_batch = y[batch_slice, ...], eta3d[:, batch_slice, ...], w[batch_slice],
        dparams_batch = cal_dcloss3d(
            params, y_batch, eta3d_batch, w_batch, crf)
        dparams = {key: dparams[key] + dparams_batch[key] * (w[batch_slice].sum() / sum_w) for key in dparams}
    return dparams


@jit
def cal_cov_eta3d(eta3d):
    cov_eta = jnp.mean(
        vmap(lambda eta:
             eta.T @ eta / eta.shape[0], in_axes=0)(eta3d),
        axis=0)
    return cov_eta


@jit
def ls_cor(lr, gain, params, dparams, masks, eta3d, w):
    c_armijo = 0.001
    d_armijo = 1.0
    lr_armijo = 0
    iter_armijo = 0
    phi = params["phi"]
    bloss = cal_bloss3d(params, eta3d, w)
    value = (d_armijo, params, lr_armijo, iter_armijo)

    def body_fun(value):
        d_armijo, params, lr_armijo, iter_armijo = value
        lr_armijo = (0.5 ** iter_armijo) * lr
        params["phi"] = phi - lr_armijo * gain * dparams["phi"] * masks["phi"]
        d_armijo = cal_bloss3d(params, eta3d, w) - bloss + c_armijo * lr_armijo * jnp.sum(dparams["phi"] ** 2)
        iter_armijo = iter_armijo + 1
        return d_armijo, params, lr_armijo, iter_armijo

    def cond_fun(value):
        d_armijo, params, lr_armijo, iter_armijo = value
        cond = d_armijo > 0.
        return cond

    d_armijo, params, lr_armijo, iter_armijo = jax.lax.while_loop(cond_fun, body_fun, value)
    return params["phi"], lr_armijo


@jit
def rescale_params(params, masks):
    scale = jnp.sqrt(params["phi"].diagonal())
    params["phi"] = params["phi"] / jnp.outer(scale, scale)
    params["phi"] = params["phi"] * masks["phi"] + jnp.diag(params["phi"].diagonal())
    params["labda"] = params["labda"] * scale
    return params


def update_params(lr, gain, cor_update,
                  params, dparams, masks,
                  y, eta3d, w, crf):
    params["nu"] = params["nu"] - lr * gain * dparams["nu"] * masks["nu"]
    params["labda"] = params["labda"] - lr * gain * dparams["labda"] * masks["labda"]
    if jnp.sum(masks["phi"]) >= 1.0:
        if cor_update == "heuristic":
            params["phi"] = cal_cov_eta3d(eta3d)
            params = rescale_params(params, masks)
            dparams["phi"] = dparams["phi"] * .0
        else:
            if gain >= 1.:
                params["phi"] = cal_cov_eta3d(eta3d)
                params = rescale_params(params, masks)
            else:
                dparams["phi"] = dparams["phi"] + dparams["phi"].T - jnp.diag(dparams["phi"].diagonal())
                if cor_update == "gd_ls":
                    params["phi"], lr_armijo = ls_cor(
                        lr, gain, params, dparams, masks, eta3d, w)
                    dparams["phi"] = dparams["phi"] * lr_armijo / lr
                elif cor_update == "gd":
                    params["phi"] = params["phi"] - lr * gain * dparams["phi"] * masks["phi"]
                else:
                    dparams["phi"] = dparams["phi"] * .0
    else:
        dparams["phi"] = dparams["phi"] * .0
    return params, dparams


def fit_mhrm(lr,
             max_iter,
             discard_iter,
             tol,
             window_size,
             chains,
             warm_up,
             jump_scale,
             adaptive_jump,
             target_rate,
             sa_power,
             cor_update,
             verbose,
             key,
             batch_size,
             params,
             masks,
             y,
             eta,
             w,
             crf):
    start = time.time()
    eta3d = jnp.repeat(eta[None, ...], chains, axis=0)
    if not isinstance(batch_size, type(None)):
        n_cases = y.shape[1]
        n_batches = jnp.ceil(n_cases / batch_size)
        batch_slices = [
            slice(batch_size * i, min(batch_size * (i + 1), n_cases), 1) for i in range(n_batches)]
    aparams = {key: jnp.zeros(value.shape) for key, value in params.items()}
    trace = {"accept_rate": [], "closs": [], "delta_params": []}
    converged = False
    if verbose:
        timer1, timer2 = set_timers(max_iter, discard_iter)
    for iter_ in range(1, max_iter + 1):
        if iter_ > discard_iter:
            sa_count = (iter_ - discard_iter)
            gain = 1. / (sa_count ** sa_power)
        else:
            gain = 1.
        key, subkey = jax.random.split(key)
        eta3d, accept_rate = conduct_mcmc(
            subkey, warm_up, jump_scale,
            eta3d, y, params, w, crf)
        if isinstance(batch_size, type(None)):
            dparams = cal_dcloss3d(
                params, y, eta3d, w, crf)
        else:
            dparams = cal_dparams_batch(
                batch_slices, params,
                y, eta3d, w, crf)
        params, dparams = update_params(
            lr, gain, cor_update,
            params, dparams, masks,
            y, eta3d, w, crf)
        delta_params = max(
            [jnp.max(
                jnp.abs(
                    dparams[key] * masks[key]) * lr * gain) for key in dparams.keys()])
        if iter_ > discard_iter:
            aparams = {key: ((sa_count - 1.) / sa_count) * aparams[key] + (1. / sa_count) * params[key]
                       for key, value in aparams.items()}
        closs = cal_closs3d(params, y, eta3d, w, crf)
        trace["accept_rate"].append(accept_rate)
        trace["delta_params"].append(delta_params)
        trace["closs"].append(closs)
        if verbose:
            update_timers(
                timer1, timer2, iter_, discard_iter, trace)
        if iter_ <= discard_iter:
            if adaptive_jump:
                jump_scale = adjust_jump_scale(
                    jump_scale, accept_rate, target_rate)
        else:
            if max(trace["delta_params"][-window_size:]) < tol:
                converged = True
                break
    end = time.time()
    trace["jump_scale"] = jump_scale
    trace["iter"] = iter_
    trace["converged"] = converged
    trace["time"] = end - start
    eta = jnp.mean(eta3d, axis=0)
    return params, aparams, eta3d, eta, trace
