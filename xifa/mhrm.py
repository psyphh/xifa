import time
from functools import partial

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from tqdm.auto import tqdm

from .utils import cal_multinomial_logpmf



@partial(jit, static_argnums=(4,))
def cal_closs2d_i(
        params, y, eta2d, freq, crf):
    closs_i = cal_floss2d_i(params, y, eta2d, freq, crf) + cal_bloss2d_i(params, eta2d, freq)
    return closs_i


@partial(jit, static_argnums=(4,))
def cal_closs2d(
        params, y, eta2d, freq, crf):
    closs = cal_closs2d_i(params, y, eta2d, freq, crf).sum() / freq.sum()
    return closs


@partial(jit, static_argnums=(4,))
def cal_dcloss2d(
        params, y, eta2d, freq, crf):
    dcloss = grad(cal_closs2d, argnums=0)(params, y, eta2d, freq, crf)
    return dcloss


@partial(jit, static_argnums=(4,))
def cal_dcloss2d_i(
        params, y, eta2d, freq, crf):
    dcloss_i = jax.jacfwd(cal_closs2d_i, argnums=0)(params, y, eta2d, freq, crf)
    return dcloss_i


@partial(jit, static_argnums=(4,))
def cal_d2closs2d(
        params, y, eta2d, freq, crf):
    return jax.jacfwd(cal_dcloss2d, argnums=0)(params, y, eta2d, freq, crf)


@partial(jit, static_argnums=(4,))
def cal_closs3d_i(
        params, y, eta3d, freq, crf):
    closs_i = vmap(cal_closs2d_i,
                   in_axes=(None, None, 0, None, None))(
        params, y, eta3d, freq, crf)
    return closs_i


@partial(jit, static_argnums=(4,))
def cal_closs3d(
        params, y, eta3d, freq, crf):
    closs = cal_closs3d_i(params, y, eta3d, freq, crf).mean(axis=0).sum() / freq.sum()
    return closs


@partial(jit, static_argnums=(4,))
def cal_dcloss3d(
        params, y, eta3d, freq, crf):
    dcloss = grad(cal_closs3d, argnums=0)(params, y, eta3d, freq, crf)
    return dcloss


@partial(jit, static_argnums=(4,))
def cal_dcloss3d_i(
        params, y, eta3d, freq, crf):
    dcloss_i = vmap(cal_dcloss2d_i,
                    in_axes=(None, None, 0, None, None))(
        params, y, eta3d, freq, crf)
    return dcloss_i


@partial(jit, static_argnums=(4,))
def cal_d2closs3d(
        params, y, eta3d, freq, crf):
    d2closs = jax.jacfwd(cal_dcloss3d, argnums=0)(params, y, eta3d, freq, crf)
    return d2closs


@partial(jit, static_argnums=(4,))
def cal_floss2d_i(
        params, y, eta2d, freq, crf):
    floss_i = -jnp.sum(
        cal_multinomial_logpmf(
            k=y,
            p=crf(eta2d, params)),
        axis=-1) * freq
    return floss_i


@jit
def cal_bloss2d_i(
        params, eta2d, freq):
    n_factors = eta2d.shape[-1]
    bloss_i = - jax.scipy.stats.multivariate_normal.logpdf(
        x=eta2d,
        mean=jnp.zeros(n_factors),
        cov=params["corr"]) * freq
    return bloss_i


@jit
def cal_bloss3d_i(
        params, eta3d, freq):
    bloss_i = vmap(cal_bloss2d_i,
                   in_axes=(None, 0, None))(
        params, eta3d, freq)
    return bloss_i


@jit
def cal_bloss3d(
        params, eta3d, freq):
    bloss = cal_bloss3d_i(params, eta3d, freq).mean(axis=0).sum() / freq.sum()
    return bloss


def adjust_jump_std(
        jump_std, accept_rate, target_rate, jump_change):
    if accept_rate > (target_rate + .01):
        jump_std = jump_std + jump_change
    elif accept_rate < (target_rate - .01):
        jump_std = jump_std - jump_change
    else:
        pass
    return jump_std


def set_timers(
        max_iters, stem_iters):
    timer1 = tqdm(
        total=stem_iters,
        disable=False,
        position=0)
    timer2 = tqdm(
        total=max_iters - stem_iters,
        disable=False,
        position=1)
    return timer1, timer2


def update_timers(
        timer1, timer2, n_iter, stem_iters, trace):
    if n_iter <= stem_iters:
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
def conduct_mcmc(
        key, n_warmups, jump_std,
        eta3d, y, freq, params, crf):
    accept_rate = 0
    n_factors = eta3d.shape[-1]

    def sample_eta3d(i, value):
        key, eta3d, accept_rate = value
        key, subkey = jax.random.split(key)
        eta3d_new = jax.random.multivariate_normal(
            key=subkey,
            mean=eta3d,
            cov=jnp.diag(jnp.repeat(jump_std**2, n_factors)),
            shape=eta3d.shape[:-1])
        ratio = jnp.exp(
            - cal_closs3d_i(params, y, eta3d_new, freq, crf) + cal_closs3d_i(params, y, eta3d, freq, crf))
        key, subkey = jax.random.split(key)
        accept = jax.random.bernoulli(
            key=subkey,
            p=jnp.minimum(ratio, 1))
        eta3d = jnp.where(
            jnp.repeat(
                accept[..., None],
                n_factors,
                axis=-1),
            x=eta3d_new,
            y=eta3d)
        accept_rate = accept.mean()
        return key, eta3d, accept_rate

    key, eta3d, accept_rate = jax.lax.fori_loop(
        0, n_warmups + 1, sample_eta3d, (key, eta3d, accept_rate))
    return eta3d, accept_rate


@jit
def cal_cov_eta3d(eta3d):
    cov_eta = jnp.mean(
        vmap(lambda eta:
             eta.T @ eta / eta.shape[0], in_axes=0)(eta3d),
        axis=0)
    return cov_eta


@jit
def ls_corr(
        lr, gain, params, dparams, masks, eta3d, freq):
    c_armijo = 0.001
    d_armijo = 1.0
    lr_armijo = 0
    iter_armijo = 0
    corr = params["corr"]
    bloss = cal_bloss3d(params, eta3d, freq)
    value = (d_armijo, params, lr_armijo, iter_armijo)

    def body_fun(value):
        d_armijo, params, lr_armijo, iter_armijo = value
        lr_armijo = (0.5 ** iter_armijo) * lr
        params["corr"] = corr - lr_armijo * gain * dparams["corr"] * masks["corr"]
        d_armijo = cal_bloss3d(params, eta3d, freq) - bloss + c_armijo * lr_armijo * jnp.sum(dparams["corr"] ** 2)
        iter_armijo = iter_armijo + 1
        return d_armijo, params, lr_armijo, iter_armijo

    def cond_fun(value):
        d_armijo, params, lr_armijo, iter_armijo = value
        cond = d_armijo > 0.
        return cond

    d_armijo, params, lr_armijo, iter_armijo = jax.lax.while_loop(cond_fun, body_fun, value)
    return params["corr"], lr_armijo


@jit
def project_corr(
        params, masks):
    scale = jnp.sqrt(params["corr"].diagonal())
    params["corr"] = params["corr"] / jnp.outer(scale, scale)
    params["corr"] = params["corr"] * masks["corr"] + jnp.diag(params["corr"].diagonal())
    return params


def update_params(
        lr, gain,
        stage, corr_update,
        params, dparams, masks,
        eta3d, freq):
    params["intercept"] = params["intercept"] - lr * gain * dparams["intercept"] * masks["intercept"]
    params["loading"] = params["loading"] - lr * gain * dparams["loading"] * masks["loading"]
    if jnp.sum(masks["corr"]) >= 1.0:
        if stage == 1:
            params["corr"] = cal_cov_eta3d(eta3d)
            params = project_corr(params, masks)
        else:
            if corr_update == "empirical":
                corr = params["corr"]
                params["corr"] = cal_cov_eta3d(eta3d)
                params = project_corr(params, masks)
                params["corr"] = ((1 - lr * gain) * corr) + (lr * gain * params["corr"])
            else:
                dparams["corr"] = dparams["corr"] + dparams["corr"].T - jnp.diag(dparams["corr"].diagonal())
                if corr_update == "gd":
                    params["corr"] = params["corr"] - lr * gain * dparams["corr"] * masks["corr"]
                elif corr_update == "gd_ls":
                    params["corr"], lr_armijo = ls_corr(
                        lr, gain, params, dparams, masks, eta3d, freq)
                else:
                    pass
    else:
        pass
    return params


def fit_mhrm(
        lr,
        max_iters,
        stem_iters,
        tol,
        window_size,
        n_chains,
        n_warmups,
        jump_std,
        jump_change,
        target_rate,
        gain_decay,
        corr_update,
        batch_size,
        cycling,
        verbose,
        key,
        params,
        masks,
        y,
        eta,
        freq,
        crf):
    start = time.time()
    eta3d = jnp.repeat(eta[None, ...], 1, axis=0)
    if not isinstance(batch_size, type(None)):
        n_cases = y.shape[0]
        n_batches = int(jnp.ceil(n_cases / batch_size))
        batch_slices = [
            slice(batch_size * i, min(batch_size * (i + 1), n_cases), 1) for i in range(n_batches)]
        whole_idx = jnp.arange(n_cases)
        if isinstance(cycling, type(None)):
            cycling = True
    aparams = {key: jnp.zeros(value.shape) for key, value in params.items()}
    trace = {"accept_rate": [], "closs": [], "change_param": []}
    stage = 1
    gain = 1.
    is_converged = False
    is_nan = False
    if verbose:
        timer1, timer2 = set_timers(max_iters, stem_iters)
    for n_iters in range(1, max_iters + 1):
        if n_iters == stem_iters + 1:
            stage = 2
            eta3d = jnp.repeat(eta3d, n_chains, axis=0)
        if stage == 2:
            sa_count = (n_iters - stem_iters)
            gain = 1. / (sa_count ** gain_decay)
        temp = params.copy()
        if isinstance(batch_size, type(None)):
            key, subkey = jax.random.split(key)
            eta3d, accept_rate = conduct_mcmc(
                subkey, n_warmups, jump_std,
                eta3d, y, freq, params, crf)
            dparams = cal_dcloss3d(
                params, y, eta3d, freq, crf)
            params = update_params(
                lr, gain,
                stage, corr_update,
                params, dparams, masks,
                eta3d, freq)
        else:
            sum_freq = freq.sum()
            accept_rate = jnp.zeros(())
            key, subkey = jax.random.split(key)
            whole_idx = jax.random.permutation(subkey, whole_idx)
            if cycling:
                for batch_slice in batch_slices:
                    batch_idx = whole_idx[batch_slice]
                    y_batch, eta3d_batch, freq_batch = y[batch_idx, ...], eta3d[:, batch_idx, :], freq[batch_idx]
                    key, subkey = jax.random.split(key)
                    eta3d_batch, accept_rate_batch = conduct_mcmc(
                        subkey, n_warmups, jump_std,
                        eta3d_batch, y_batch, freq_batch, params, crf)
                    dparams = cal_dcloss3d(
                        params, y_batch, eta3d_batch, freq_batch, crf)
                    params = update_params(
                        lr, gain,
                        stage, corr_update,
                        params, dparams, masks,
                        eta3d_batch, freq_batch)
                    accept_rate = accept_rate + (freq_batch.sum() / sum_freq) * accept_rate_batch
                    eta3d = jax.ops.index_update(
                        eta3d, jax.ops.index[:, batch_idx, :], eta3d_batch)
            else:
                batch_idx = whole_idx[batch_slices[0]]
                y_batch, eta3d_batch, freq_batch = y[batch_idx, ...], eta3d[:, batch_idx, :], freq[batch_idx]
                key, subkey = jax.random.split(key)
                eta3d_batch, accept_rate_batch = conduct_mcmc(
                    subkey, n_warmups, jump_std,
                    eta3d_batch, y_batch, freq_batch, params, crf)
                dparams = cal_dcloss3d(
                    params, y_batch, eta3d_batch, freq_batch, crf)
                params = update_params(
                    lr, gain,
                    stage, corr_update,
                    params, dparams, masks,
                    eta3d_batch, freq_batch)
                accept_rate = accept_rate_batch
                eta3d = jax.ops.index_update(
                    eta3d, jax.ops.index[:, batch_idx, :], eta3d_batch)
        eta3d = eta3d / jnp.sqrt(params["corr"].diagonal())
        closs = cal_closs3d(params, y, eta3d, freq, crf)
        dparams = {key: params[key] - temp[key] for key in params.keys()}
        change_param = max(
            [jnp.max(
                jnp.abs(
                    dparams[key] * masks[key])) for key in dparams.keys()])
        trace["accept_rate"].append(accept_rate)
        trace["change_param"].append(change_param)
        trace["closs"].append(closs)
        if verbose:
            update_timers(
                timer1, timer2, n_iters, stem_iters, trace)
        if jnp.isnan(closs):
            is_nan = True
            break
        if stage == 1:
            jump_std = adjust_jump_std(
                jump_std, accept_rate, target_rate, jump_change)
        else:
            aparams = {key: ((sa_count - 1.) / sa_count) * aparams[key] + (1. / sa_count) * params[key]
                       for key in aparams.keys()}
            if max(trace["change_param"][-window_size:]) < tol:
                is_converged = True
                break
    end = time.time()
    fit_time = end - start
    eta = jnp.mean(eta3d, axis=0)
    trace["jump_std"] = jump_std
    trace["n_iters"] = n_iters
    trace["is_converged"] = is_converged
    trace["is_nan"] = is_nan
    trace["fit_time"] = fit_time
    return params, aparams, eta, trace
