# **XIFA**: Accelerated Item Factor Analysis

## What is `xifa`?
`xifa` is a python package for conducting *item factor analysis* (IFA), a representative multivariate technique in psychometrics.

`xifa` is build on [`jax`](https://github.com/google/jax), a package for Autograd and XLA (accelerated linear algebra). Hence, `xifa` can run IFA on GPUs and TPUs to hugely speed up the training process. That is why we call it *Accelerated IFA*.

To calculate a marginal maximum likelihood (MML) estimate, `xifa` implements a vectorized version of Metropolis-Hastings Robbins-Monro (MH-RM) algorithm  ([Cai, 2010](https://doi.org/10.1007/s11336-009-9136-x.)). The vectorized algorithm is designed for parallel computing with GPUs of TPUs. The vectorized algorithm includes two stages: the first stage updates the parameter estimate by a stochastic expectation-maximization (StEM) algorithm and the second stage conducts stochastic approximation (SA) to iteratively refine the estimate.

For a tutorial, please see the [IPIP 50 Items Example](https://github.com/psyphh/xifa/blob/master/examples/ipip50.ipynb).

For a large-scale application, please see the [IPIP 300 Items Example (v2)](https://github.com/psyphh/xifa/blob/master/examples/ipip300v2.ipynb). 

## Features in `xifa`
`xifa` supports ordinal data IFA with 
+ graded response model (GRM; [Semejima, 1969](https://link.springer.com/article/10.1007%2FBF03372160))
+ generalized partial credit model (GPCM; [Muraki, 1992](https://doi.org/10.1177/014662169201600206)). 

The analysis can be either *exploratory* or *confirmatory*. In addition, the vectorized algorithm is able to handle the presence of 
+ missing responses
+ unequal category items

Features in statistical inference (e.g., goodness-of-fit statistics, parameter standard errors, etc.) are still under development.




