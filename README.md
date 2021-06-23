# **XIFA**: Accelerated Item Factor Analysis
`xifa` is a python package for conducting *item factor analysis* (IFA). 

`xifa` is build on [`jax`](https://github.com/google/jax), a package for Autograd and XLA. Hence, `xifa` can run IFA on GPUs and TPUs to speed up the training process. That is why we call it *Accelerated IFA*.

In the current version (0.1.0), `xifa` supports ordinal data IFA with graded response model (GRM; [Semejima, 1969](https://link.springer.com/article/10.1007%2FBF03372160)) and generalized partial credit model (GPCM; [Muraki, 1992](https://doi.org/10.1177/014662169201600206)). 

For a tutorial, please see [IPIP 50 Items Example](http://colab.research.google.com/github/psyphh/xifa/blob/master/examples/ipip50.ipynb).


