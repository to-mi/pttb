Probabilistic formulation of the Take The Best heuristic
========================================================

This repository contains Matlab code implementing the probabilistic model of Take The Best heuristic and experiments described in

 * Peltola, Jokinen, Kaski. Probabilistic formulation of the Take The Best heuristic, to appear in the proceedings of CogSci2018.

## Running the experiments

 * `run_tests_on_real_data.m`: accuracy comparisons on benchmark datasets (this can take a lot of time with 1000 repetitions; make `n_reps` smaller in `test_on_real_data.m` for faster results).
 * `run_linear_function_learning_experiment.m`: function learning task given biased (TTB generated) pairwise feedback.

The functions `ttbfit` (exact inference using exhaustive computation) and `ttbmcmc` (MCMC inference) can be used to learn the probabilistic TTB model from training data.

## Requirements

The implementation of the probabilistic TTB does not have any requirements beyond Matlab. 

Running the experiments requires:

 * [`R`](https://www.r-project.org/) and packages [`TTBABC`](https://github.com/ericschulz/TTBABC) and [`heuristica`](https://cran.r-project.org/web/packages/heuristica/).
 * [`matlab2tikz`](http://www.mathworks.com/matlabcentral/fileexchange/22022-matlab2tikz), [`save2pdf`](https://www.mathworks.com/matlabcentral/fileexchange/16179-save2pdf), and [`tight_subplot`](https://www.mathworks.com/matlabcentral/fileexchange/27991-tight-subplot-nh--nw--gap--marg-h--marg-w-) for generating and saving result figures and [`latexTable`](https://www.mathworks.com/matlabcentral/fileexchange/44274-latextable) for tables.

## Contact

Tomi Peltola, tomi.peltola@aalto.fi
