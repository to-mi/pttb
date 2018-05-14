function samples = ttbmcmcremovewarmup(samples, n_warmup)

samples.tree(:, 1:n_warmup) = [];
samples.thresholds(:, 1:n_warmup) = [];
samples.log_prob(1:n_warmup) = [];
samples.N_undecided(1:n_warmup) = [];
samples.N_correct(1:n_warmup) = [];
samples.N_incorrect(1:n_warmup) = [];

end