function loglik = flipnoise_sign_likelihood(y, x, w, discount_ratio)

alpha = 1;
beta = 1;

log_half = log(0.5);

N_pairs = length(y);

v = x * w;

N_undecided = sum(v == 0);
N_correct = sum(v .* y > 0);
N_incorrect = N_pairs - N_undecided - N_correct;

% hack:
N_undecided = N_undecided * discount_ratio;
N_correct = N_correct * discount_ratio;
N_incorrect = N_incorrect * discount_ratio;

loglik = N_undecided * log_half + log(betainc(0.5, N_incorrect + alpha, N_correct + beta)) + betaln(N_incorrect + alpha, N_correct + beta);

end