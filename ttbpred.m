function yhat = ttbpred(model, x, opts)
% Gives the posterior mean of the probability of first object winning.
% Note: predicts each single pairwise comparison independently (that is, not
% multivariate prediction).

if nargin < 3
    opts.verbosity = 0;
end

orders = model.orders;
directions = model.directions;
tree_probs = model.tree_probs;

n_orders = size(orders, 2);
n_directions = size(directions, 2);

log_half = log(0.5);
alpha = 1;
beta = 1;

% assume x is N_te x n_cues
N_te = size(x, 1);
yhat = zeros(N_te, 1);

st = tic;
for i_o = 1:n_orders
    for i_d = 1:n_directions
        tree = directions(:, i_d) .* orders(:, i_o);
%         v = x * double(tree);
%
%         yhat = yhat + tree_probs(i_d, i_o) * ((v > 0) + 0.5 * (v == 0));

        N_u = model.N_undecided(i_d, i_o);
        N_c = model.N_correct(i_d, i_o);
        N_i = model.N_incorrect(i_d, i_o);

        tree_prediction = sign(x * double(tree)) + 2; % [1, 2, 3] for -1, undecided, 1

        pred_T1_y1 = N_u * log_half + log(betainc(0.5, N_i + alpha, N_c + beta + 1)) + betaln(N_i + alpha, N_c + beta + 1);
        pred_T1_y0 = N_u * log_half + log(betainc(0.5, N_i + alpha + 1, N_c + beta)) + betaln(N_i + alpha + 1, N_c + beta);
        pred_T0_y1 = pred_T1_y0;
        pred_T0_y0 = pred_T1_y1;

        probs = [1 / (1 + exp(pred_T0_y0 - pred_T0_y1));...
            0.5;...
            1 / (1 + exp(pred_T1_y0 - pred_T1_y1))];

        yhat = yhat + probs(tree_prediction) * tree_probs(i_d, i_o);
    end
end
if opts.verbosity > 0
    fprintf('Completed, time %d sec.\n', round(toc(st)));
end

end