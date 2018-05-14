function log_ml = ttb_log_marg_lik(y, x, model, discounting_method)

if nargin < 4
    % one of 'independent', 'N', 'info'
    discounting_method = 'info';
end

if nargin < 5
    opts.verbosity = 0;
end

n_cues = size(x, 2);
n_orders = factorial(n_cues);
n_directions = 2^n_cues;
N_pairs = length(y);

switch discounting_method
    case 'independent'
        discount_ratio = discount_independent(N_pairs);
    case 'N'
        discount_ratio = discount_N(N_pairs);
    case 'info'
        discount_ratio = discount_info(N_pairs);
    otherwise
        error('Unknown discounting method.');
end

% generates all permutations of the cues
orders = perms(int16(2.^(0:(n_cues - 1))))';
% generates all directions of the cues
directions = ((int16(dec2bin(0:2^n_cues-1))-48)*2-1)';

log_tree_probs = zeros(n_directions, n_orders);

log_half = log(0.5);
alpha = 1;
beta = 1;
% alpha_beta = alpha + beta;
% epsilon = alpha / (alpha + beta);
% log_1_m_epsilon = log1p(-epsilon);
% log_epsilon = log(epsilon);

% assume x is N_tr x n_cues

st = tic;
for i_o = 1:n_orders
    for i_d = 1:n_directions
        tree = directions(:, i_d) .* orders(:, i_o);
        v = x * double(tree);
        N_undecided = sum(v == 0);
        %N_correct = sum((v > 0 & y) | (v < 0 & not_y));
        N_correct = sum(v .* y > 0);
        N_incorrect = N_pairs - N_undecided - N_correct;

        % hack:
        N_undecided = N_undecided * discount_ratio;
        N_correct = N_correct * discount_ratio;
        N_incorrect = N_incorrect * discount_ratio;
        
        N_u_p = model.N_undecided(i_d, i_o);
        N_i_p = model.N_incorrect(i_d, i_o);
        N_c_p = model.N_correct(i_d, i_o);

%         log_tree_probs(i_d, i_o) = N_undecided * log_half ...
%                                    + log_incomplete_beta(0.5, N_incorrect + N_i_p + alpha, N_correct + N_c_p + beta) ...
%                                    - log_incomplete_beta(0.5, N_i_p + alpha, N_c_p + beta) ...
%                                    + model.log_tree_probs(i_d, i_o);
        log_tree_probs(i_d, i_o) = (N_undecided + N_u_p) * log_half ...
                                   + (log(betainc(0.5, N_incorrect + N_i_p + alpha, N_correct + N_c_p + beta)) + betaln(N_incorrect + N_i_p + alpha, N_correct + N_c_p + beta));

        % note: betainc in Matlab is the regularized incomplete beta (hence we add betaln)
    end
end
if opts.verbosity > 0
    fprintf('Completed, time %d sec.\n', round(toc(st)));
end

max_log_tree_probs = max(log_tree_probs(:));
log_ml = log(sum(exp(log_tree_probs(:) - max_log_tree_probs))) + max_log_tree_probs - model.log_z;

end