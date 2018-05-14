function [model, prop_correct] = ttbfit(y, x, discounting_method, opts)

if nargin < 3
    % one of 'independent', 'N', 'info'
    discounting_method = 'info';
end

if nargin < 4
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

log_unnorm_tree_probs = zeros(n_directions, n_orders);
prop_correct = zeros(n_directions, n_orders);
N_u = zeros(n_directions, n_orders);
N_i = zeros(n_directions, n_orders);
N_c = zeros(n_directions, n_orders);

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

        prop_correct(i_d, i_o) = N_correct / N_pairs;
        
        % hack:
        N_undecided = N_undecided * discount_ratio;
        N_correct = N_correct * discount_ratio;
        N_incorrect = N_incorrect * discount_ratio;
        
        N_u(i_d, i_o) = N_undecided;
        N_c(i_d, i_o) = N_correct;
        N_i(i_d, i_o) = N_incorrect;
        
        % if fixed epsilon:
        %log_tree_probs(i_d, i_o) = N_undecided * log_half + N_correct * log_1_m_epsilon + N_incorrect * log_epsilon;
        
        % if epsilon runs from 0 to 1 and has beta distribution:
        %log_tree_probs(i_d, i_o) = N_undecided * log_half + gammaln(N_incorrect + alpha) + gammaln(N_correct + beta) - gammaln(alpha_beta + N_incorrect + N_correct);
        
        % if epsilon runs from 0 to 1/2 and has "half-beta distribution":
        log_unnorm_tree_probs(i_d, i_o) = N_undecided * log_half + log(betainc(0.5, N_incorrect + alpha, N_correct + beta)) + betaln(N_incorrect + alpha, N_correct + beta);
        %log_unnorm_tree_probs(i_d, i_o) = N_undecided * log_half + log_incomplete_beta(0.5, N_incorrect + alpha, N_correct + beta);
        % note: betainc in Matlab is the regularized incomplete beta (hence we add betaln)
        
        % note: neither of the latter ones include the normalization constant
    end
end
if opts.verbosity > 0
    fprintf('Completed, time %d sec.\n', round(toc(st)));
end

% normalize tree probs
max_log_unnorm_tree_probs = max(log_unnorm_tree_probs(:));
log_unnorm_tree_probs = log_unnorm_tree_probs - max_log_unnorm_tree_probs;
lse = log(sum(exp(log_unnorm_tree_probs(:))));
log_z = max_log_unnorm_tree_probs + lse;
log_unnorm_tree_probs = log_unnorm_tree_probs - lse;

%tree_probs = exp(log_unnorm_tree_probs - max(log_unnorm_tree_probs(:)));
%tree_probs = tree_probs ./ sum(tree_probs(:));
tree_probs = exp(log_unnorm_tree_probs);

model.orders = orders;
model.directions = directions;
model.log_tree_probs = log_unnorm_tree_probs; % note: these are now normalized
model.tree_probs = tree_probs;
model.N_undecided = N_u;
model.N_correct = N_c;
model.N_incorrect = N_i;
model.log_z = log_z; % normalization constant (without the normalization constant of the half-beta prior)

end