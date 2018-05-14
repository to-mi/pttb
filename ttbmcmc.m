function [samples, initial_log_prob] = ttbmcmc(y, x, prior_thresholds, nsamples, discounting_method, initial_tree, initial_thresholds, opts)

% note: y can be either N_pairs x 1 or N x 1
% x must be N x n_cues

[N, n_cues] = size(x);
N_pairs = nchoosek(N, 2);

if length(y) == N
    y = props_to_discrimination(y);
end
assert(N_pairs == length(y));

if isempty(prior_thresholds)
    prior_thresholds = cell(1, n_cues);
    for i = 1:n_cues
        prior_thresholds{i} = 0;
    end
end

if nargin < 5
    % one of 'independent', 'N', 'info'
    discounting_method = 'info';
end
if nargin < 6
    initial_ordinal_cue_validity= randperm(n_cues)';
    initial_direction = 2 * (rand(n_cues, 1) > 0.5) - 1;
    initial_tree = (2.^(initial_ordinal_cue_validity - 1)) .* initial_direction;
else
    initial_ordinal_cue_validity = tiedrank(initial_tree);
    initial_direction = sign(initial_tree);
end

if nargin < 7
    initial_thresholds = zeros(n_cues, 1);
    for i = 1:n_cues
        initial_thresholds(i) = prior_thresholds{i}(randi(length(prior_thresholds{i})));
    end
end

if nargin < 8
    opts.verbosity = 1;
end

alpha = 1;
beta = 1;
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

samples.tree = zeros(n_cues, nsamples);
samples.thresholds = zeros(n_cues, nsamples);
samples.N_undecided = zeros(nsamples, 1);
samples.N_correct = zeros(nsamples, 1);
samples.N_incorrect = zeros(nsamples, 1);
samples.log_prob = zeros(nsamples, 1);
samples.alpha = alpha;
samples.beta = beta;

tree = initial_tree;
ordinal_cue_validity = initial_ordinal_cue_validity;
direction = initial_direction;
thresholds = initial_thresholds;

if nargout > 1
    x_dat = props_to_discrimination(x, thresholds);
    initial_log_prob = log_tree_prob(tree, x_dat, y, alpha, beta, discount_ratio);
end

for iter = 1:nsamples
    % sample cue
    %cue = randi(n_cues);
    
    % iterate over all cues in random order before collecting sample
    for cue = randperm(n_cues)
        % generate proposal trees, such that the selected cue takes on all
        % possible positions in the tree while the order of other cues
        % remain the same
        [~, inds] = sort(ordinal_cue_validity);
        inds = inds(inds ~= cue);
        ordinal_cue_validities = repmat(1:n_cues, n_cues, 1);
        for i = 1:(n_cues - 1)
            ordinal_cue_validities(inds(i), 1:i) = i + 1;
            ordinal_cue_validities(inds(i), (i + 1):end) = i;
        end
        trees = bsxfun(@times, 2.^(ordinal_cue_validities - 1), direction);
        % add the same trees but with the sign of the selected cue flipped
        trees2 = trees;
        trees2(cue, :) = -trees2(cue, :);
        trees = [trees trees2];
        % in total, this will sample from 2 * n_cues trees
        
        % compute tree probabilities
        tree_probs = zeros(2 * n_cues, length(prior_thresholds{cue}));
        N_undecided = zeros(2 * n_cues, length(prior_thresholds{cue}));
        N_correct = zeros(2 * n_cues, length(prior_thresholds{cue}));
        N_incorrect = zeros(2 * n_cues, length(prior_thresholds{cue}));
        
        thresholds_prop = thresholds;
        
        for t_i = 1:length(prior_thresholds{cue})
            thresholds_prop(cue) = prior_thresholds{cue}(t_i);
            x_dat = props_to_discrimination(x, thresholds_prop);
            [tree_probs(:, t_i), N_undecided(:, t_i), N_correct(:, t_i), N_incorrect(:, t_i)] = log_tree_prob(trees, x_dat, y, alpha, beta, discount_ratio);
        end
        tree_probs_unscaled = tree_probs;
        max_tree_prob = max(tree_probs(:));
        tree_probs = exp(tree_probs - max_tree_prob);
        tree_probs = tree_probs / sum(tree_probs(:));
        
        % sample next tree
        [ind, t_ind] = ind2sub(size(tree_probs), find(mnrnd(1, tree_probs(:))));
        
        % for log prob trace
        tree_prob_unscaled = tree_probs_unscaled(ind, t_ind);
        
        % these are only needed when collecting the sample (useful for
        % prediction)
        N_u = N_undecided(ind, t_ind);
        N_c = N_correct(ind, t_ind);
        N_i = N_incorrect(ind, t_ind);
        
        tree = trees(:, ind);
        if ind > n_cues
            direction(cue) = -direction(cue);
            ind = ind - n_cues;
        end
        ordinal_cue_validity = ordinal_cue_validities(:, ind);
        thresholds(cue) = prior_thresholds{cue}(t_ind);
    end
    
    % collect sample
    samples.tree(:, iter) = tree;
    samples.log_prob(iter) = tree_prob_unscaled;
    samples.thresholds(:, iter) = thresholds;
    samples.N_undecided(iter) = N_u;
    samples.N_correct(iter) = N_c;
    samples.N_incorrect(iter) = N_i;
end

end


function [ltp, N_undecided, N_correct, N_incorrect] = log_tree_prob(trees, x, y, alpha, beta, discount_ratio)

N_pairs = size(x, 1);
log_half = log(0.5);

% x is N x n_cues, trees is n_cues x n_trees => v is N x n_trees
v = x * trees;
N_undecided = sum(v == 0);
N_correct = sum(v .* y > 0);
N_incorrect = N_pairs - N_undecided - N_correct;

% hack:
N_undecided = N_undecided * discount_ratio;
N_correct = N_correct * discount_ratio;
N_incorrect = N_incorrect * discount_ratio;

ltp = N_undecided * log_half + log(betainc(0.5, N_incorrect + alpha, N_correct + beta)) + betaln(N_incorrect + alpha, N_correct + beta);

end


