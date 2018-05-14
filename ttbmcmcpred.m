function yhat = ttbmcmcpred(samples,  x)

nsamples = size(samples.tree, 2);
alpha = samples.alpha;
beta = samples.beta;
log_half = log(0.5);

yhat = zeros(nchoosek(size(x, 1), 2), 1);

for iter = 1:nsamples
    tree = samples.tree(:, iter);
    thresholds = samples.thresholds(:, iter);
    N_u = samples.N_undecided(iter);
    N_c = samples.N_correct(iter);
    N_i = samples.N_incorrect(iter);
    
    x_dat = props_to_discrimination(x, thresholds);
    
    tree_prediction = sign(x_dat * tree) + 2; % [1, 2, 3] for -1, undecided, 1
    
    pred_T1_y1 = N_u * log_half + log(betainc(0.5, N_i + alpha, N_c + beta + 1)) + betaln(N_i + alpha, N_c + beta + 1);
    pred_T1_y0 = N_u * log_half + log(betainc(0.5, N_i + alpha + 1, N_c + beta)) + betaln(N_i + alpha + 1, N_c + beta);
    pred_T0_y1 = pred_T1_y0;
    pred_T0_y0 = pred_T1_y1;
    
    probs = [1 / (1 + exp(pred_T0_y0 - pred_T0_y1));...
             0.5;...
             1 / (1 + exp(pred_T1_y0 - pred_T1_y1))];
    
    yhat = yhat + probs(tree_prediction);
end

yhat = yhat / nsamples;

end
