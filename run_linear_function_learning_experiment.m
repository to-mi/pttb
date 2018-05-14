rng(3248)

%% generate some data
% [x1, x2, x3, x4] = ndgrid(linspace(-1, 1, 5));
% X = [x1(:), x2(:), x3(:), x4(:)];
[x1, x2] = ndgrid(linspace(-1, 1, 15));
X = [x1(:), x2(:)];

prop_threshold = 0.2;
y_eval_noise = 0.0001;
m = 2;
n_tr = 2;
n_h = 10;
wres = 231;
w_true = randn(m, 1);
w_true = [1; 0.8];

eval_w = @(w_, x_) x_ * w_;
prior_w = @(w_) -0.5 * (w_' * w_);

% also some training points
tr = randperm(size(X, 1));
tr = tr(1:n_tr);
x_tr = X(tr, :);
y_tr = x_tr * w_true + randn(n_tr, 1);
log_prob_true_w = prior_w(w_true) + sum(log(normpdf(y_tr, eval_w(w_true, x_tr), 1)));

%% and pairwise comparison observations
y_all = eval_w(w_true, X) + y_eval_noise * randn(size(X, 1), 1);
x_ttb = props_to_discrimination(X, prop_threshold);
y_ttb = props_to_discrimination(y_all);
assert(sum(y_ttb == 0) == 0);

model = ttbfit(y_ttb, x_ttb);

h_obs_inds = randperm(size(X, 1));
h_obs_inds = h_obs_inds(1:n_h);

x_h = props_to_discrimination(X(h_obs_inds, :), prop_threshold);
x_hFN = props_to_pairwise_differences(X(h_obs_inds, :));

h = 2 * binornd(1, ttbpred(model, x_h)) - 1;
% generate also the feedback without heuristic
assert(all(x_hFN * w_true ~= 0))
h_no_heuristic = 2 * (x_hFN * w_true > 0) - 1;

assert(all(unique(h) == [-1; 1]))
assert(all(unique(h_no_heuristic) == [-1; 1]))

discount_ratio = discount_info(length(h));

logml_prior_true_w = model.log_z;
logml_true_w = ttb_log_marg_lik(h, x_h, model);
logml_noh_true_w = ttb_log_marg_lik(h_no_heuristic, x_h, model);
logmlFN_true_w = flipnoise_sign_likelihood(h, x_hFN, w_true, discount_ratio);
logmlFN_noh_true_w = flipnoise_sign_likelihood(h_no_heuristic, x_hFN, w_true, discount_ratio);

yhat = 2 * (ttbpred(model, x_h) > 0.5) - 1;
acc_true_w = mean(yhat == h);


%% compare marginal likelihoods of different w
wr = linspace(-2, 2, wres);
[w1, w2] = ndgrid(wr);
logml = zeros(wres, wres);
logml_noh = zeros(wres, wres);
logmlFN = zeros(wres, wres);
logmlFN_noh = zeros(wres, wres);
logml_prior = zeros(wres, wres);
log_prob = zeros(wres, wres);
acc = zeros(wres, wres);

x_ttb = props_to_discrimination(X, prop_threshold);
%xFN = props_to_pairwise_differences(X);

for i = 1:wres
    for j = 1:wres
        w = [w1(i, j); w2(i, j)];
        
        y_all = eval_w(w, X) + y_eval_noise * randn(size(X, 1), 1);
        y_ttb = props_to_discrimination(y_all);
        assert(sum(y_ttb == 0) == 0);
        
        model_ = ttbfit(y_ttb, x_ttb);
        
        logml_prior(i, j) = model_.log_z;
        logml(i, j) = ttb_log_marg_lik(h, x_h, model_);
        logml_noh(i, j) = ttb_log_marg_lik(h_no_heuristic, x_h, model_);
        logmlFN(i, j) = flipnoise_sign_likelihood(h, x_hFN, w, discount_ratio);
        logmlFN_noh(i, j) = flipnoise_sign_likelihood(h_no_heuristic, x_hFN, w, discount_ratio);
        log_prob(i, j) = prior_w(w) + sum(log(normpdf(y_tr, eval_w(w, x_tr), 1)));
  
        yhat = 2 * (ttbpred(model_, x_h) > 0.5) - 1;
        acc(i, j) = mean(yhat == h);
    end
end

%% result figure
figure(7); clf;
ha = tight_subplot(2 ,3, 0.01, [.1 .01], [.1 .01]);
%delete(ha(4));

%subplot(2, 3, 1);
axes(ha(1));
max_ = max(log_prob(:));
imagesc(wr, wr, exp(log_prob - max_));
set(gca, 'YDir', 'normal');
ylabel('w1');
xlabel('w2');
%title('No pairwise obs.');
text(0, -1.8, 'No pairwise obs.', 'HorizontalAlignment', 'Center', 'color', [1 1 1]);
%colorbar
hold on; plot(w_true(2), w_true(1), 'ro');
legend('off');

axes(ha(4)); colorbar; axis off

%subplot(2, 3, 2);
axes(ha(2));
max_ = max(logml(:) + log_prob(:));
imagesc(wr, wr, exp(logml + log_prob - max_));
set(gca, 'YDir', 'normal');
%title('TTB obs. & TTB model');
text(0, -1.8, 'TTB obs. & TTB model', 'HorizontalAlignment', 'Center', 'color', [1 1 1]);
hold on; plot(w_true(2), w_true(1), 'ro');
legend('off');
set(gca, 'YTickLabel', []);
set(gca, 'XTickLabel', []);

%subplot(2, 3, 3);
axes(ha(3));
max_ = max(logml_noh(:) + log_prob(:));
imagesc(wr, wr, exp(logml_noh + log_prob - max_));
set(gca, 'YDir', 'normal');
%title('Unbiased obs. & TTB model');
text(0, -1.8, 'Unbiased obs. & TTB model', 'HorizontalAlignment', 'Center', 'color', [1 1 1]);
hold on; plot(w_true(2), w_true(1), 'ro');
legend('off');
set(gca, 'YTickLabel', []);
set(gca, 'XTickLabel', []);

%subplot(2, 3, 5);
axes(ha(5));
max_ = max(logmlFN(:) + log_prob(:));
imagesc(wr, wr, exp(logmlFN + log_prob - max_));
set(gca, 'YDir', 'normal');
%title('TTB obs. & unbiased model');
text(0, -1.8, 'TTB obs. & unbiased model', 'HorizontalAlignment', 'Center', 'color', [1 1 1]);
hold on; plot(w_true(2), w_true(1), 'ro');
legend('off');
set(gca, 'YTickLabel', []);
set(gca, 'XTickLabel', []);

%subplot(2, 3, 6);
axes(ha(6));
max_ = max(logmlFN_noh(:) + log_prob(:));
imagesc(wr, wr, exp(logmlFN_noh + log_prob - max_));
set(gca, 'YDir', 'normal');
%title('Unbiased obs. & unbiased model');
text(0, -1.8, 'Unbiased obs. & unbiased model', 'HorizontalAlignment', 'Center', 'color', [1 1 1]);
hold on; plot(w_true(2), w_true(1), 'ro');
legend('off');
set(gca, 'YTickLabel', []);
set(gca, 'XTickLabel', []);

matlab2tikz('results/2d_lin_reg.tex', 'height', '\fheight', 'width', '\fwidth');

%% other and older figure
if 0
figure(1); clf;
imagesc(wr, wr, logml);
ylabel('w1');
xlabel('w2');
title(sprintf('w=[%.2f, %.2f], logml=%.2f', w_true(1), w_true(2), logml_true_w));
colorbar
hold on; plot(w_true(2), w_true(1), 'ro');

figure(2); clf;
imagesc(wr, wr, acc);
ylabel('w1');
xlabel('w2');
title(sprintf('w=[%.2f, %.2f], acc=%.2f', w_true(1), w_true(2), acc_true_w));
colorbar
hold on; plot(w_true(2), w_true(1), 'ro');

figure(3); clf;
imagesc(wr, wr, logml_prior);
ylabel('w1');
xlabel('w2');
title(sprintf('w=[%.2f, %.2f], logml\\_prior=%.2f', w_true(1), w_true(2), logml_prior_true_w));
colorbar
hold on; plot(w_true(2), w_true(1), 'ro');

figure(4); clf;
subplot(2, 2, 1);
imagesc(wr, wr, logml + log_prob);
ylabel('w1');
xlabel('w2');
title(sprintf('w=[%.2f, %.2f], logml+logprob=%.2f', w_true(1), w_true(2), logml_true_w + log_prob_true_w));
colorbar
hold on; plot(w_true(2), w_true(1), 'ro');

subplot(2, 2, 2);
imagesc(wr, wr, logml_noh + log_prob);
ylabel('w1');
xlabel('w2');
title(sprintf('w=[%.2f, %.2f], logml\\_noh+logprob=%.2f', w_true(1), w_true(2), logml_noh_true_w + log_prob_true_w));
colorbar
hold on; plot(w_true(2), w_true(1), 'ro');

subplot(2, 2, 3);
imagesc(wr, wr, logmlFN + log_prob);
ylabel('w1');
xlabel('w2');
title(sprintf('w=[%.2f, %.2f], logmlFN+logprob=%.2f', w_true(1), w_true(2), logmlFN_true_w + log_prob_true_w));
colorbar
hold on; plot(w_true(2), w_true(1), 'ro');

subplot(2, 2, 4);
imagesc(wr, wr, logmlFN_noh + log_prob);
ylabel('w1');
xlabel('w2');
title(sprintf('w=[%.2f, %.2f], logmlFN\\_noh+logprob=%.2f', w_true(1), w_true(2), logmlFN_noh_true_w + log_prob_true_w));
colorbar
hold on; plot(w_true(2), w_true(1), 'ro');

figure(5); clf;
subplot(2, 3, 1);
max_ = max(log_prob(:));
imagesc(wr, wr, exp(log_prob - max_));
ylabel('w1');
xlabel('w2');
title(sprintf('No pairwise obs., r.d. at w_{true}=%.2f', exp(log_prob_true_w - max_)));
colorbar
hold on; plot(w_true(2), w_true(1), 'ro');

subplot(2, 3, 2);
max_ = max(logml(:) + log_prob(:));
imagesc(wr, wr, exp(logml + log_prob - max_));
ylabel('w1');
xlabel('w2');
title(sprintf('TTB obs. & TTB model, r.d. at w_{true}=%.2f', exp(logml_true_w + log_prob_true_w - max_)));
colorbar
hold on; plot(w_true(2), w_true(1), 'ro');

subplot(2, 3, 3);
max_ = max(logml_noh(:) + log_prob(:));
imagesc(wr, wr, exp(logml_noh + log_prob - max_));
ylabel('w1');
xlabel('w2');
title(sprintf('Unbiased obs. & TTB model, r.d. at w_{true}=%.2f', exp(logml_noh_true_w + log_prob_true_w - max_)));
colorbar
hold on; plot(w_true(2), w_true(1), 'ro');

subplot(2, 3, 5);
max_ = max(logmlFN(:) + log_prob(:));
imagesc(wr, wr, exp(logmlFN + log_prob - max_));
ylabel('w1');
xlabel('w2');
title(sprintf('TTB obs. & unbiased model, r.d. at w_{true}=%.2f', exp(logmlFN_true_w + log_prob_true_w - max_)));
colorbar
hold on; plot(w_true(2), w_true(1), 'ro');

subplot(2, 3, 6);
max_ = max(logmlFN_noh(:) + log_prob(:));
imagesc(wr, wr, exp(logmlFN_noh + log_prob - max_));
ylabel('w1');
xlabel('w2');
title(sprintf('Unbiased obs. & unbiased model, r.d. at w_{true}=%.2f', exp(logmlFN_noh_true_w + log_prob_true_w - max_)));
colorbar
hold on; plot(w_true(2), w_true(1), 'ro');

figure(6); clf;
imagesc(wr, wr, log_prob);
ylabel('w1');
xlabel('w2');
title(sprintf('w=[%.2f, %.2f], logprob=%.2f', w_true(1), w_true(2), log_prob_true_w));
colorbar
hold on; plot(w_true(2), w_true(1), 'ro');
end