%% load data
% these files have the target variable as last feature
%dataset_name = 'city';

nsamples = 1100;
nwarmup = 100;

switch dataset_name
    case 'homeless'
        dat = readtable('data/homeless.csv'); dat = dat(:, 2:end); prior_thresholds = {[0,10,20],[0,1,2,3,4,5],[0,5,10,20],[0,5,10,20],[0,2,5,10],[0,100,500,1000]};       
        n_reps = 1000;
        tr_p = (10:10:90) / 100;
    case 'profsalary'
        dat = readtable('data/profsalary.csv'); dat = dat(:, 3:end); prior_thresholds = {[0], [0 2 4 6], [0 2 4 6], [0], [0]};
        n_reps = 1000;
        tr_p = (10:10:90) / 100;
    case 'city'
        dat = readtable('data/city.csv'); dat = dat(:, 2:end); prior_thresholds = [];
        n_reps = 1000;
        tr_p = (10:10:90) / 100;
    case 'mileage'
        % number  of  cylinders
        % engine  displacement
        % horsepower
        % vehicle  weight
        % time  to  accelerate from  0  to  60  mph
        % model  year
        % origin  (American,  European,  Japanese)
        load carbig
        origin = zeros(406, 1);
        origin(all(bsxfun(@eq, Origin, 'USA    '), 2)) = 1;
        origin(all(bsxfun(@eq, Origin, 'Japan  '), 2)) = 2;
        origin(all(bsxfun(@eq, Origin, 'France '), 2)) = 3;
        origin(all(bsxfun(@eq, Origin, 'Germany'), 2)) = 3;
        origin(all(bsxfun(@eq, Origin, 'Sweden '), 2)) = 3;
        origin(all(bsxfun(@eq, Origin, 'Italy  '), 2)) = 3;
        origin(all(bsxfun(@eq, Origin, 'England'), 2)) = 3;
        dat = [Cylinders, Displacement, Horsepower, Weight, Acceleration, Model_Year, origin, MPG];
        dat = array2table(dat(~any(isnan(dat), 2), :), 'VariableNames',{'Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Model_Year', 'Origin', 'MPG'});
        prior_thresholds = {[0 2 4], [0 10 30 50 100 200], [0 10 30 50 100], [0 100 300 500 1000], [0 1 3 5], [0 1 3 5 10], [0 1]};
        n_reps = 1000;
        %tr_p = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5];
        tr_p = [0.01, 0.05, 0.1, 0.3, 0.5];
    otherwise
        error('unknown dataset')
end

%%

props = table2array(dat);
n = size(props, 1);
props_x = props(:, 1:(end - 1));
props_y = props(:, end);

if length(props_y) ~= length(unique(props_y))
    warning('data has ties in the criterion variable, adding some noise')
    props_y = props_y + 1e-10 * randn(length(props_y), 1);
end

%%
n_tr_sizes = length(tr_p);

%methods = {'independent', 'N', 'info'};
if isempty(prior_thresholds)
    methods = {'info-nopr'};
else
    methods = {'info-nopr', 'info-pr'};
end
%methods_r = {'ttb', 'ttbGreedy', 'logReg', 'unitWeight', 'ttb_from_ttbabc', 'ttbabc'};
methods_r = {'ttb', 'ttbGreedy', 'logReg', 'unitWeight', 'ttbabc'};
acc = zeros(n_reps, n_tr_sizes, length(methods));
acc_r = zeros(n_reps, n_tr_sizes, length(methods_r));
opts.verbosity = 0;

st = tic;
for tr_size_i = 1:n_tr_sizes
    n_tr = round(n * tr_p(tr_size_i));
    n_te = n - n_tr;
    for rep = 1:n_reps
        %% preprocess
        inds = randperm(n);
        inds_tr = inds(1:n_tr);
        inds_te = inds((n_tr + 1):end);
        
        props_x_tr = props_x(inds_tr, :);
        props_x_te = props_x(inds_te, :);
        props_y_tr = props_y(inds_tr, :);
        props_y_te = props_y(inds_te, :);
              
        for method_i = 1:length(methods)
            method_s = strsplit(methods{method_i}, '-');
            method_d = method_s{1};
            method_pr = method_s{2};
            switch method_pr
                case 'nopr'
                    prior_ths = [];
                case 'pr'
                    prior_ths = prior_thresholds;
                otherwise
                    error('unknown prior threshold method');
            end
            
            %% fit
            %[model, prop_correct] = ttbfit(y_tr, x_tr, opts);
            samples = ttbmcmc(props_y_tr, props_x_tr, prior_ths, nsamples, method_d);
            
            
            %% predict
            %yhat = ttbpred(model, x_te, opts);
            samples2 = ttbmcmcremovewarmup(samples, nwarmup);
            yhat = ttbmcmcpred(samples2, props_x_te);

            %% evaluate
            acc(rep, tr_size_i, method_i) = mean((2*(yhat > 0.5)-1) == props_to_discrimination(props_y_te));
        end
        
        %% R methods
        yhat = r_algs(props_x_tr, props_y_tr, props_x_te, props_y_te, methods_r);
        acc_r(rep, tr_size_i, :) = mean(bsxfun(@eq, yhat, props_to_discrimination(props_y_te)), 1);

        fprintf('Inner loop: Run %d of %d completed, time %d sec.\n', rep, n_reps, round(toc(st)));
    end
    fprintf('Outer loop: Run %d of %d completed, time %d sec.\n', tr_size_i, n_tr_sizes, round(toc(st)));
end

save(sprintf('results/%s.mat', dataset_name), 'acc', 'acc_r', 'methods_r', 'tr_p', 'methods', 'n_reps', 'nsamples', 'nwarmup');

%% plot
acc_combined = zeros(n_reps, n_tr_sizes, length(methods) + length(methods_r));
acc_combined(:, :, 1:length(methods)) = acc;
acc_combined(:, :, (length(methods) + 1):end) = acc_r;
methods_combined = [methods methods_r];

figure(1); clf;
offset = 0.005;
cols = lines(length(methods_combined));
m_acc = mean(acc_combined, 1);
hs = [];
for method_i = 1:length(methods_combined)
    hs(method_i) = plot(tr_p  + (method_i - 1)*offset, m_acc(1, :, method_i), '.-', 'color', cols(method_i, :)); hold on
    for tr_size_i = 1:n_tr_sizes
        plot([1 1] * tr_p(tr_size_i) + (method_i - 1)*offset, m_acc(1, tr_size_i, method_i) + 1.96 * [-1 1] * std(acc_combined(:, tr_size_i, method_i)) / sqrt(n_reps), '-', 'color', cols(method_i, :));
    end
end
legend(hs, methods_combined);
xlim([0 1])
ylim([0.5 1]);
ylabel('Accuracy');
xlabel('Training set size (fraction of full data)');

save2pdf(sprintf('results/%s_acc.pdf', dataset_name));

%% fit for full data
[samples, lp0] = ttbmcmc(props_y, props_x, prior_thresholds, nsamples);
samples2 = ttbmcmcremovewarmup(samples, nwarmup);
save(sprintf('results/%s_fulldata_mcmc_samples.mat', dataset_name), 'samples', 'lp0');

% probs of being at certain position in the tree
n_cues = size(props_x, 2);
probs = zeros(n_cues, n_cues);
for i = 1:n_cues
    probs(i, :) = mean(log2(abs(samples2.tree)) == (n_cues-i), 2);
end

tb = array2table(probs);
tb.Properties.VariableNames = dat.Properties.VariableNames(1:end-1);
rn = cell(1, n_cues); for i = 1:n_cues, rn{i} = sprintf('Pick %d', i); end;
tb.Properties.RowNames = rn;
tb

figure(2); clf;
imagesc(probs', [0 1]); colorbar
set(gca, 'YTick', 1:size(probs, 1));
set(gca, 'YTickLabel', strrep(tb.Properties.VariableNames, '_', ' '))
set(gca, 'XTick', 1:size(probs, 1));
set(gca, 'XTickLabel', 1:size(probs, 1));
xlabel('Pick')
save2pdf(sprintf('results/%s_pick_probs.pdf', dataset_name));

save(sprintf('results/%s_pick_probs.mat', dataset_name), 'tb');

figure(3); clf; plot(samples.log_prob)

[~, fit] = r_algs(props_x, props_y, props_x, props_y, {'ttb'});
save(sprintf('results/%s_fulldata_ttb_fit.mat', dataset_name), 'fit');
[~, fit] = r_algs(props_x, props_y, props_x, props_y, {'ttbGreedy'});
save(sprintf('results/%s_fulldata_ttbGreedy_fit.mat', dataset_name), 'fit');
