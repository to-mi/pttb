rng(747432);

%%
clear all;
dataset_name = 'homeless';
test_on_real_data;
tb.Properties.VariableNames = strrep(tb.Properties.VariableNames, '_', '');
input_.data = tb;
input_.tableCaption = sprintf('Cue order probabilities for %s dataset', dataset_name);
input_.tableLabel = sprintf('%s_pick_probs', dataset_name);
lat_ = latexTable(input_);
fid = fopen(sprintf('results/%s_pick_probs.tex', dataset_name), 'w');
for i = 1:length(lat_)
    fprintf(fid, strrep(lat_{i}, '\', '\\'));
    fprintf(fid, '\n');
end
fclose(fid);

clear all;
dataset_name = 'profsalary';
test_on_real_data;
tb.Properties.VariableNames = strrep(tb.Properties.VariableNames, '_', '');
input_.data = tb;
input_.tableCaption = sprintf('Cue order probabilities for %s dataset', dataset_name);
input_.tableLabel = sprintf('%s_pick_probs', dataset_name);
lat_ = latexTable(input_);
fid = fopen(sprintf('results/%s_pick_probs.tex', dataset_name), 'w');
for i = 1:length(lat_)
    fprintf(fid, strrep(lat_{i}, '\', '\\'));
    fprintf(fid, '\n');
end
fclose(fid);

clear all;
dataset_name = 'city';
test_on_real_data;
tb.Properties.VariableNames = strrep(tb.Properties.VariableNames, '_', '');
input_.data = tb;
input_.tableCaption = sprintf('Cue order probabilities for %s dataset', dataset_name);
input_.tableLabel = sprintf('%s_pick_probs', dataset_name);
lat_ = latexTable(input_);
fid = fopen(sprintf('results/%s_pick_probs.tex', dataset_name), 'w');
for i = 1:length(lat_)
    fprintf(fid, strrep(lat_{i}, '\', '\\'));
    fprintf(fid, '\n');
end
fclose(fid);

clear all;
dataset_name = 'mileage';
test_on_real_data;
tb.Properties.VariableNames = strrep(tb.Properties.VariableNames, '_', '');
input_.data = tb;
input_.tableCaption = sprintf('Cue order probabilities for %s dataset', dataset_name);
input_.tableLabel = sprintf('%s_pick_probs', dataset_name);
lat_ = latexTable(input_);
fid = fopen(sprintf('results/%s_pick_probs.tex', dataset_name), 'w');
for i = 1:length(lat_)
    fprintf(fid, strrep(lat_{i}, '\', '\\'));
    fprintf(fid, '\n');
end
fclose(fid);

%% do combined figure
clear all;
figure(1); clf;
dataset_names = {'homeless', 'profsalary', 'city', 'mileage'};
Ns = [50, 52, 83, 392];
Ncues = [6, 5, 9, 7];
dirs = {'results', 'results', 'results', 'results'};
methods_include = {'logReg', 'ttb', 'ttbabc', 'info-nopr', 'info-pr'};
method_names = {'Logistic regression', 'TTB', 'ABC-TTB', 'PTTB', 'PTTB-CDT'};
cols = [0 0 0; 0 0 1; 0.5 0 0.5; 1 0 0; 1 0.4 0.4];
ha = tight_subplot(2 ,2, 0.01, [.1 .01], [.1 .01]);

for i = 1:4
    load(sprintf('%s/%s.mat', dirs{i}, dataset_names{i}));
    n_tr_sizes = length(tr_p);
    
    acc_combined = zeros(n_reps, n_tr_sizes, length(methods) + length(methods_r));
    acc_combined(:, :, 1:length(methods)) = acc;
    acc_combined(:, :, (length(methods) + 1):end) = acc_r;
    methods_combined = [methods methods_r];
    
    [~, ia, ib] = intersect(methods_combined, methods_include);
    [~, inds] = sort(ib); ia = ia(inds); ib = ib(inds);
    assert(all(strcmp(methods_combined(ia), methods_include(ib))));
    acc_combined = acc_combined(:, :, ia);
    methods_combined = method_names(ib);
    
    %subplot(2, 2, i);
    axes(ha(i));
    offset = 0.005;
    
    m_acc = mean(acc_combined, 1);
    hs = [];
    for method_i = 1:length(methods_combined)
        hs(method_i) = plot(tr_p  + (method_i - 1)*offset, m_acc(1, :, method_i), '.-', 'color', cols(method_i, :)); hold on
        for tr_size_i = 1:n_tr_sizes
            plot([1 1] * tr_p(tr_size_i) + (method_i - 1)*offset, m_acc(1, tr_size_i, method_i) + 1.96 * [-1 1] * std(acc_combined(:, tr_size_i, method_i)) / sqrt(n_reps), '-', 'color', cols(method_i, :));
        end
    end
    
    switch i
        case 1
            set(gca, 'XTickLabel', []);
            ylabel('Accuracy');
        case 2
            set(gca, 'YTickLabel', []);
            set(gca, 'XTickLabel', []);
        case 3
            ylabel('Accuracy');
            xlabel('Training set size');
        case 4
            legend(hs, methods_combined, 'location', 'southeast');
            xlabel('Training set size');
            set(gca, 'YTickLabel', []);
    end
    
    xlim([0 1])
    ylim([0.5 1]);
    text(0.5, 0.97, sprintf('%s (N=%d, %d cues)', dataset_names{i}, Ns(i), Ncues(i)), 'HorizontalAlignment', 'Center');
    %title(dataset_names{i});
end
save2pdf('results/combined_acc.pdf');
matlab2tikz('results/combined_acc.tex', 'height', '\fheightacc', 'width', '\fwidthacc');

%% MCMC trace plot
clear all
figure(1); clf;
dataset_names = {'homeless', 'profsalary', 'city', 'mileage'};
cols = [0 0 0; 0 0 1; 0.5 0 0.5; 1 0 0; 1 0.4 0.4];
nshow = 50;
hs = zeros(4, 1);

for i = 1:4
    dataset_name = dataset_names{i};
    load(sprintf('results/%s_fulldata_mcmc_samples.mat', dataset_name))
    v = [lp0; samples.log_prob];
    v = v - min(v);
    v = v ./ max(v);

    hs(i) = plot(0:nshow, v(1:(nshow+1)), '-', 'color', cols(i, :));
    hold on;
end
ylim([-0.1 1.1]);
xlim([-1 (nshow+1)]);
xlabel('MCMC iteration')
ylabel('scaled log probability');
legend(hs, dataset_names, 'location', 'SouthEast');
save2pdf('results/mcmc_traces.pdf');
matlab2tikz('results/mcmc_traces.tex', 'height', '\fheightmcmc', 'width', '\fwidthmcmc');
