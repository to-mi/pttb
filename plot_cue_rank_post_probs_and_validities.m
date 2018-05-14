%% do combined figure
clear all;
figure(1); clf;
dataset_names = {'homeless', 'profsalary', 'city', 'mileage'};
Ns = [50, 52, 83, 392];
Ncues = [6, 5, 9, 7];
dirs = {'results', 'results', 'results', 'results'};

for i = 1:4
    load(sprintf('%s/%s_pick_probs.mat', dirs{i}, dataset_names{i}));
    load(sprintf('%s/%s_fulldata_ttb_fit.mat', dirs{i}, dataset_names{i}));
    
    [cue_validities, inds] = sort(fit.cue_validities, 'descend');
    tb = tb(:, inds);
    n_cues = length(cue_validities);
    assert(n_cues == size(tb, 1));
    pp = table2array(tb);
    
    subplot(2, 2, i);
    imagesc(pp, [0 1]);

    set(gca, 'ycolor', [0 0 0]);
    set(gca, 'XTick', 1:n_cues);
    set(gca, 'YTick', 1:n_cues);

    switch i
        case 1
            ylabel('Rank Validity');
        case 2
        case 3
            ylabel('Rank Validity');
            xlabel('Cue');
        case 4
            xlabel('Cue');
    end
    
    hold on;
    scaled_cue_validites = n_cues * (1 - 2 * (cue_validities-0.5)) + 0.5;
    plot(scaled_cue_validites, 'r.-')
    text(n_cues + 0.5, 0.5, '1.0', 'color', [1 0 0]);
    text(n_cues + 0.5, n_cues + 0.5, '0.5', 'color', [1 0 0]);

    title(dataset_names{i});
end
matlab2tikz('results/cue_rank_post_probs_and_validities.tex', 'imagesAsPng', false, 'height', '\fheightcrp', 'width', '\fwidthcrp');
