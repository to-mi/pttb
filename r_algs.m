function [yhat, fit] = r_algs(props_x_tr, props_y_tr, props_x_te, props_y_te, methods)

if nargin < 5
    methods = {'ttb', 'ttbGreedy', 'logReg', 'unitWeight', 'ttb_from_ttbabc', 'ttbabc'};
end

tmpfn = tempname;

% make input data
tr_data = [props_x_tr, props_y_tr];
tr_data_cues_pw = props_to_discrimination(props_x_tr);
tr_data_criterion_pw = props_to_discrimination(props_y_tr);
te_data = [props_x_te, props_y_te];
te_data_cues_pw = props_to_discrimination(props_x_te);
te_data_criterion_pw = props_to_discrimination(props_y_te);
criterion_col = size(tr_data, 2);
cue_cols = 1:(criterion_col - 1);

save(sprintf('%s_input.mat', tmpfn), ...
    'tr_data', 'tr_data_cues_pw', 'tr_data_criterion_pw', ...
    'te_data', 'te_data_cues_pw', 'te_data_criterion_pw',...
    'criterion_col', 'cue_cols', 'methods');

% run algs
system(sprintf('Rscript r_algs.R %s', tmpfn));

% load predictions
load(sprintf('%s_output.mat', tmpfn));

% clean up
delete(sprintf('%s_input.mat', tmpfn));
delete(sprintf('%s_output.mat', tmpfn));

end
