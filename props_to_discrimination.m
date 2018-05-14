function x = props_to_discrimination(x_prop, thresholds)
% Takes in x_prop of size N x n_cues and transforms it to x of size
% N_pairs x n_cues, such that each pair of rows in x_prop is compared in
% each column and values are set to -1,0,1 depending on whether first of
% the pair has smaller, equal (up to threshold), larger original value.

if nargin < 2
    thresholds = [];
end

[N, n_cues] = size(x_prop);
N_pairs = nchoosek(N, 2);
x = zeros(N_pairs, n_cues);

for i = 1:N_pairs
    k = i - 1;
    ii = N - 2 - floor(sqrt(-8 * k + 4 * N * (N - 1) - 7) / 2.0 - 0.5);
    jj = k + ii + 1 - N * (N - 1) / 2 + (N - ii) * ((N - ii) - 1) / 2;
    ii = ii + 1;
    jj = jj + 1;
    
    x(i, :) = x_prop(ii, :) - x_prop(jj, :);
end

if ~isempty(thresholds)
    %x(abs(x) <= thresholds) = 0;
    x(bsxfun(@le, abs(x), thresholds(:)')) = 0;
end

x = sign(x);

end