function ratio = discount_N(N_pairs)

% inverse of nchoosek(n, 2)
n = (1 + sqrt(1 + 8 * N_pairs)) / 2;
ratio = n / N_pairs;
    
end