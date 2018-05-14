function ratio = discount_info(N_pairs)

% number of bits of information in ranking (N!)
N = (1 + sqrt(1 + 8 * N_pairs)) / 2;
n = sum(log2(1:N));
ratio = n / N_pairs;

end