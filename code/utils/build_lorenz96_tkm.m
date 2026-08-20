function [A_uncertain_tkm, tkm_sparsity] = build_lorenz96_tkm(dim, monomial_indices, K)
%% BUILD_LORENZ96_TKM  Temporal Knowledge Matrix for Lorenz-96
%   [A_uncertain_tkm, tkm_sparsity] = build_lorenz96_tkm(dim, monomial_indices, K)
%   Prunes monomials that mix variables from different time steps.
%   Assumes monomial_indices are 1-indexed.
    if nargin < 3, K = 2; end
    n_monomials = length(monomial_indices);
    A_uncertain_tkm = ones(dim, n_monomials, 'single');

    for h = 1:n_monomials
        midx = monomial_indices{h};
        % Convert to 0-based time step: 0..K-1
        time_steps = unique(floor((midx - 1) / dim));
        if length(time_steps) > 1
            A_uncertain_tkm(:, h) = 0;  % Cross-temporal -> prune
        end
    end

    tkm_sparsity = 1.0 - mean(A_uncertain_tkm(:));
end
