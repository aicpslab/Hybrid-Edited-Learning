function [A_uncertain_tkm, tkm_sparsity] = build_tkm_lorenz96(N, K, monomials)
% BUILD_TKM_LORENZ96 Build Temporal Knowledge Matrix (TKM) mask.
%
%   When the PhNN input is [x(k), x(k-1), ..., x(k-K+1)] (K*N dimensions),
%   the Taylor expansion generates monomials mixing variables from DIFFERENT
%   time steps. For Lorenz-96, since the system is FIRST-ORDER MARKOV:
%       x(k+1) depends ONLY on x(k), NOT on x(k-1), x(k-2), ...
%
%   Therefore, ALL cross-temporal monomials (those involving variables from
%   more than one time step) have ZERO contribution and should be pruned.
%
%   TKM is the matrix encoding this temporal knowledge:
%       A_uncertain_tkm(i,h) = 0  if monomial h mixes multiple time steps
%       A_uncertain_tkm(i,h) = 1  if monomial h is within a single time step
%
%   Theoretical guarantee: For any first-order ODE system discretized
%   via Euler/RK, the one-step mapping is Markov by construction.
%   Cross-temporal weights in the TRUE function are IDENTICALLY ZERO.
%
%   Inputs:
%       N         - State dimension per time step
%       K         - Number of temporal steps in input (K >= 2)
%       monomials - Cell array from GENERATE_MONOMIALS(N*K, order)
%
%   Outputs:
%       A_uncertain_tkm - (N, n_monomials) TKM mask (0 = prune cross-temporal)
%       tkm_sparsity    - Fraction additionally pruned by TKM
%
%   Example:
%       N=40, K=2 -> 80D input
%       Variables 1:40 belong to x(k), variables 41:80 belong to x(k-1)
%       Monomial [1, 41] involves BOTH -> pruned by TKM
%       Monomial [1, 2]  involves only x(k) -> kept
%       Monomial [41,42] involves only x(k-1) -> kept
%
%   See also: BUILD_PIM_LORENZ96, GENERATE_MONOMIALS

    n_monomials = length(monomials);

    % Initialize as all learnable (will prune cross-temporal)
    A_uncertain_tkm = ones(N, n_monomials, 'single');

    for h = 1:n_monomials
        indices_h = monomials{h};

        % Determine which time step(s) this monomial involves
        % Variables 1:N belong to step k
        % Variables N+1:2N belong to step k-1, etc.
        time_steps = floor((indices_h - 1) / N);  % 0-based time step index

        if length(unique(time_steps)) > 1
            % This monomial mixes variables from multiple time steps
            % -> PRUNE for all output dimensions
            A_uncertain_tkm(:, h) = 0;
        end
    end

    tkm_sparsity = 1.0 - mean(A_uncertain_tkm(:));

    n_pruned_tkm = sum(A_uncertain_tkm(:) == 0);
    n_kept_tkm = sum(A_uncertain_tkm(:) == 1);

    fprintf('  TKM Construction for temporal input (K=%d, total dim=%d):\n', K, N*K);
    fprintf('    Total connections:     %d\n', N * n_monomials);
    fprintf('    Kept (within-time):    %d\n', n_kept_tkm);
    fprintf('    Pruned (cross-time):   %d\n', n_pruned_tkm);
    fprintf('    TKM sparsity:          %.1f%%\n', tkm_sparsity * 100);
end
