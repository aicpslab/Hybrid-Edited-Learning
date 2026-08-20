function [A_value, A_uncertain, pim_sparsity] = build_pim_lorenz96(N, dt, monomials)
% BUILD_PIM_LORENZ96 Build Physics Information Matrix (PIM) for Lorenz-96.
%
%   Lorenz-96 dynamics (Euler discretization):
%       x_i(k+1) = x_i(k) + dt * [(x_{i+1}(k) - x_{i-2}(k)) * x_{i-1}(k) - x_i(k) + F]
%
%   Known structure:
%       - Each x_i only depends on {x_{i-2}, x_{i-1}, x_i, x_{i+1}} (cyclic)
%       - Self-term x_i has known coefficient (1 - dt)
%       - Cross-terms x_{i-1}*x_{i+1} and x_{i-1}*x_{i-2} exist but unknown
%       - All other monomials have ZERO contribution
%
%   PIM definition (from the paper):
%       A_physics(i,j) = R   -> Known non-zero relationship (fixed weight = R)
%       A_physics(i,j) = 0   -> Known zero (connection pruned)
%       A_physics(i,j) = *   -> Unknown relationship (learnable, A_uncertain=1)
%
%   This function returns the PIM decomposed into:
%       A_value(i,h)     = known physical value (0 if unknown or zero)
%       A_uncertain(i,h) = 1 if learnable, 0 if pruned
%
%   For Lorenz-96 N=40, r=2: 860 monomials, only ~14 relevant per output
%       -> PIM sparsity = 98.4%
%
%   Inputs:
%       N         - System dimension
%       dt        - Time step (for known coefficient: 1-dt)
%       monomials - Cell array from GENERATE_MONOMIALS
%
%   Outputs:
%       A_value      - (N, n_monomials) Known physical weight values
%       A_uncertain  - (N, n_monomials) Binary mask (1=learn, 0=prune)
%       pim_sparsity - Fraction of pruned connections (scalar)
%
%   See also: BUILD_TKM_LORENZ96, GENERATE_MONOMIALS

    n_monomials = length(monomials);

    A_value = zeros(N, n_monomials, 'single');
    A_uncertain = zeros(N, n_monomials, 'single');

    for i = 1:N  % Output dimension (next state of variable i)
        % Relevant input variables for output i (cyclic, 1-based indexing)
        im2 = mod(i-3, N) + 1;  % i-2
        im1 = mod(i-2, N) + 1;  % i-1
        ip1 = mod(i, N) + 1;    % i+1
        relevant_set = [im2, im1, i, ip1];

        for h = 1:n_monomials
            indices_h = monomials{h};

            % Check if ALL variables in this monomial belong to the
            % relevant neighbor set of output i
            all_in_relevant = all(ismember(indices_h, relevant_set));

            if all_in_relevant
                A_uncertain(i, h) = 1;  % Learnable connection
            end
            % else: stays 0 (pruned -- connection known to be zero)

            % ---- Set known physical coefficients ----
            % Self-term x_i: coefficient = 1 - dt (from Euler discretization)
            if length(indices_h) == 1 && indices_h(1) == i
                A_value(i, h) = single(1.0 - dt);
            end
        end
    end

    % Compute sparsity
    pim_sparsity = 1.0 - mean(A_uncertain(:));

    n_learnable = sum(A_uncertain(:));
    n_total = N * n_monomials;
    n_fixed = sum(abs(A_value(:)) > 1e-10);

    fprintf('  PIM Construction for Lorenz-96 (N=%d, r=2):\n', N);
    fprintf('    Total connections:     %d\n', n_total);
    fprintf('    Learnable (A_unc=1):   %d\n', n_learnable);
    fprintf('    Fixed (A_val ~= 0):    %d\n', n_fixed);
    fprintf('    Pruned (A_unc=0):      %d\n', n_total - n_learnable - n_fixed);
    fprintf('    PIM sparsity:          %.1f%%\n', pim_sparsity * 100);
end
