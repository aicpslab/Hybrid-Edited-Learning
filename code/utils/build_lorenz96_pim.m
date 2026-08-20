function [A_value, A_uncertain, pim_sparsity] = build_lorenz96_pim(dim, dt, monomial_indices)
%% BUILD_LORENZ96_PIM  Physics Information Matrix for Lorenz-96
%   [A_value, A_uncertain, pim_sparsity] = build_lorenz96_pim(dim, dt, monomial_indices)
%   dim: state dimension (e.g. 40)
%   dt: discrete time step
%   monomial_indices: cell array of 1-indexed monomial vectors
    n_monomials = length(monomial_indices);
    A_value = zeros(dim, n_monomials, 'single');
    A_uncertain = zeros(dim, n_monomials, 'single');

    for i = 1:dim
        % Relevant neighbors (cyclic, 1-based)
        i_m2 = mod(i-3, dim) + 1;  % i-2
        i_m1 = mod(i-2, dim) + 1;  % i-1
        i_p1 = mod(i, dim) + 1;    % i+1
        relevant = [i_m2, i_m1, i, i_p1];

        for h = 1:n_monomials
            midx = monomial_indices{h};
            if all(ismember(midx, relevant))
                A_uncertain(i, h) = 1;  % Learnable
            end
            % Known: first-order self-term x_i coefficient = 1 - dt
            if length(midx) == 1 && midx(1) == i
                A_value(i, h) = 1.0 - dt;
            end
        end
    end

    pim_sparsity = 1.0 - mean(A_uncertain(:));
end
