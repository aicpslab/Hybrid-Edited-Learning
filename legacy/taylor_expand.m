function M = taylor_expand(X, monomial_indices)
% TAYLOR_EXPAND Expand input data into Taylor monomials.
%
%   M = TAYLOR_EXPAND(X, monomial_indices) expands each sample in X
%   into its Taylor monomial terms specified by monomial_indices.
%
%   This is the core operation of the PhNN hidden layer: given input
%   vector x = [x_1, ..., x_d], generate all monomials m(x, r).
%
%   Example:
%       X = [1, 2, 3; 4, 5, 6];  % 2 samples, 3 variables
%       mono = {[1], [2], [3], [1,1], [1,2]};
%       M = taylor_expand(X, mono);
%       % M(1,:) = [1, 2, 3, 1, 2]  (x1, x2, x3, x1^2, x1*x2)
%
%   Inputs:
%       X               - (n_samples, dim) data matrix
%       monomial_indices - Cell array from GENERATE_MONOMIALS
%
%   Output:
%       M - (n_samples, n_monomials) expanded feature matrix
%
%   See also: GENERATE_MONOMIALS, PHNN_FORWARD

    [n_samples, dim] = size(X);  %#ok<ASGLU>
    n_monomials = length(monomial_indices);
    M = ones(n_samples, n_monomials);

    for h = 1:n_monomials
        indices = monomial_indices{h};
        for k = 1:length(indices)
            M(:, h) = M(:, h) .* X(:, indices(k));
        end
    end
end
