function expanded = taylor_expand(X, monomial_indices)
%% TAYLOR_EXPAND  Expand input matrix into Taylor monomials
%   expanded = taylor_expand(X, monomial_indices)
%   X: (n_samples, dim) matrix
%   monomial_indices: cell array of 1-indexed index vectors
%   Returns: (n_samples, n_monomials) matrix of monomial values
    [n_samples, ~] = size(X);
    n_monomials = length(monomial_indices);
    expanded = ones(n_samples, n_monomials);

    for h = 1:n_monomials
        idx = monomial_indices{h};
        for k = 1:length(idx)
            expanded(:, h) = expanded(:, h) .* X(:, idx(k));
        end
    end
end
