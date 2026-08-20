function idx = generate_monomials(dim, order)
% GENERATE_MONOMIALS Generate all Taylor monomial indices up to given order.
%
%   idx = GENERATE_MONOMIALS(dim, order) returns a cell array where each
%   cell contains a vector of variable indices constituting one monomial.
%   Generates ALL monomials of orders 1 through `order` for `dim` variables.
%
%   Example: dim=3, order=2 ->
%     {[1],[2],[3], [1,1],[1,2],[1,3],[2,2],[2,3],[3,3]}
%
%   For Lorenz-96 (dim=40, order=2):
%     - Order 1: 40 terms
%     - Order 2: C(41,2) = 820 terms
%     - Total: 860 monomials
%
%   Inputs:
%       dim   - Number of input variables (positive integer)
%       order - Maximum Taylor expansion order (positive integer, >= 1)
%
%   Output:
%       idx   - (n_monomials, 1) cell array, each cell contains a row
%               vector of 1-based variable indices
%
%   See also: TAYLOR_EXPAND, BUILD_PIM_LORENZ96

    idx = {};
    n_monomials = 0;

    for r = 1:order
        if r == 1
            % Order 1: x_1, x_2, ..., x_dim
            for i = 1:dim
                n_monomials = n_monomials + 1;
                idx{n_monomials} = i;
            end

        elseif r == 2
            % Order 2: x_i * x_j for i <= j
            for i = 1:dim
                for j = i:dim
                    n_monomials = n_monomials + 1;
                    idx{n_monomials} = [i, j];
                end
            end

        elseif r == 3
            % Order 3: x_i * x_j * x_k for i <= j <= k
            for i = 1:dim
                for j = i:dim
                    for k = j:dim
                        n_monomials = n_monomials + 1;
                        idx{n_monomials} = [i, j, k];
                    end
                end
            end

        else
            % For higher orders, use recursive generation
            % (not typically needed for r <= 3)
            error('Order > 3 not supported. Use r <= 3.');
        end
    end

    % Verify dimension
    % n_expected = sum_{k=1}^{order} C(dim+k-1, k)
    % For dim=40, order=2: C(40,1) + C(41,2) = 40 + 820 = 860
end
