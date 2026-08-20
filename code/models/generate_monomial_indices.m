function indices = generate_monomial_indices(dim, order)
%% GENERATE_MONOMIAL_INDICES  Taylor expansion monomial index generator
%   indices = generate_monomial_indices(dim, order)
%   Returns a cell array of 1-indexed monomial index vectors.
%   Generates all combinations with replacement for orders 1..order.
    indices = {};
    for r = 1:order
        combos = generate_cwr(dim, r, 1);
        for c = 1:size(combos, 1)
            indices{end+1} = combos(c, :); %#ok<AGROW>
        end
    end
end

function result = generate_cwr(n, k, start_val)
% Recursive combinations-with-replacement generator (1-indexed output)
    if k == 1
        result = (start_val:start_val+n-1)';
        return;
    end
    result = zeros(0, k);
    for i = 0:n-1
        val = start_val + i;
        sub = generate_cwr(n - i, k - 1, val);
        nr = size(sub, 1);
        result = [result; val * ones(nr, 1), sub]; %#ok<AGROW>
    end
end
