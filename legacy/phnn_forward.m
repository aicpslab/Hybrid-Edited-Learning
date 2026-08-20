function Y = phnn_forward(X, monomials, A_value, A_uncertain, W_learn, bias)
% PHNN_FORWARD Forward pass of a Physics-compatible Neural Network (PhNN).
%
%   Y = PHNN_FORWARD(X, monomials, A_value, A_uncertain, W_learn, bias)
%   computes the output of a PhNN with Taylor expansion hidden layer.
%
%   Architecture:
%       Y = (A_value + A_uncertain .* W_learn) * m(X, r) + bias
%
%   where:
%       m(X, r) = Taylor expansion of X (computed internally)
%       A_value    = fixed known physical weights (from PIM)
%       A_uncertain = binary mask (1 = learnable, 0 = pruned)
%       W_learn    = learnable weight matrix
%       bias       = bias vector
%
%   This is the NumPy-free equivalent of:
%       Weff = A_value + A_uncertain * W_learn (element-wise for masks)
%       Y = M @ Weff' + bias'
%
%   Inputs:
%       X           - (n_samples, dim_in) input data
%       monomials   - Cell array from GENERATE_MONOMIALS
%       A_value     - (dim_out, n_monomials) fixed physical weights
%       A_uncertain - (dim_out, n_monomials) learnability mask (0/1)
%       W_learn     - (dim_out, n_monomials) learnable weights
%       bias        - (dim_out, 1) bias vector
%
%   Output:
%       Y - (n_samples, dim_out) network output
%
%   See also: PHNN_TRAIN, TAYLOR_EXPAND, BUILD_PIM_LORENZ96

    % Step 1: Taylor expansion (hidden layer)
    M = taylor_expand(X, monomials);  % (n_samples, n_monomials)

    % Step 2: Effective weight matrix
    % W_eff = A_value + A_uncertain .* W_learn
    W_eff = A_value + A_uncertain .* W_learn;  % (dim_out, n_monomials)

    % Step 3: Linear output layer
    % Y = M * W_eff' + bias'
    Y = M * W_eff' + bias';  % (n_samples, dim_out)
end
