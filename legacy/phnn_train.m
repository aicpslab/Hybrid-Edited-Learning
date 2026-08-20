function [W_learn, bias, train_losses, val_losses, best_val_loss] = ...
    phnn_train(X_train, Y_train, X_val, Y_val, monomials, A_value, A_uncertain, ...
               W_learn_init, bias_init, learning_rate, n_epochs, batch_size, verbose)
% PHNN_TRAIN Train a PhNN using mini-batch gradient descent with Adam optimizer.
%
%   The PhNN architecture is:
%       Y = (A_value + A_uncertain .* W_learn) * m(X, r) + bias
%
%   Only connections where A_uncertain(i,j) == 1 are updated.
%   Connections where A_uncertain(i,j) == 0 are frozen at zero (pruned).
%   Connections where A_value(i,j) ~= 0 are frozen at their known physical value.
%
%   Inputs:
%       X_train, Y_train - Training data
%       X_val, Y_val     - Validation data
%       monomials        - Cell array from GENERATE_MONOMIALS
%       A_value          - (dim_out, n_mono) Fixed physical values
%       A_uncertain      - (dim_out, n_mono) Mask: 1=learnable, 0=pruned
%       W_learn_init     - Initial learnable weight matrix
%       bias_init        - Initial bias vector
%       learning_rate    - Adam learning rate (default: 0.001)
%       n_epochs         - Number of training epochs
%       batch_size       - Mini-batch size
%       verbose          - If true, print progress every 20 epochs
%
%   Outputs:
%       W_learn       - Trained learnable weights
%       bias          - Trained bias
%       train_losses  - Training loss history
%       val_losses    - Validation loss history
%       best_val_loss - Best validation loss achieved
%
%   See also: PHNN_FORWARD, TAYLOR_EXPAND

    % Default parameters
    if nargin < 11, learning_rate = 0.001; end
    if nargin < 12, n_epochs = 200; end
    if nargin < 13, batch_size = 256; end
    if nargin < 14, verbose = true; end

    [n_samples, dim_in] = size(X_train);  %#ok<ASGLU>
    dim_out = size(Y_train, 2);
    n_monomials = length(monomials);

    % Initialize weights
    W_learn = W_learn_init;
    bias = bias_init;

    % Adam optimizer state
    m_W = zeros(dim_out, n_monomials);
    v_W = zeros(dim_out, n_monomials);
    m_b = zeros(dim_out, 1);
    v_b = zeros(dim_out, 1);
    beta1 = 0.9; beta2 = 0.999; eps_val = 1e-8;
    t = 0;

    % Training history
    train_losses = zeros(n_epochs, 1);
    val_losses = zeros(n_epochs, 1);
    best_val_loss = inf;
    best_W = W_learn;
    best_bias = bias;
    patience = 20;
    patience_counter = 0;

    n_batches = max(1, floor(n_samples / batch_size));

    for epoch = 1:n_epochs
        % Shuffle training data
        perm = randperm(n_samples);
        X_shuf = X_train(perm, :);
        Y_shuf = Y_train(perm, :);

        epoch_loss = 0;

        for b = 1:n_batches
            start_idx = (b-1) * batch_size + 1;
            end_idx = min(b * batch_size, n_samples);
            X_batch = X_shuf(start_idx:end_idx, :);
            Y_batch = Y_shuf(start_idx:end_idx, :);
            n_batch = end_idx - start_idx + 1;

            % ---- Forward pass ----
            M = taylor_expand(X_batch, monomials);  % (n_batch, n_mono)
            W_eff = A_value + A_uncertain .* W_learn;
            Y_pred = M * W_eff' + bias';  % (n_batch, dim_out)

            % ---- Loss ----
            error_mat = Y_pred - Y_batch;  % (n_batch, dim_out)
            batch_loss = mean(error_mat(:).^2);
            epoch_loss = epoch_loss + batch_loss;

            % ---- Gradients ----
            % dL/dW_learn:  (dim_out, n_mono)
            % Only for connections where A_uncertain == 1
            dW = (error_mat' * M) / n_batch;  % (dim_out, n_mono)
            dW = dW .* A_uncertain;  % Mask: only update learnable connections

            % dL/dbias
            db = mean(error_mat, 1)';  % (dim_out, 1)

            % ---- Adam update ----
            t = t + 1;

            m_W = beta1 * m_W + (1 - beta1) * dW;
            v_W = beta2 * v_W + (1 - beta2) * (dW.^2);
            m_W_hat = m_W / (1 - beta1^t);
            v_W_hat = v_W / (1 - beta2^t);
            W_learn = W_learn - learning_rate * m_W_hat ./ (sqrt(v_W_hat) + eps_val);

            m_b = beta1 * m_b + (1 - beta1) * db;
            v_b = beta2 * v_b + (1 - beta2) * (db.^2);
            m_b_hat = m_b / (1 - beta1^t);
            v_b_hat = v_b / (1 - beta2^t);
            bias = bias - learning_rate * m_b_hat ./ (sqrt(v_b_hat) + eps_val);
        end

        epoch_loss = epoch_loss / n_batches;

        % ---- Validation ----
        Y_val_pred = phnn_forward(X_val, monomials, A_value, A_uncertain, W_learn, bias);
        val_loss = mean((Y_val_pred(:) - Y_val(:)).^2);

        train_losses(epoch) = epoch_loss;
        val_losses(epoch) = val_loss;

        % Early stopping
        if val_loss < best_val_loss
            best_val_loss = val_loss;
            best_W = W_learn;
            best_bias = bias;
            patience_counter = 0;
        else
            patience_counter = patience_counter + 1;
        end

        if verbose && (mod(epoch, 20) == 0 || epoch == 1 || epoch == n_epochs)
            fprintf('  Epoch %4d: train_loss=%.6e, val_loss=%.6e\n', ...
                    epoch, epoch_loss, val_loss);
        end

        if patience_counter >= patience
            if verbose
                fprintf('  Early stopping at epoch %d\n', epoch);
            end
            break;
        end
    end

    % Restore best weights
    W_learn = best_W;
    bias = best_bias;

    % Trim loss histories
    train_losses = train_losses(1:epoch);
    val_losses = val_losses(1:epoch);
end
