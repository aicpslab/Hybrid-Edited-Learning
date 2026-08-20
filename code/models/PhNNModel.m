classdef PhNNModel < handle
    %% PhNNModel  Physics-compatible Neural Network with Taylor expansion
    %
    % Architecture:
    %   Input:  x in R^{dim_in}
    %   Hidden: m(x,r) in R^{n_monomials} (Taylor expansion)
    %   Output: y = (W_value + W_uncertain .* W_learn) * m(x,r) + b
    %
    % where:
    %   W_value     - Fixed known physical weights (from PIM)
    %   W_learn     - Learnable weights
    %   W_uncertain - Binary mask (1=learn, 0=prune)
    %
    % Usage:
    %   model = PhNNModel(dim_in, dim_out, monomial_indices)
    %   model = PhNNModel(dim_in, dim_out, monomial_indices, A_value, A_uncertain)

    properties
        dim_in
        dim_out
        monomial_indices
        n_monomials
        A_value
        A_uncertain
        W_learn
        bias
        n_total
        n_learnable
        n_fixed
        n_pruned
        sparsity
    end

    methods
        function obj = PhNNModel(dim_in, dim_out, monomial_indices, A_value, A_uncertain)
            % Constructor
            obj.dim_in = dim_in;
            obj.dim_out = dim_out;
            obj.monomial_indices = monomial_indices;
            obj.n_monomials = length(monomial_indices);

            if nargin < 4 || isempty(A_value)
                obj.A_value = zeros(dim_out, obj.n_monomials, 'single');
            else
                obj.A_value = single(A_value);
            end

            if nargin < 5 || isempty(A_uncertain)
                obj.A_uncertain = ones(dim_out, obj.n_monomials, 'single');
            else
                obj.A_uncertain = single(A_uncertain);
            end

            % Learnable weights (initialized only for uncertain connections)
            obj.W_learn = single(randn(dim_out, obj.n_monomials) * 0.01);
            obj.bias = zeros(dim_out, 1, 'single');

            % Count parameters
            obj.n_total = obj.n_monomials * dim_out;
            obj.n_learnable = sum(obj.A_uncertain(:));
            obj.n_fixed = sum(abs(obj.A_value(:)) > 1e-10);
            % Pruned = weights that are neither learnable (A_uncertain=1)
            % nor fixed (A_value ~= 0). The two sets can overlap, so we
            % count their union rather than subtracting both.
            obj.n_pruned = obj.n_total - sum(obj.A_uncertain(:) | (abs(obj.A_value(:)) > 1e-10));
            if obj.n_total > 0
                obj.sparsity = obj.n_pruned / obj.n_total;
            else
                obj.sparsity = 0.0;
            end
        end

        function Y = forward(obj, X)
            % Forward pass.
            % X: (n_samples, dim_in) array
            % Returns: (n_samples, dim_out) array
            M = taylor_expand(X, obj.monomial_indices);
            W_eff = obj.A_value + obj.A_uncertain .* obj.W_learn;
            Y = M * W_eff' + obj.bias';
        end

        function loss = compute_loss(obj, X, Y_true)
            % MSE loss
            Y_pred = obj.forward(X);
            loss = mean((Y_pred(:) - Y_true(:)).^2);
        end

        function [dW_learn, dbias] = compute_gradient(obj, X, Y_true)
            % Compute gradients of loss w.r.t. W_learn and bias.
            n_samples = size(X, 1);
            M = taylor_expand(X, obj.monomial_indices);

            W_eff = obj.A_value + obj.A_uncertain .* obj.W_learn;
            Y_pred = M * W_eff' + obj.bias';

            error = Y_pred - Y_true;

            dW_learn = (error' * M) / n_samples;
            dW_learn = dW_learn .* obj.A_uncertain;

            dbias = mean(error, 1)';
        end

        function [train_losses, val_losses, best_val_loss] = train(obj, ...
                X_train, Y_train, X_val, Y_val, learning_rate, n_epochs, batch_size, early_stopping_patience)
            % Train using mini-batch gradient descent with Adam.
            if nargin < 9, early_stopping_patience = 20; end

            n_samples = size(X_train, 1);
            n_batches = max(1, floor(n_samples / batch_size));

            % Adam optimizer state
            m_W = zeros(size(obj.W_learn), 'single');
            v_W = zeros(size(obj.W_learn), 'single');
            m_b = zeros(size(obj.bias), 'single');
            v_b = zeros(size(obj.bias), 'single');
            beta1 = 0.9; beta2 = 0.999; eps_val = 1e-8;
            t_cnt = 0;

            train_losses = zeros(1, n_epochs);
            val_losses = zeros(1, n_epochs);
            best_val_loss = inf;
            best_weights = [];
            best_bias = [];
            patience_counter = 0;
            actual_epochs = n_epochs;

            for epoch = 1:n_epochs
                % Shuffle
                idx = randperm(n_samples);
                X_s = X_train(idx, :);
                Y_s = Y_train(idx, :);

                epoch_loss = 0.0;
                for b = 1:n_batches
                    st = (b-1) * batch_size + 1;
                    en = min(st + batch_size - 1, n_samples);
                    X_batch = X_s(st:en, :);
                    Y_batch = Y_s(st:en, :);

                    % Compute gradients
                    [dW, db] = obj.compute_gradient(X_batch, Y_batch);

                    % Adam update
                    t_cnt = t_cnt + 1;
                    m_W = beta1 * m_W + (1 - beta1) * dW;
                    v_W = beta2 * v_W + (1 - beta2) * (dW.^2);
                    m_W_hat = m_W / (1 - beta1^t_cnt);
                    v_W_hat = v_W / (1 - beta2^t_cnt);
                    obj.W_learn = obj.W_learn - learning_rate * m_W_hat ./ (sqrt(v_W_hat) + eps_val);

                    m_b = beta1 * m_b + (1 - beta1) * db;
                    v_b = beta2 * v_b + (1 - beta2) * (db.^2);
                    m_b_hat = m_b / (1 - beta1^t_cnt);
                    v_b_hat = v_b / (1 - beta2^t_cnt);
                    obj.bias = obj.bias - learning_rate * m_b_hat ./ (sqrt(v_b_hat) + eps_val);

                    Y_batch_pred = obj.forward(X_batch);
                    epoch_loss = epoch_loss + mean((Y_batch_pred(:) - Y_batch(:)).^2);
                end

                epoch_loss = epoch_loss / n_batches;
                train_losses(epoch) = epoch_loss;

                % Validation
                val_loss = obj.compute_loss(X_val, Y_val);
                val_losses(epoch) = val_loss;

                if val_loss < best_val_loss
                    best_val_loss = val_loss;
                    best_weights = obj.W_learn;
                    best_bias = obj.bias;
                    patience_counter = 0;
                else
                    patience_counter = patience_counter + 1;
                end

                if mod(epoch, 20) == 0 || epoch == n_epochs
                    fprintf('  Epoch %4d: train_loss=%.6e, val_loss=%.6e\n', epoch, epoch_loss, val_loss);
                end

                if patience_counter >= early_stopping_patience
                    fprintf('  Early stopping at epoch %d\n', epoch);
                    actual_epochs = epoch;
                    break;
                end
            end

            % Trim loss history
            train_losses = train_losses(1:actual_epochs);
            val_losses = val_losses(1:actual_epochs);

            % Restore best weights
            if ~isempty(best_weights)
                obj.W_learn = best_weights;
                obj.bias = best_bias;
            end
        end

        function summary(obj)
            % Print model summary.
            fprintf('  Input dim:       %d\n', obj.dim_in);
            fprintf('  Hidden (Taylor): %d\n', obj.n_monomials);
            fprintf('  Output dim:      %d\n', obj.dim_out);
            fprintf('  Total weights:   %d\n', obj.n_total);
            fprintf('  Fixed (PIM):     %d\n', obj.n_fixed);
            fprintf('  Learnable:       %d\n', obj.n_learnable);
            fprintf('  Pruned:          %d\n', obj.n_pruned);
            fprintf('  Sparsity:        %.1f%%\n', obj.sparsity * 100);
        end
    end
end

%% Local helper function (also used by external functions via the class)
function expanded = taylor_expand(X, monomial_indices)
% Expand input into Taylor monomials.
    [n_samples, dim] = size(X);  %#ok<NASGU>
    n_monomials = length(monomial_indices);
    expanded = ones(n_samples, n_monomials);

    for h = 1:n_monomials
        indices = monomial_indices{h};
        for k = 1:length(indices)
            expanded(:, h) = expanded(:, h) .* X(:, indices(k));
        end
    end
end
