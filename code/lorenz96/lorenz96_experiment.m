function results =       lorenz96_experiment(N, dt, F, expansion_order, n_train, n_val, n_test, n_epochs, batch_size, learning_rate, temporal_steps)
%% LORENZ96_EXPERIMENT  Physics-Regulated Neural Network Editing for Lorenz-96
%   Lorenz-96 PhNN Editing Experiment with PIM and TKM
%
%   results = lorenz96_experiment()
%   results = lorenz96_experiment(N, dt, F, expansion_order, n_train, n_val, n_test, n_epochs, batch_size, learning_rate, temporal_steps)
%
%   Inputs (all optional, with defaults):
%     N              - Lorenz-96 dimension (default: 40)
%     dt             - Time step for discretization (default: 0.01)
%     F              - Forcing parameter (default: 8.0)
%     expansion_order - Taylor expansion order r (default: 2)
%     n_train        - Training samples (default: 15000)
%     n_val          - Validation samples (default: 3000)
%     n_test         - Test samples (default: 3000)
%     n_epochs       - Training epochs (default: 150)
%     batch_size     - Mini-batch size (default: 256)
%     learning_rate  - Adam learning rate (default: 0.001)
%     temporal_steps - Number of temporal steps for TKM (default: 2)
%
%   This function implements the PhNN (Physics-compatible Neural Network)
%   framework with PIM (Physics Information Matrix) and TKM (Temporal
%   Knowledge Matrix) editing for the 40-dimensional Lorenz-96 system.
%
%   Key features:
%     - Lorenz-96 data generation (F=8, dt=0.01, RK4)
%     - Taylor-expansion-based PhNN architecture
%     - PIM editing: sparse neighbor-coupling structure
%     - TKM editing: temporal input decoupling
%     - Comparison: Unedited vs PIM vs TKM vs PIM+TKM

% =========================================================================
% Default parameters
% =========================================================================
if nargin < 1,  N = 40;              end
if nargin < 2,  dt = 0.01;           end
if nargin < 3,  F = 8.0;             end
if nargin < 4,  expansion_order = 2; end
if nargin < 5,  n_train = 15000;     end
if nargin < 6,  n_val = 3000;        end
if nargin < 7,  n_test = 3000;       end
if nargin < 8,  n_epochs = 150;      end
if nargin < 9,  batch_size = 256;    end
if nargin < 10, learning_rate = 0.001; end
if nargin < 11, temporal_steps = 2;  end
save_figures = true;

fprintf('%s\n', repmat('=', 1, 70));
fprintf('Lorenz-96 PhNN Editing Experiment\n');
fprintf('  Dimension N=%d, dt=%.2f, F=%.1f, expansion order r=%d\n', N, dt, F, expansion_order);
fprintf('%s\n', repmat('=', 1, 70));

% ------------------------------------------------------------------
% Step 1: Generate data
% ------------------------------------------------------------------
fprintf('\n[Step 1] Generating Lorenz-96 data...\n');
[train_traj, val_traj, test_traj] = generate_train_val_test_data(N, dt, F, n_train, n_val, n_test, 42);

% Standard (non-temporal) setup: x(k) -> x(k+1)
X_train = single(train_traj(1:end-1, :));
Y_train = single(train_traj(2:end, :));
X_val   = single(val_traj(1:end-1, :));
Y_val   = single(val_traj(2:end, :));
X_test  = single(test_traj(1:end-1, :));
Y_test  = single(test_traj(2:end, :));

fprintf('  Train: %d samples\n', size(X_train, 1));
fprintf('  Val:   %d samples\n', size(X_val, 1));
fprintf('  Test:  %d samples\n', size(X_test, 1));

% ------------------------------------------------------------------
% Step 2: Build monomial indices
% ------------------------------------------------------------------
fprintf('\n[Step 2] Building Taylor expansion (r=%d)...\n', expansion_order);

% Standard (no temporal)
monomials_std = generate_monomial_indices(N, expansion_order);
n_mono_std = length(monomials_std);
fprintf('  Standard input (%dD): %d monomials\n', N, n_mono_std);

% Temporal input
dim_temporal = N * temporal_steps;
monomials_temp = generate_monomial_indices(dim_temporal, expansion_order);
n_mono_temp = length(monomials_temp);
fprintf('  Temporal input (%dD, K=%d): %d monomials\n', dim_temporal, temporal_steps, n_mono_temp);

% ------------------------------------------------------------------
% Step 3: Build PIM and TKM masks for Lorenz-96
% ------------------------------------------------------------------
fprintf('\n[Step 3] Building PIM and TKM masks...\n');

% PIM for standard input
[A_value_pim, A_uncertain_pim, pim_sparsity] = build_lorenz96_pim(N, dt, monomials_std);
fprintf('  PIM sparsity: %.1f%%\n', pim_sparsity * 100);

% TKM for temporal input
[A_uncertain_tkm, tkm_sparsity] = build_lorenz96_tkm(N, monomials_temp, temporal_steps);
fprintf('  TKM sparsity (temporal input): %.1f%%\n', tkm_sparsity * 100);

% PIM+TKM combined for temporal input
[A_uncertain_pim_tkm, A_value_pim_temporal, pim_tkm_sparsity] = ...
    build_lorenz96_pim_tkm(N, dt, monomials_temp, temporal_steps, A_uncertain_tkm);
fprintf('  PIM+TKM combined sparsity: %.1f%%\n', pim_tkm_sparsity * 100);

% ------------------------------------------------------------------
% Step 4: Create and train models
% ------------------------------------------------------------------
fprintf('\n[Step 4] Creating and training models...\n');

results = struct();
models = struct();

% Prepare temporal data (for TKM models)
[X_train_t, Y_train_t] = build_temporal_data(train_traj, N, temporal_steps);
[X_val_t,   Y_val_t]   = build_temporal_data(val_traj,   N, temporal_steps);
[X_test_t,  Y_test_t]  = build_temporal_data(test_traj,  N, temporal_steps);

% ----- Model 1: Unedited PhNN (standard input) -----
fprintf('\n  [1/4] Unedited PhNN...\n');
model_unedited = PhNNModel(N, N, monomials_std);
model_unedited.summary();

t0 = tic;
[train_l, val_l, best_v] = model_unedited.train(X_train, Y_train, X_val, Y_val, ...
    learning_rate, n_epochs, batch_size, n_epochs+1);
t1 = toc(t0);

[rmse_u, rmse_std_u] = compute_autoregressive_rmse(model_unedited, X_test, test_traj, 200);

results.unedited = struct(...
    'train_losses', train_l, 'val_losses', val_l, 'best_val_loss', best_v, ...
    'rmse_by_step', rmse_u, 'rmse_std', rmse_std_u, ...
    'n_total', model_unedited.n_total, 'n_learnable', model_unedited.n_learnable, ...
    'sparsity', model_unedited.sparsity, 'train_time', t1);
models.unedited = model_unedited;
fprintf('  Train time: %.1fs, Best val loss: %.6e\n', t1, best_v);

% ----- Model 2: PIM-Edited PhNN -----
fprintf('\n  [2/4] PIM-Edited PhNN...\n');
model_pim = PhNNModel(N, N, monomials_std, A_value_pim, A_uncertain_pim);
model_pim.summary();

t0 = tic;
[train_l, val_l, best_v] = model_pim.train(X_train, Y_train, X_val, Y_val, ...
    learning_rate, n_epochs, batch_size, n_epochs+1);
t1 = toc(t0);

[rmse_p, rmse_std_p] = compute_autoregressive_rmse(model_pim, X_test, test_traj, 200);

results.pim = struct(...
    'train_losses', train_l, 'val_losses', val_l, 'best_val_loss', best_v, ...
    'rmse_by_step', rmse_p, 'rmse_std', rmse_std_p, ...
    'n_total', model_pim.n_total, 'n_learnable', model_pim.n_learnable, ...
    'sparsity', model_pim.sparsity, 'train_time', t1);
models.pim = model_pim;
fprintf('  Train time: %.1fs, Best val loss: %.6e\n', t1, best_v);

% ----- Model 3: TKM-Edited PhNN (temporal input) -----
fprintf('\n  [3/4] TKM-Edited PhNN (temporal input)...\n');
model_tkm = PhNNModel(dim_temporal, N, monomials_temp, [], A_uncertain_tkm);
model_tkm.summary();

t0 = tic;
[train_l, val_l, best_v] = model_tkm.train(X_train_t, Y_train_t, X_val_t, Y_val_t, ...
    learning_rate, n_epochs, batch_size, n_epochs+1);
t1 = toc(t0);

[rmse_t, rmse_std_t] = compute_autoregressive_rmse(model_tkm, X_test_t, test_traj, 200);

results.tkm = struct(...
    'train_losses', train_l, 'val_losses', val_l, 'best_val_loss', best_v, ...
    'rmse_by_step', rmse_t, 'rmse_std', rmse_std_t, ...
    'n_total', model_tkm.n_total, 'n_learnable', model_tkm.n_learnable, ...
    'sparsity', model_tkm.sparsity, 'train_time', t1);
models.tkm = model_tkm;
fprintf('  Train time: %.1fs, Best val loss: %.6e\n', t1, best_v);

% ----- Model 4: PIM+TKM Combined -----
fprintf('\n  [4/4] PIM+TKM Edited PhNN...\n');
model_pim_tkm = PhNNModel(dim_temporal, N, monomials_temp, ...
    A_value_pim_temporal, A_uncertain_pim_tkm);
model_pim_tkm.summary();

t0 = tic;
[train_l, val_l, best_v] = model_pim_tkm.train(X_train_t, Y_train_t, X_val_t, Y_val_t, ...
    learning_rate, n_epochs, batch_size, n_epochs+1);
t1 = toc(t0);

[rmse_pt, rmse_std_pt] = compute_autoregressive_rmse(model_pim_tkm, X_test_t, test_traj, 200);

results.pim_tkm = struct(...
    'train_losses', train_l, 'val_losses', val_l, 'best_val_loss', best_v, ...
    'rmse_by_step', rmse_pt, 'rmse_std', rmse_std_pt, ...
    'n_total', model_pim_tkm.n_total, 'n_learnable', model_pim_tkm.n_learnable, ...
    'sparsity', model_pim_tkm.sparsity, 'train_time', t1);
models.pim_tkm = model_pim_tkm;
fprintf('  Train time: %.1fs, Best val loss: %.6e\n', t1, best_v);

% ------------------------------------------------------------------
% Step 5: Results summary
% ------------------------------------------------------------------
fprintf('\n%s\n', repmat('=', 1, 70));
fprintf('[Step 5] RESULTS SUMMARY\n');
fprintf('%s\n', repmat('=', 1, 70));

fprintf('\n%-18s %-16s %-12s %-12s %-12s %-12s\n', ...
    'Model', 'Best Val Loss', 'Learnable', 'Sparsity', 'Train Time', 'RMSE@50');
fprintf('%s\n', repmat('-', 1, 82));

labels = struct('unedited', 'Unedited PhNN', 'pim', 'PIM-Edited', ...
                'tkm', 'TKM-Edited', 'pim_tkm', 'PIM+TKM');
names = fieldnames(results);
for i = 1:length(names)
    name = names{i};
    r = results.(name);
    if length(r.rmse_by_step) > 49
        rmse_50 = r.rmse_by_step(50);
    else
        rmse_50 = r.rmse_by_step(end);
    end
    fprintf('%-18s %-16.6e %-12d %-11.1f%% %-11.1fs %-12.4f\n', ...
        labels.(name), r.best_val_loss, r.n_learnable, ...
        r.sparsity*100, r.train_time, rmse_50);
end

% ------------------------------------------------------------------
% Step 6: Visualization
% ------------------------------------------------------------------
fprintf('\n[Step 6] Generating visualizations...\n');

% Generate all figures
plot_training_curves(results, save_figures);
plot_prediction_comparison(models, X_test, test_traj, 100, save_figures);
plot_rmse_vs_horizon(results, save_figures);
plot_model_complexity(results, save_figures);
plot_weight_matrix_comparison(models, save_figures);

fprintf('\n%s\n', repmat('=', 1, 70));
fprintf('Lorenz-96 PhNN Editing Experiment Complete!\n');
fprintf('%s\n', repmat('=', 1, 70));

% Save experiment data for offline figure regeneration (no re-simulation)
out_dir = fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))), 'results');  % <repo>/results
if ~exist(out_dir, 'dir'), mkdir(out_dir); end
save(fullfile(out_dir, 'lorenz96_results.mat'), 'results', 'models', 'test_traj');
fprintf('  Results saved to results/lorenz96_results.mat\n');

end

%% ========================================================================
%  Part 1: Lorenz-96 System Functions
%  ========================================================================

function dx = lorenz96_derivative(x, F)
% Lorenz-96 derivative: dx_i/dt = (x_{i+1} - x_{i-2})*x_{i-1} - x_i + F
    if nargin < 2, F = 8.0; end
    N = length(x);
    dx = zeros(N, 1);
    for i = 1:N
        ip1 = mod(i, N) + 1;           % i+1 (cyclic)
        im2 = mod(i-3, N) + 1;         % i-2 (cyclic)
        im1 = mod(i-2, N) + 1;         % i-1 (cyclic)
        dx(i) = (x(ip1) - x(im2)) * x(im1) - x(i) + F;
    end
end

function x_next = rk4_step(x, dt, F)
% Single RK4 integration step
    if nargin < 3, F = 8.0; end
    k1 = lorenz96_derivative(x, F);
    k2 = lorenz96_derivative(x + 0.5 * dt * k1, F);
    k3 = lorenz96_derivative(x + 0.5 * dt * k2, F);
    k4 = lorenz96_derivative(x + dt * k3, F);
    x_next = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4);
end

function trajectory = generate_lorenz96_trajectory(x0, dt, n_steps, F, spinup)
% Generate a Lorenz-96 trajectory.
    if nargin < 4, F = 8.0; end
    if nargin < 5, spinup = 5000; end

    N = length(x0);
    x = x0(:);

    % Spinup
    for s = 1:spinup
        x = rk4_step(x, dt, F);
    end

    % Collect trajectory
    trajectory = zeros(n_steps, N);
    for t = 1:n_steps
        x = rk4_step(x, dt, F);
        trajectory(t, :) = x';
    end
end

function [train_traj, val_traj, test_traj] = generate_train_val_test_data(N, dt, F, n_train, n_val, n_test, seed)
% Generate training, validation, and test datasets.
    rng(seed);

    % Generate training trajectories from different initial conditions
    n_train_traj = 4;
    train_data = cell(n_train_traj, 1);
    for i = 1:n_train_traj
        x0 = randn(N, 1) * 3.0 + F;
        train_data{i} = generate_lorenz96_trajectory(x0, dt, n_train / n_train_traj, F, 2000);
    end
    train_traj = cat(1, train_data{:});

    % Validation
    x0_val = randn(N, 1) * 3.0 + F;
    val_traj = generate_lorenz96_trajectory(x0_val, dt, n_val, F, 2000);

    % Test
    x0_test = randn(N, 1) * 3.0 + F;
    test_traj = generate_lorenz96_trajectory(x0_test, dt, n_test, F, 2000);
end

%% ========================================================================
%  Part 2: Taylor Expansion Functions
%  ========================================================================

function indices = generate_monomial_indices(dim, order)
% Generate indices for all Taylor monomials of orders 1..order.
% Returns a cell array of index vectors.
    indices = {};
    for r = 1:order
        combos = generate_combinations_with_replacement(dim, r);
        for c = 1:size(combos, 1)
            indices{end+1} = combos(c, :);  %#ok<AGROW>
        end
    end
end

function combos = generate_combinations_with_replacement(n, k)
% Generate all combinations with replacement of k elements from 1..n.
% Returns a matrix where each row is a combination (1-indexed for MATLAB).
    if k == 0
        combos = zeros(0, 0);
        return;
    end
    combos = generate_cwr_recursive(n, k, 1);
end

function result = generate_cwr_recursive(n, k, start_val)
% Recursively generate combinations with replacement (1-indexed output).
% n: number of values remaining, k: items to pick, start_val: smallest value.
    if k == 1
        result = (start_val:start_val+n-1)';
        return;
    end
    result = zeros(0, k);
    for i = 0:n-1
        val = start_val + i;
        sub = generate_cwr_recursive(n - i, k - 1, val);
        nr = size(sub, 1);
        result = [result; val * ones(nr, 1), sub]; %#ok<AGROW>
    end
end

function expanded = taylor_expand(X, monomial_indices)
% Expand input into Taylor monomials.
% X: (n_samples, dim) or (dim,) array
% monomial_indices: cell array of index vectors (1-based after conversion)
    [n_samples, dim] = size(X);
    n_monomials = length(monomial_indices);
    expanded = ones(n_samples, n_monomials);

    for h = 1:n_monomials
        indices = monomial_indices{h};
        for k = 1:length(indices)
            expanded(:, h) = expanded(:, h) .* X(:, indices(k));
        end
    end
end

function total = get_expanded_dim(dim, order)
% Calculate the number of Taylor monomials.
    total = 0;
    for r = 1:order
        total = total + nchoosek(dim + r - 1, r);
    end
end

%% ========================================================================
%  Part 3: PIM and TKM Construction
%  ========================================================================

function [A_value, A_uncertain, pim_sparsity] = build_lorenz96_pim(dim, dt, monomial_indices)
% Build the Physics Information Matrix (PIM) masks for Lorenz-96.
    n_monomials = length(monomial_indices);
    A_value = zeros(dim, n_monomials, 'single');
    A_uncertain = zeros(dim, n_monomials, 'single');

    for i = 1:dim
        % Relevant inputs for output i (cyclic), 1-based indexing
        i_m2 = mod(i-3, dim) + 1;  % i-2
        i_m1 = mod(i-2, dim) + 1;  % i-1
        i_p1 = mod(i, dim) + 1;    % i+1
        relevant = [i_m2, i_m1, i, i_p1];

        for h = 1:n_monomials
            midx = monomial_indices{h};

            % Check: do all inputs in this monomial belong to relevant?
            if all(ismember(midx, relevant))
                A_uncertain(i, h) = 1;  % Learnable
            end

            % Set known coefficients: first-order self-term x_i coeff = 1 - dt
            if length(midx) == 1 && midx(1) == i
                A_value(i, h) = 1.0 - dt;
            end
        end
    end

    pim_sparsity = 1.0 - mean(A_uncertain(:));
end

function [A_uncertain_tkm, tkm_sparsity] = build_lorenz96_tkm(dim, monomial_indices, K)
% Build Temporal Knowledge Matrix (TKM) masks.
    if nargin < 3, K = 2; end
    n_monomials = length(monomial_indices);
    A_uncertain_tkm = ones(dim, n_monomials, 'single');

    for h = 1:n_monomials
        midx = monomial_indices{h};
        % Determine which time step each variable belongs to (1-based)
        time_steps = unique(floor((midx - 1) / dim));

        if length(time_steps) > 1
            % This monomial mixes variables from different time steps -> PRUNE
            A_uncertain_tkm(:, h) = 0;
        end
    end

    tkm_sparsity = 1.0 - mean(A_uncertain_tkm(:));
end

function [A_uncertain_pim_tkm, A_value_pim_temporal, pim_tkm_sparsity] = ...
    build_lorenz96_pim_tkm(dim, dt, monomials_temp, temporal_steps, A_uncertain_tkm)
% Build combined PIM+TKM mask for temporal input.
    n_mono_temp = length(monomials_temp);
    A_uncertain_pim_temporal = zeros(dim, n_mono_temp, 'single');
    A_value_pim_temporal = zeros(dim, n_mono_temp, 'single');

    for h = 1:n_mono_temp
        midx = monomials_temp{h};

        for k = 0:temporal_steps-1
            block_start = k * dim + 1;
            block_end = (k + 1) * dim;
            % Get indices that fall in this temporal block (convert to 0..dim-1)
            in_block_mask = (midx >= block_start) & (midx <= block_end);
            if all(in_block_mask)
                indices_in_block = midx - block_start + 1;  % 1-based within block

                for i = 1:dim
                    i_m2 = mod(i-3, dim) + 1;
                    i_m1 = mod(i-2, dim) + 1;
                    i_p1 = mod(i, dim) + 1;
                    relevant = [i_m2, i_m1, i, i_p1];

                    if all(ismember(indices_in_block, relevant))
                        A_uncertain_pim_temporal(i, h) = 1;
                    end
                    if k == temporal_steps - 1 && length(indices_in_block) == 1 && indices_in_block(1) == i
                        A_value_pim_temporal(i, h) = 1.0 - dt;
                    end
                end
            end
        end
    end

    % Apply TKM mask on top
    A_uncertain_pim_tkm = A_uncertain_pim_temporal .* A_uncertain_tkm;
    pim_tkm_sparsity = 1.0 - mean(A_uncertain_pim_tkm(:));
end

function [X_temp, Y_temp] = build_temporal_data(traj, dim, K)
% Build temporal input [x(k-K+1), ..., x(k-1), x(k)] -> x(k+1)
% (oldest first, latest last)
    n_samples = size(traj, 1) - K;
    X_temp = zeros(n_samples, dim * K, 'single');
    for k = 1:K
        col_start = (k-1) * dim + 1;
        col_end = k * dim;
        X_temp(:, col_start:col_end) = single(traj(k:n_samples+k-1, :));
    end
    Y_temp = single(traj(K+1:end, :));
end

%% ========================================================================
%  Part 4: PhNN Model Class (using MATLAB classdef for clarity)
%  Note: The class is defined at the bottom as a separate classdef file.
%  For single-file deployment, we use a struct-based approach here.
%  See PhNNModel.m for the class-based implementation.
%  ========================================================================

function [train_losses, val_losses, best_val_loss] = train_phnn(W_learn, bias, A_value, A_uncertain, monomial_indices, ...
    X_train, Y_train, X_val, Y_val, learning_rate, n_epochs, batch_size, early_stopping_patience)
% Train PhNN using mini-batch gradient descent with Adam.
% This is a standalone function for use when the class is not available.
    if nargin < 13, early_stopping_patience = 20; end

    [n_samples, ~] = size(X_train);
    n_batches = max(1, floor(n_samples / batch_size));

    % Adam optimizer state
    m_W = zeros(size(W_learn), 'single');
    v_W = zeros(size(W_learn), 'single');
    m_b = zeros(size(bias), 'single');
    v_b = zeros(size(bias), 'single');
    beta1 = 0.9; beta2 = 0.999; eps_val = 1e-8;
    t_cnt = 0;

    train_losses = []; val_losses = [];
    best_val_loss = inf;
    best_W = []; best_b = [];
    patience_counter = 0;

    for epoch = 1:n_epochs
        % Shuffle
        idx = randperm(n_samples);
        X_shuffled = X_train(idx, :);
        Y_shuffled = Y_train(idx, :);

        epoch_loss = 0.0;
        for b = 1:n_batches
            start_idx = (b-1) * batch_size + 1;
            end_idx = min(start_idx + batch_size - 1, n_samples);
            X_batch = X_shuffled(start_idx:end_idx, :);
            Y_batch = Y_shuffled(start_idx:end_idx, :);

            % Forward pass
            M = taylor_expand(X_batch, monomial_indices);
            W_eff = A_value + A_uncertain .* W_learn;
            Y_pred = M * W_eff' + bias';

            % Gradients
            error = Y_pred - Y_batch;
            dW = (error' * M) / size(X_batch, 1);
            dW = dW .* A_uncertain;
            db = mean(error, 1)';

            % Adam update
            t_cnt = t_cnt + 1;
            m_W = beta1 * m_W + (1 - beta1) * dW;
            v_W = beta2 * v_W + (1 - beta2) * (dW.^2);
            m_W_hat = m_W / (1 - beta1^t_cnt);
            v_W_hat = v_W / (1 - beta2^t_cnt);
            W_learn = W_learn - learning_rate * m_W_hat ./ (sqrt(v_W_hat) + eps_val);

            m_b = beta1 * m_b + (1 - beta1) * db;
            v_b = beta2 * v_b + (1 - beta2) * (db.^2);
            m_b_hat = m_b / (1 - beta1^t_cnt);
            v_b_hat = v_b / (1 - beta2^t_cnt);
            bias = bias - learning_rate * m_b_hat ./ (sqrt(v_b_hat) + eps_val);

            epoch_loss = epoch_loss + mean(error(:).^2);
        end

        epoch_loss = epoch_loss / n_batches;
        train_losses(end+1) = epoch_loss;  %#ok<AGROW>

        % Validation
        M_val = taylor_expand(X_val, monomial_indices);
        W_eff_val = A_value + A_uncertain .* W_learn;
        Y_val_pred = M_val * W_eff_val' + bias';
        val_loss = mean((Y_val_pred(:) - Y_val(:)).^2);
        val_losses(end+1) = val_loss;  %#ok<AGROW>

        if val_loss < best_val_loss
            best_val_loss = val_loss;
            best_W = W_learn;
            best_b = bias;
            patience_counter = 0;
        else
            patience_counter = patience_counter + 1;
        end

        if mod(epoch, 20) == 0 || epoch == n_epochs
            fprintf('  Epoch %4d: train_loss=%.6e, val_loss=%.6e\n', epoch, epoch_loss, val_loss);
        end

        if patience_counter >= early_stopping_patience
            fprintf('  Early stopping at epoch %d\n', epoch);
            break;
        end
    end

    % Restore best weights
    if ~isempty(best_W)
        W_learn = best_W;
        bias = best_b;
    end
end

%% ========================================================================
%  Part 5: Multi-step Prediction and Error Analysis
%  ========================================================================

function predictions = multi_step_predict(model, x0, n_steps)
% Perform multi-step autoregressive prediction.
    [dim_out, ~] = size(model.A_value);
    predictions = zeros(n_steps, dim_out);
    x_current = x0(:)';

    for t = 1:n_steps
        x_next = model.forward(x_current);
        predictions(t, :) = x_next;
        x_current = x_next;
    end
end

function [rmse_by_step, rmse_std] = compute_autoregressive_rmse(model, X_test, trajectory, horizon)
% Compute multi-step autoregressive RMSE via true windowed rollout.
% Supports both standard (K=1) and temporal (K>1) models. With a window of
% K consecutive states [traj(s),...,traj(s+K-1)] (oldest first, matching
% build_temporal_data), the model predicts traj(s+K); horizon h targets
% traj(s+K+h-1).
    if nargin < 4, horizon = 200; end

    dim_out = model.dim_out;
    K = model.dim_in / dim_out;   % window length: 1 for standard, 2 for temporal

    n_traj = size(trajectory, 1);
    max_start = n_traj - K - horizon + 1;
    if max_start < 1
        rmse_by_step = NaN(1, horizon);
        rmse_std = NaN(1, horizon);
        return;
    end

    n_test_points = min(20, max_start);
    start_indices = randperm(max_start, n_test_points);

    all_errors = zeros(n_test_points, horizon);

    for s = 1:n_test_points
        start_idx = start_indices(s);
        window = trajectory(start_idx : start_idx + K - 1, :);   % K x dim_out

        for h = 1:horizon
            x_input = reshape(window', 1, []);      % 1 x dim_in, oldest first
            x_pred  = model.forward(x_input);       % 1 x dim_out

            true_state = trajectory(start_idx + K + h - 1, :);
            all_errors(s, h) = sqrt(mean((x_pred - true_state).^2));

            window = [window(2:end, :); x_pred];    % slide: drop oldest, append prediction
        end
    end

    rmse_by_step = mean(all_errors, 1);
    rmse_std = std(all_errors, 0, 1);
end

%% ========================================================================
%  Part 6: Visualization Functions
%  ========================================================================

function plot_training_curves(results, save_figures)
% Plot training and validation loss curves.
    if nargin < 2, save_figures = false; end

    colors = struct('unedited', [0.85 0.33 0.10], 'pim', [0 0.45 0.74], ...
                    'tkm', [0.93 0.69 0.13], 'pim_tkm', [0.49 0.18 0.56]);
    labels = struct('unedited', 'Unedited PhNN', 'pim', 'PIM-Edited PhNN', ...
                    'tkm', 'TKM-Edited PhNN', 'pim_tkm', 'PIM+TKM Edited PhNN');

    figure('Position', [100, 100, 1200, 450]);

    subplot(1, 2, 1);
    hold on;
    names = fieldnames(results);
    for i = 1:length(names)
        name = names{i};
        r = results.(name);
        c = colors.(name);
        lbl = labels.(name);
        smoothed = conv(r.train_losses, ones(1, 10)/10, 'valid');
        plot(smoothed, 'Color', [c, 0.5], 'LineWidth', 0.5);
        plot(smoothed, 'Color', c, 'LineWidth', 2, 'DisplayName', [lbl ' (train)']);
    end
    xlabel('Epoch'); ylabel('Training MSE Loss');
    set(gca, 'YScale', 'log');
    title('Training Loss (smoothed)');
    legend('Location', 'best', 'FontSize', 8);
    grid on;

    subplot(1, 2, 2);
    hold on;
    for i = 1:length(names)
        name = names{i};
        r = results.(name);
        c = colors.(name);
        lbl = labels.(name);
        plot(r.val_losses, 'Color', c, 'LineWidth', 2, ...
            'DisplayName', sprintf('%s (val=%.4e)', lbl, r.best_val_loss));
    end
    xlabel('Epoch'); ylabel('Validation MSE Loss');
    set(gca, 'YScale', 'log');
    title('Validation Loss');
    legend('Location', 'best', 'FontSize', 8);
    grid on;

    if save_figures
        if ~exist('fig','dir'), mkdir('fig'); end
        saveas(gcf, 'fig/Lorenz96_TrainingCurves.png');
    end
end

function plot_prediction_comparison(models, X_test, trajectory, n_steps, save_figures)
% Plot multi-step prediction for representative dimensions.
    if nargin < 4, n_steps = 100; end
    if nargin < 5, save_figures = false; end

    colors = struct('unedited', [0.85 0.33 0.10], 'pim', [0 0.45 0.74], ...
                    'tkm', [0.93 0.69 0.13], 'pim_tkm', [0.49 0.18 0.56]);
    labels = struct('unedited', 'Unedited', 'pim', 'PIM', ...
                    'tkm', 'TKM', 'pim_tkm', 'PIM+TKM');

    dims_to_plot = [1, 6, 11, 21, 31, 36];  % 1-based (0,5,10,20,30,35 in Python)
    x0 = X_test(1, :);

    figure('Position', [100, 100, 1400, 700]);

    for ax_idx = 1:6
        subplot(2, 3, ax_idx);
        dim = dims_to_plot(ax_idx);
        hold on;

        % Ground truth
        true_vals = trajectory(1:n_steps, dim);
        plot(true_vals, 'k-', 'LineWidth', 1.5, 'DisplayName', 'True');

        names = fieldnames(models);
        for i = 1:length(names)
            name = names{i};
            if any(strcmp(name, {'tkm', 'pim_tkm'}))
                continue;  % Skip temporal models
            end
            model = models.(name);
            c = colors.(name);
            lbl = labels.(name);
            preds = multi_step_predict(model, x0, n_steps);
            plot(preds(:, dim), '--', 'Color', c, 'LineWidth', 1.2, 'DisplayName', lbl);
        end

        title(sprintf('$x_{%d}$', dim-1), 'Interpreter', 'latex');
        xlabel('Step'); ylabel('Value');
        legend('Location', 'best', 'FontSize', 7);
        grid on;
    end

    sgtitle('Multi-Step Autoregressive Prediction Comparison (Lorenz-96, N=40)', ...
        'FontWeight', 'bold', 'FontSize', 14);

    if save_figures
        if ~exist('fig','dir'), mkdir('fig'); end
        saveas(gcf, 'fig/Lorenz96_Predictions.png');
    end
end

function plot_rmse_vs_horizon(results, save_figures)
% Plot RMSE as a function of prediction horizon.
    if nargin < 2, save_figures = false; end

    colors = struct('unedited', [0.85 0.33 0.10], 'pim', [0 0.45 0.74], ...
                    'tkm', [0.93 0.69 0.13], 'pim_tkm', [0.49 0.18 0.56]);
    labels = struct('unedited', 'Unedited PhNN', 'pim', 'PIM-Edited PhNN', ...
                    'tkm', 'TKM-Edited PhNN', 'pim_tkm', 'PIM+TKM Edited PhNN');

    figure('Position', [100, 100, 900, 500]);
    hold on;

    names = fieldnames(results);
    for i = 1:length(names)
        name = names{i};
        r = results.(name);
        c = colors.(name);
        lbl = labels.(name);

        % Skip models with NaN RMSE (e.g., unedited model divergence)
        rmse_vals = r.rmse_by_step;
        if all(isnan(rmse_vals))
            continue;
        end
        % Replace any remaining NaN with Inf for log-scale visibility
        rmse_vals(isnan(rmse_vals)) = inf;

        horizon = length(rmse_vals);
        steps = 0:horizon-1;
        plot(steps, rmse_vals, 'Color', c, 'LineWidth', 2, 'DisplayName', lbl);
        if isfield(r, 'rmse_std')
            std_vals = r.rmse_std;
            std_vals(isnan(std_vals)) = 0;
            lo = rmse_vals - std_vals;
            hi = rmse_vals + std_vals;
            lo(lo <= 0) = 1e-10;  % avoid non-positive for log scale
            fill([steps, fliplr(steps)], ...
                 [lo, fliplr(hi)], ...
                 c, 'FaceAlpha', 0.15, 'EdgeColor', 'none');
        end
    end

    xlabel('Prediction Horizon (steps)', 'FontSize', 12);
    ylabel('RMSE', 'FontSize', 12);
    set(gca, 'YScale', 'log');
    title('Autoregressive Prediction Error vs Horizon (Lorenz-96, N=40)', 'FontSize', 13);
    legend('Location', 'best', 'FontSize', 10);
    grid on;

    if save_figures
        if ~exist('fig','dir'), mkdir('fig'); end
        saveas(gcf, 'fig/Lorenz96_RMSEvsHorizon.png');
    end
end

function plot_model_complexity(results, save_figures)
% Bar chart comparing model complexity metrics.
    if nargin < 2, save_figures = false; end

    names_order = {'unedited', 'pim', 'tkm', 'pim_tkm'};
    display_names = {'Unedited', 'PIM-Edited', 'TKM-Edited', 'PIM+TKM'};
    cs = [0.85 0.33 0.10; 0 0.45 0.74; 0.93 0.69 0.13; 0.49 0.18 0.56];

    figure('Position', [100, 100, 1300, 400]);

    total_params = zeros(1, 4);
    learnable = zeros(1, 4);
    sparsity = zeros(1, 4);
    for i = 1:4
        name = names_order{i};
        r = results.(name);
        total_params(i) = r.n_total;
        learnable(i) = r.n_learnable;
        sparsity(i) = r.sparsity * 100;
    end

    subplot(1, 3, 1);
    b = bar(total_params, 'FaceAlpha', 0.7);
    b.FaceColor = 'flat';
    for i = 1:4, b.CData(i,:) = cs(i,:); end
    set(gca, 'XTickLabel', display_names);
    title('Total Connections'); ylabel('Count');
    for i = 1:4, text(i, total_params(i) + max(total_params)*0.02, num2str(total_params(i)), ...
            'HorizontalAlignment', 'center', 'FontSize', 9); end

    subplot(1, 3, 2);
    b = bar(learnable, 'FaceAlpha', 0.7);
    b.FaceColor = 'flat';
    for i = 1:4, b.CData(i,:) = cs(i,:); end
    set(gca, 'XTickLabel', display_names);
    title('Learnable Parameters'); ylabel('Count');
    for i = 1:4, text(i, learnable(i) + max(learnable)*0.02, num2str(learnable(i)), ...
            'HorizontalAlignment', 'center', 'FontSize', 9); end

    subplot(1, 3, 3);
    b = bar(sparsity, 'FaceAlpha', 0.7);
    b.FaceColor = 'flat';
    for i = 1:4, b.CData(i,:) = cs(i,:); end
    set(gca, 'XTickLabel', display_names);
    title('Sparsity (%)'); ylabel('%');
    for i = 1:4, text(i, sparsity(i) + 1, sprintf('%.1f%%', sparsity(i)), ...
            'HorizontalAlignment', 'center', 'FontSize', 9); end

    sgtitle('Model Complexity Comparison', 'FontWeight', 'bold', 'FontSize', 13);

    if save_figures
        if ~exist('fig','dir'), mkdir('fig'); end
        saveas(gcf, 'fig/Lorenz96_Complexity.png');
    end
end

function plot_weight_matrix_comparison(models, save_figures)
% Visualize the weight matrices of different models.
    if nargin < 2, save_figures = false; end

    names_order = {'unedited', 'pim', 'tkm', 'pim_tkm'};
    display_names = {'Unedited PhNN', 'PIM-Edited', 'TKM-Edited', 'PIM+TKM'};

    figure('Position', [100, 100, 1600, 450]);

    for ax_idx = 1:4
        subplot(1, 4, ax_idx);
        name = names_order{ax_idx};
        model = models.(name);
        W_eff = abs(model.A_value + model.A_uncertain .* model.W_learn);
        n_show = min(100, size(W_eff, 2));
        imagesc(W_eff(:, 1:n_show));
        colormap('hot'); colorbar;
        title(sprintf('%s\n%d learnable params', display_names{ax_idx}, model.n_learnable), 'FontSize', 10);
        xlabel('Hidden Neuron Index');
        if ax_idx == 1, ylabel('Output Dimension'); end
    end

    sgtitle('Effective Weight Matrix |W_{eff}| (first 100 hidden neurons)', ...
        'FontWeight', 'bold', 'FontSize', 13);

    if save_figures
        if ~exist('fig','dir'), mkdir('fig'); end
        saveas(gcf, 'fig/Lorenz96_Weights.png');
    end
end
