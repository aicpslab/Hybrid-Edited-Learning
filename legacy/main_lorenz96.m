%% ========================================================================
% LORENZ-96 PhNN EDITING EXPERIMENT
% ========================================================================
% Physics-Regulated Neural Network Editing for High-Dimensional Chaos
%
% This script demonstrates the PIM (Physics Information Matrix) and
% TKM (Temporal Knowledge Matrix) editing methods on the 40-dimensional
% Lorenz-96 chaotic system.
%
% Experiment structure:
%   Model 1: Unedited PhNN (baseline, all connections learnable)
%   Model 2: PIM-Edited PhNN (physics-guided pruning, sparse ring topology)
%   Model 3: TKM-Edited PhNN (temporal decoupling pruning)
%   Model 4: PIM+TKM Edited PhNN (combined editing)
%
% All models use IDENTICAL architecture, data, epochs, and hyperparameters.
% The ONLY difference is the editing mask.
%
% Author:  Yang Yejiang (Southwest Minzu University)
% Date:    July 2026
% Journal: Journal of the Franklin Institute (target)
% ========================================================================

clear; close all; clc;

%% ========================================================================
% SECTION 1: EXPERIMENT PARAMETERS
% ========================================================================
fprintf('=============================================================\n');
fprintf('LORENZ-96 PhNN EDITING EXPERIMENT\n');
fprintf('=============================================================\n\n');

% --- Lorenz-96 system parameters ---
N = 40;         % State dimension (40D)
dt = 0.01;      % Time step for discrete mapping
F = 8.0;        % Forcing (standard chaotic regime)

% --- Taylor expansion ---
expansion_order = 2;  % Taylor order r

% --- Dataset sizes ---
n_train = 8000;   % Training samples
n_val   = 2000;   % Validation samples
n_test  = 2000;   % Test samples

% --- Training hyperparameters (IDENTICAL for all models) ---
n_epochs = 200;
batch_size = 256;
learning_rate = 0.001;
seed = 42;

% --- Temporal input setup for TKM ---
K_temporal = 2;  % Number of time steps: [x(k), x(k-1)]

%% ========================================================================
% SECTION 2: DATA GENERATION
% ========================================================================
fprintf('\n[Step 1/5] Generating Lorenz-96 data (N=%d, F=%.1f)...\n', N, F);

[X_train, Y_train, X_val, Y_val, X_test, Y_test] = ...
    generate_lorenz96_data(N, dt, F, n_train, n_val, n_test, seed);

% Build temporal datasets for TKM models
% Input: [x(k), x(k-1)] -> Output: x(k+1)
fprintf('  Building temporal datasets (K=%d)...\n', K_temporal);
[X_train_t, Y_train_t] = build_temporal_data(X_train, Y_train, N, K_temporal);
[X_val_t, Y_val_t]     = build_temporal_data(X_val, Y_val, N, K_temporal);
[X_test_t, Y_test_t]   = build_temporal_data(X_test, Y_test, N, K_temporal);

fprintf('  Standard:  train(%d,%d), val(%d,%d), test(%d,%d)\n', ...
        size(X_train), size(X_val), size(X_test));
fprintf('  Temporal:  train(%d,%d), val(%d,%d), test(%d,%d)\n', ...
        size(X_train_t), size(X_val_t), size(X_test_t));

%% ========================================================================
% SECTION 3: TAYLOR EXPANSION & EDITING MASKS
% ========================================================================
fprintf('\n[Step 2/5] Building Taylor expansion and editing masks...\n');

% --- Monomials for standard input (40D) ---
monomials_std = generate_monomials(N, expansion_order);
n_mono_std = length(monomials_std);
fprintf('  Standard input (%dD): %d monomials (r=%d)\n', N, n_mono_std, expansion_order);

% --- Monomials for temporal input (80D) ---
dim_temporal = N * K_temporal;
monomials_temp = generate_monomials(dim_temporal, expansion_order);
n_mono_temp = length(monomials_temp);
fprintf('  Temporal input (%dD): %d monomials (r=%d)\n', dim_temporal, n_mono_temp, expansion_order);

% --- PIM Mask ---
fprintf('\n  --- PIM Mask ---\n');
[A_value_pim, A_uncertain_pim, pim_sparsity] = build_pim_lorenz96(N, dt, monomials_std);

% --- TKM Mask ---
fprintf('\n  --- TKM Mask ---\n');
[A_uncertain_tkm, tkm_sparsity] = build_tkm_lorenz96(N, K_temporal, monomials_temp);

% --- PIM+TKM Combined Mask (for temporal input) ---
% Apply PIM within each temporal block, then TKM on top
fprintf('\n  --- PIM+TKM Combined Mask ---\n');
[A_value_pt, A_uncertain_pt] = build_pim_tkm_combined(N, dt, K_temporal, ...
    monomials_temp, A_uncertain_tkm);
combined_sparsity = 1.0 - mean(A_uncertain_pt(:));
fprintf('  PIM+TKM combined sparsity: %.1f%%\n', combined_sparsity * 100);

%% ========================================================================
% SECTION 4: MODEL TRAINING
% ========================================================================
fprintf('\n[Step 3/5] Training models (all: %d epochs, batch=%d, lr=%.4f)...\n', ...
        n_epochs, batch_size, learning_rate);

results = struct();

% ---- MODEL 1: Unedited PhNN (standard 40D input) ----
fprintf('\n  [1/4] Unedited PhNN (standard 40D input)...\n');

W_init_std = single(randn(N, n_mono_std) * 0.01);
b_init_std = single(zeros(N, 1));
A_val_zero = single(zeros(N, n_mono_std));
A_unc_all  = single(ones(N, n_mono_std));

tic;
[W_u, b_u, tl_u, vl_u, bv_u] = phnn_train(...
    X_train, Y_train, X_val, Y_val, monomials_std, ...
    A_val_zero, A_unc_all, W_init_std, b_init_std, ...
    learning_rate, n_epochs, batch_size, true);
t_u = toc;

Y_pred_u = phnn_forward(X_test, monomials_std, A_val_zero, A_unc_all, W_u, b_u);
rmse_u = sqrt(mean((Y_pred_u(:) - Y_test(:)).^2));

results.unedited.W = W_u; results.unedited.b = b_u;
results.unedited.train_losses = tl_u;
results.unedited.val_losses = vl_u;
results.unedited.best_val_loss = bv_u;
results.unedited.test_rmse = rmse_u;
results.unedited.n_learnable = sum(A_unc_all(:));
results.unedited.sparsity = 0;
results.unedited.train_time = t_u;

fprintf('  -> Best val loss=%.6e, Test RMSE=%.6e, Params=%d, Time=%.0fs\n', ...
        bv_u, rmse_u, results.unedited.n_learnable, t_u);

% ---- MODEL 2: PIM-Edited PhNN (standard 40D input) ----
fprintf('\n  [2/4] PIM-Edited PhNN (standard 40D input)...\n');
fprintf('  PIM sparsity: %.1f%%, Learnable: %d\n', ...
        pim_sparsity*100, sum(A_uncertain_pim(:)));

W_init_pim = single(randn(N, n_mono_std) * 0.01);

tic;
[W_p, b_p, tl_p, vl_p, bv_p] = phnn_train(...
    X_train, Y_train, X_val, Y_val, monomials_std, ...
    A_value_pim, A_uncertain_pim, W_init_pim, b_init_std, ...
    learning_rate, n_epochs, batch_size, true);
t_p = toc;

Y_pred_p = phnn_forward(X_test, monomials_std, A_value_pim, A_uncertain_pim, W_p, b_p);
rmse_p = sqrt(mean((Y_pred_p(:) - Y_test(:)).^2));

results.pim.W = W_p; results.pim.b = b_p;
results.pim.train_losses = tl_p;
results.pim.val_losses = vl_p;
results.pim.best_val_loss = bv_p;
results.pim.test_rmse = rmse_p;
results.pim.n_learnable = sum(A_uncertain_pim(:));
results.pim.sparsity = pim_sparsity;
results.pim.train_time = t_p;

fprintf('  -> Best val loss=%.6e, Test RMSE=%.6e, Params=%d, Time=%.0fs\n', ...
        bv_p, rmse_p, results.pim.n_learnable, t_p);

% ---- MODEL 3: TKM-Edited PhNN (temporal 80D input) ----
fprintf('\n  [3/4] TKM-Edited PhNN (temporal 80D input)...\n');
fprintf('  TKM sparsity: %.1f%%, Learnable: %d\n', ...
        tkm_sparsity*100, sum(A_uncertain_tkm(:)));

W_init_tkm = single(randn(N, n_mono_temp) * 0.01);
b_init_tkm = single(zeros(N, 1));
A_val_tkm = single(zeros(N, n_mono_temp));

tic;
[W_t, b_t, tl_t, vl_t, bv_t] = phnn_train(...
    X_train_t, Y_train_t, X_val_t, Y_val_t, monomials_temp, ...
    A_val_tkm, A_uncertain_tkm, W_init_tkm, b_init_tkm, ...
    learning_rate, n_epochs, batch_size, true);
t_t = toc;

Y_pred_t = phnn_forward(X_test_t, monomials_temp, A_val_tkm, A_uncertain_tkm, W_t, b_t);
rmse_t = sqrt(mean((Y_pred_t(:) - Y_test_t(:)).^2));

results.tkm.W = W_t; results.tkm.b = b_t;
results.tkm.train_losses = tl_t;
results.tkm.val_losses = vl_t;
results.tkm.best_val_loss = bv_t;
results.tkm.test_rmse = rmse_t;
results.tkm.n_learnable = sum(A_uncertain_tkm(:));
results.tkm.sparsity = tkm_sparsity;
results.tkm.train_time = t_t;

fprintf('  -> Best val loss=%.6e, Test RMSE=%.6e, Params=%d, Time=%.0fs\n', ...
        bv_t, rmse_t, results.tkm.n_learnable, t_t);

% ---- MODEL 4: PIM+TKM Edited PhNN (temporal 80D input) ----
fprintf('\n  [4/4] PIM+TKM Edited PhNN (temporal 80D input)...\n');
fprintf('  Combined sparsity: %.1f%%, Learnable: %d\n', ...
        combined_sparsity*100, sum(A_uncertain_pt(:)));

W_init_pt = single(randn(N, n_mono_temp) * 0.01);
b_init_pt = single(zeros(N, 1));

tic;
[W_pt, b_pt, tl_pt, vl_pt, bv_pt] = phnn_train(...
    X_train_t, Y_train_t, X_val_t, Y_val_t, monomials_temp, ...
    A_value_pt, A_uncertain_pt, W_init_pt, b_init_pt, ...
    learning_rate, n_epochs, batch_size, true);
t_pt = toc;

Y_pred_pt = phnn_forward(X_test_t, monomials_temp, A_value_pt, A_uncertain_pt, W_pt, b_pt);
rmse_pt = sqrt(mean((Y_pred_pt(:) - Y_test_t(:)).^2));

results.pim_tkm.W = W_pt; results.pim_tkm.b = b_pt;
results.pim_tkm.train_losses = tl_pt;
results.pim_tkm.val_losses = vl_pt;
results.pim_tkm.best_val_loss = bv_pt;
results.pim_tkm.test_rmse = rmse_pt;
results.pim_tkm.n_learnable = sum(A_uncertain_pt(:));
results.pim_tkm.sparsity = combined_sparsity;
results.pim_tkm.train_time = t_pt;

fprintf('  -> Best val loss=%.6e, Test RMSE=%.6e, Params=%d, Time=%.0fs\n', ...
        bv_pt, rmse_pt, results.pim_tkm.n_learnable, t_pt);

%% ========================================================================
% SECTION 5: RESULTS SUMMARY
% ========================================================================
fprintf('\n[Step 4/5] Results Summary...\n');
fprintf('=================================================================\n');
fprintf('Lorenz-96 PhNN Editing Experiment Results (N=%d, r=%d)\n', N, expansion_order);
fprintf('All models: %d epochs, batch=%d, lr=%.4f\n', n_epochs, batch_size, learning_rate);
fprintf('=================================================================\n');

fprintf('\n%-20s %-14s %-14s %-10s %-10s %-8s\n', ...
        'Model', 'Val Loss', 'Test RMSE', 'Learnable', 'Sparsity', 'Time');
fprintf('%s\n', repmat('-', 1, 80));

models = {'unedited', 'pim', 'tkm', 'pim_tkm'};
model_names = {'Unedited PhNN', 'PIM-Edited', 'TKM-Edited', 'PIM+TKM'};

for i = 1:length(models)
    r = results.(models{i});
    fprintf('%-20s %-14.6e %-14.6e %-10d %-9.1f%% %-7.0fs\n', ...
            model_names{i}, r.best_val_loss, r.test_rmse, ...
            r.n_learnable, r.sparsity*100, r.train_time);
end

% --- Key Ratios ---
fprintf('\n--- Key Performance Ratios ---\n');
fprintf('PIM / Unedited loss ratio:     %.4f\n', ...
        results.pim.best_val_loss / max(results.unedited.best_val_loss, 1e-15));
fprintf('TKM / Unedited loss ratio:     %.4f\n', ...
        results.tkm.best_val_loss / max(results.unedited.best_val_loss, 1e-15));
fprintf('PIM+TKM / Unedited loss ratio: %.4f\n', ...
        results.pim_tkm.best_val_loss / max(results.unedited.best_val_loss, 1e-15));

fprintf('\nPIM parameter reduction: %.1f%%\n', ...
        (1 - results.pim.n_learnable / max(results.unedited.n_learnable, 1)) * 100);

%% ========================================================================
% SECTION 6: VISUALIZATION
% ========================================================================
fprintf('\n[Step 5/5] Generating figures...\n');

% --- Figure 1: Training Curves ---
figure('Position', [100, 100, 1400, 500]);

subplot(1,2,1);
colors = {'r', 'b', [0.9, 0.7, 0], [0.5, 0, 0.5]};
color_matrix = [1 0 0; 0 0 1; 0.9 0.7 0; 0.5 0 0.5];  % Nx3 RGB for bar CData
hold on;
for i = 1:length(models)
    r = results.(models{i});
    tl_smooth = movmean(r.train_losses, 10);
    semilogy(tl_smooth, 'Color', colors{i}, 'LineWidth', 1.5, ...
             'DisplayName', model_names{i});
end
hold off;
xlabel('Epoch'); ylabel('Training MSE (log scale)');
title('Training Loss (smoothed)');
legend('Location', 'best'); grid on;

subplot(1,2,2);
hold on;
for i = 1:length(models)
    r = results.(models{i});
    semilogy(r.val_losses, 'Color', colors{i}, 'LineWidth', 1.5, ...
             'DisplayName', sprintf('%s (%.2e)', model_names{i}, r.best_val_loss));
end
hold off;
xlabel('Epoch'); ylabel('Validation MSE (log scale)');
title('Validation Loss');
legend('Location', 'best'); grid on;

sgtitle(sprintf('Figure 1: Training Performance - Lorenz-96 (N=%d, r=%d)', N, expansion_order), ...
       'FontWeight', 'bold', 'FontSize', 13);

saveas(gcf, 'fig/Fig1_TrainingCurves.png');
fprintf('  Fig1_TrainingCurves.png saved.\n');

% --- Figure 2: Test RMSE Bar Chart ---
figure('Position', [100, 100, 900, 500]);
rmse_vals = [results.unedited.test_rmse, results.pim.test_rmse, ...
             results.tkm.test_rmse, results.pim_tkm.test_rmse];
b = bar(rmse_vals);
b.FaceColor = 'flat';
b.CData = [1 0 0; 0 0 1; 0.9 0.7 0; 0.5 0 0.5];
set(gca, 'XTickLabel', model_names, 'YScale', 'log');
ylabel('Test RMSE (log scale)');
title('Prediction Accuracy Comparison');
grid on;

% Add value labels
for i = 1:length(rmse_vals)
    text(i, rmse_vals(i)*1.2, sprintf('%.4f', rmse_vals(i)), ...
         'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 9);
end

sgtitle('Figure 2: Test RMSE - All Models, Same Training', ...
       'FontWeight', 'bold', 'FontSize', 13);
saveas(gcf, 'fig/Fig2_TestRMSE.png');
fprintf('  Fig2_TestRMSE.png saved.\n');

% --- Figure 3: Model Complexity ---
figure('Position', [100, 100, 1400, 400]);

subplot(1,3,1);
total_params = zeros(1,4);
learn_params = zeros(1,4);
sparsity_pct = zeros(1,4);
for i = 1:length(models)
    r = results.(models{i});
    n_total = N * length(monomials_std);
    if contains(models{i}, 'tkm')
        n_total = N * n_mono_temp;
    end
    total_params(i) = n_total;
    learn_params(i) = r.n_learnable;
    sparsity_pct(i) = r.sparsity * 100;
end

subplot(1,3,1);
b = bar(total_params); b.FaceColor = 'flat'; b.CData = color_matrix;
set(gca, 'XTickLabel', model_names);
ylabel('Total Connections'); title('Total Weight Connections');
for i=1:4, text(i,total_params(i)+max(total_params)*0.02,num2str(total_params(i)),'HorizontalAlignment','center','FontSize',8); end

subplot(1,3,2);
b = bar(learn_params); b.FaceColor = 'flat'; b.CData = color_matrix;
set(gca, 'XTickLabel', model_names);
ylabel('Learnable Parameters'); title('Learnable Parameters');
for i=1:4, text(i,learn_params(i)+max(learn_params)*0.02,num2str(learn_params(i)),'HorizontalAlignment','center','FontSize',8); end

subplot(1,3,3);
b = bar(sparsity_pct); b.FaceColor = 'flat'; b.CData = color_matrix;
set(gca, 'XTickLabel', model_names);
ylabel('Sparsity (%)'); title('Weight Sparsity');
for i=1:4, text(i,sparsity_pct(i)+1,sprintf('%.1f%%',sparsity_pct(i)),'HorizontalAlignment','center','FontSize',8); end

sgtitle('Figure 3: Model Complexity Comparison', 'FontWeight', 'bold', 'FontSize', 13);
saveas(gcf, 'fig/Fig3_Complexity.png');
fprintf('  Fig3_Complexity.png saved.\n');

% --- Figure 4: Weight Matrix Visualization ---
figure('Position', [100, 100, 1800, 400]);

W_eff_u = abs(results.unedited.W);  % Unedited: A_val=0, A_unc=1
W_eff_p = abs(A_value_pim + A_uncertain_pim .* results.pim.W);
W_eff_t = abs(A_uncertain_tkm .* results.tkm.W);
W_eff_pt = abs(A_value_pt + A_uncertain_pt .* results.pim_tkm.W);

plot_idx = {W_eff_u, W_eff_p, W_eff_t, W_eff_pt};
plot_titles = {'(a) Unedited PhNN', '(b) PIM-Edited', '(c) TKM-Edited', '(d) PIM+TKM'};

for i = 1:4
    subplot(1,4,i);
    W_show = plot_idx{i};
    n_show = min(200, size(W_show, 2));
    imagesc(W_show(:, 1:n_show));
    colormap hot; colorbar;
    xlabel('Hidden Neuron Index'); ylabel('Output Dimension');
    title(plot_titles{i}, 'FontWeight', 'bold');
end

sgtitle('Figure 4: Effective Weight Matrix |W| (first 200 hidden neurons)', ...
       'FontWeight', 'bold', 'FontSize', 13);
saveas(gcf, 'fig/Fig4_WeightMatrix.png');
fprintf('  Fig4_WeightMatrix.png saved.\n');

% --- Figure 5: Loss Ratio Evolution ---
figure('Position', [100, 100, 800, 500]);

epochs = 1:length(results.unedited.val_losses);
val_u = results.unedited.val_losses;
hold on;
for i = 2:length(models)
    r = results.(models{i});
    ratio = r.val_losses ./ max(val_u(1:length(r.val_losses)), 1e-15);
    semilogy(1:length(ratio), ratio, 'Color', colors{i}, 'LineWidth', 1.8, ...
             'DisplayName', sprintf('%s / Unedited', model_names{i}));
end
plot([1, n_epochs], [1, 1], 'k--', 'LineWidth', 1.0, 'DisplayName', 'Equal to Unedited');
hold off;
xlabel('Epoch'); ylabel('Val Loss Ratio (Edited / Unedited)');
title('Relative Performance: Edited vs Unedited');
legend('Location', 'best'); grid on; xlim([1, n_epochs]);

sgtitle('Figure 5: Validation Loss Ratio Evolution', 'FontWeight', 'bold', 'FontSize', 13);
saveas(gcf, 'fig/Fig5_LossRatio.png');
fprintf('  Fig5_LossRatio.png saved.\n');

%% ========================================================================
% FINAL SUMMARY
% ========================================================================
fprintf('\n=============================================================\n');
fprintf('EXPERIMENT COMPLETE\n');
fprintf('  All figures saved to current directory.\n');
fprintf('=============================================================\n');

%% ========================================================================
% HELPER FUNCTIONS
% ========================================================================

function [X_temp, Y_temp] = build_temporal_data(X, Y, N, K)
% Build temporal input [x(k), x(k-1), ..., x(k-K+1)] -> x(k+1)
    n_samples = size(X, 1) - K + 1;
    X_temp = zeros(n_samples, N*K, 'single');
    for k = 1:K
        X_temp(:, (k-1)*N+1 : k*N) = X(k:n_samples+k-1, :);
    end
    Y_temp = Y(K:end, :);
end

function [A_value_pt, A_uncertain_pt] = build_pim_tkm_combined(N, dt, K, monomials_temp, A_uncertain_tkm)
% Build PIM+TKM combined mask for temporal input.
% Applies PIM within each temporal block, then TKM on top.

    n_monomials = length(monomials_temp);
    A_value_pt = zeros(N, n_monomials, 'single');
    A_uncertain_pim_temporal = zeros(N, n_monomials, 'single');

    for h = 1:n_monomials
        indices_h = monomials_temp{h};

        % Check if this monomial is entirely within one time block
        for k = 1:K
            block_start = (k-1) * N + 1;
            block_end = k * N;

            idx_in_block = indices_h(indices_h >= block_start & indices_h <= block_end);

            if length(idx_in_block) == length(indices_h)
                % All variables in this monomial belong to block k
                % Shift to 1-based indices within the block
                local_idx = idx_in_block - (k-1) * N;

                % Apply Lorenz-96 PIM within this block
                for i = 1:N
                    im2 = mod(i-3, N)+1; im1 = mod(i-2, N)+1;
                    ip1 = mod(i, N)+1;
                    relevant_set = [im2, im1, i, ip1];

                    if all(ismember(local_idx, relevant_set))
                        A_uncertain_pim_temporal(i, h) = 1;
                    end

                    % Known coefficient for x_i self-term
                    if length(local_idx) == 1 && local_idx(1) == i
                        A_value_pt(i, h) = single(1.0 - dt);
                    end
                end
                break;
            end
        end
    end

    % Apply TKM on top (zero out cross-temporal monomials)
    A_uncertain_pt = A_uncertain_pim_temporal .* A_uncertain_tkm;
end
