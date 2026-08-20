function [results, models] = experiment_controlled()
%% EXPERIMENT_CONTROLLED  All PhNN models trained under IDENTICAL conditions.
%   Only difference = editing mask (none / PIM / TKM / PIM+TKM).
%
%   All models share:
%     - Same 80D temporal input [x(k), x(k-1)]
%     - Same Taylor expansion (r=2)
%     - Same training data, epochs, batch size, learning rate
%     - Same evaluation: single-step test RMSE
%
%   [results, models] = experiment_controlled()

% Shared parameters (ALL models use these exactly)
N = 40; dt = 0.01; F = 8.0; r = 2; K = 2;
N_TRAIN = 6000; N_VAL = 2000; N_TEST = 3000;
N_EPOCHS = 150; BATCH_SIZE = 256; LEARNING_RATE = 0.001; SEED = 42;

dim_input = N * K;  % 80D temporal input for ALL models

fprintf('%s\n', repmat('=', 1, 70));
fprintf('CONTROLLED EXPERIMENT: All models with IDENTICAL settings\n');
fprintf('  Input: %dD temporal [x(k), x(k-1)], r=%d, epochs=%d\n', dim_input, r, N_EPOCHS);
fprintf('%s\n', repmat('=', 1, 70));

% ---- Step 1: Data generation (shared) ----
fprintf('\n[1/5] Generating shared data...\n');
[train_traj, val_traj, test_traj] = generate_train_val_test_data(...
    N, dt, F, N_TRAIN, N_VAL, N_TEST, SEED);

[X_train, Y_train] = build_temporal_data(train_traj, N, K);
[X_val,   Y_val]   = build_temporal_data(val_traj,   N, K);
[X_test,  Y_test]  = build_temporal_data(test_traj,  N, K);

fprintf('  Train: %dx%d, Val: %dx%d, Test: %dx%d\n', ...
    size(X_train,1), size(X_train,2), size(X_val,1), size(X_val,2), ...
    size(X_test,1), size(X_test,2));

% ---- Step 2: Taylor expansion and editing masks ----
fprintf('\n[2/5] Building shared Taylor expansion and editing masks...\n');
monomials = generate_monomial_indices(dim_input, r);
n_mono = length(monomials);
fprintf('  Monomials: %d (all models share this)\n', n_mono);

% --- PIM mask for temporal input ---
A_value_pim = zeros(N, n_mono, 'single');
A_unc_pim = zeros(N, n_mono, 'single');

% build_temporal_data stacks [x(k-1), x(k)] -> x(k+1) (oldest first, latest
% last). The latest state x(k) is therefore the LAST temporal block (index
% K-1); a first-order Markov system depends only on this block.
latest_block = K - 1;

for h = 1:n_mono
    midx = monomials{h};
    blocks = unique(floor((midx - 1) / N));

    % Only keep monomials purely from the latest x(k) block
    if all(blocks == latest_block)
        midx_local = midx - latest_block * N;  % map back to 1..N state indices
        for i = 1:N
            i_m2 = mod(i-3, N) + 1; i_m1 = mod(i-2, N) + 1;
            i_p1 = mod(i, N) + 1;
            relevant = [i_m2, i_m1, i, i_p1];

            if all(ismember(midx_local, relevant))
                A_unc_pim(i, h) = 1;  % learnable
            end
            if length(midx_local) == 1 && midx_local(1) == i
                A_value_pim(i, h) = 1.0 - dt;  % known self-term
            end
        end
    end
end

% --- TKM mask ---
[A_unc_tkm, ~] = build_lorenz96_tkm(N, monomials, K);

% --- PIM+TKM combined mask ---
A_unc_combined = A_unc_pim .* A_unc_tkm;

fprintf('  PIM sparsity:       %.1f%%\n', (1-mean(A_unc_pim(:)))*100);
fprintf('  TKM sparsity:       %.1f%%\n', (1-mean(A_unc_tkm(:)))*100);
fprintf('  PIM+TKM sparsity:   %.1f%%\n', (1-mean(A_unc_combined(:)))*100);

% ---- Step 3: Train all 4 models with IDENTICAL settings ----
fprintf('\n[3/5] Training all 4 models (each: %d epochs, batch=%d, lr=%.3f)...\n', ...
    N_EPOCHS, BATCH_SIZE, LEARNING_RATE);

results = struct();
models = struct();

configs = {
    'unedited', 'Unedited PhNN', [], [];
    'pim',      'PIM-Edited PhNN', A_value_pim, A_unc_pim;
    'tkm',      'TKM-Edited PhNN', [], A_unc_tkm;
    'pim_tkm',  'PIM+TKM Edited PhNN', A_value_pim, A_unc_combined;
};

for cfg = 1:size(configs, 1)
    name = configs{cfg, 1}; label = configs{cfg, 2};
    A_val = configs{cfg, 3}; A_unc = configs{cfg, 4};

    if isempty(A_val), A_val = zeros(N, n_mono, 'single'); end
    if isempty(A_unc), A_unc = ones(N, n_mono, 'single'); end

    fprintf('\n  --- %s ---\n', label);
    model = PhNNModel(dim_input, N, monomials, A_val, A_unc);
    model.summary();

    t0 = tic;
    [tl, vl, bv] = model.train(X_train, Y_train, X_val, Y_val, ...
        LEARNING_RATE, N_EPOCHS, BATCH_SIZE, N_EPOCHS+1);  % No early stopping
    dtime = toc(t0);

    % Single-step test RMSE (SAME metric for all)
    test_pred = model.forward(X_test);
    test_rmse = sqrt(mean((test_pred(:) - Y_test(:)).^2));

    results.(name) = struct('train_losses', tl, 'val_losses', vl, ...
        'best_val_loss', bv, 'test_rmse', test_rmse, ...
        'n_total', model.n_total, 'n_learnable', model.n_learnable, ...
        'sparsity', model.sparsity, 'train_time', dtime);
    models.(name) = model;

    fprintf('  Epoch %d: train_loss=%.6e, val_loss=%.6e\n', N_EPOCHS, tl(end), vl(end));
    fprintf('  Test RMSE: %.6e, Time: %.0fs\n', test_rmse, dtime);
end

% ---- Step 4: Results summary ----
fprintf('\n%s\n', repmat('=', 1, 90));
fprintf('[4/5] RESULTS -- All models: SAME data, SAME epochs, SAME batch, SAME lr\n');
fprintf('%s\n', repmat('=', 1, 90));

nms = {'unedited', 'pim', 'tkm', 'pim_tkm'};
dns = {'Unedited PhNN', 'PIM-Edited PhNN', 'TKM-Edited PhNN', 'PIM+TKM Edited PhNN'};

fprintf('\n%-20s %-10s %-10s %-14s %-14s %-14s %-8s\n', ...
    'Model', 'Params', 'Sparsity', 'Train Loss', 'Val Loss', 'Test RMSE', 'Time');
fprintf('%s\n', repmat('-', 1, 90));
for i = 1:4
    nm = nms{i}; r = results.(nm);
    fprintf('%-20s %-10d %-9.1f%% %-14.6e %-14.6e %-14.6e %-7.0fs\n', ...
        dns{i}, r.n_learnable, r.sparsity*100, ...
        r.train_losses(end), r.val_losses(end), r.test_rmse, r.train_time);
end

% Ratios
r_u = results.unedited; r_p = results.pim; r_t = results.tkm; r_pt = results.pim_tkm;
fprintf('\n  PIM / Unedited:     loss ratio = %.4f\n', r_p.val_losses(end) / max(r_u.val_losses(end), 1e-15));
fprintf('  TKM / Unedited:     loss ratio = %.4f\n', r_t.val_losses(end) / max(r_u.val_losses(end), 1e-15));
fprintf('  PIM+TKM / Unedited: loss ratio = %.4f\n', r_pt.val_losses(end) / max(r_u.val_losses(end), 1e-15));

% ---- Step 5: Generate 6 MATLAB-style figures ----
fprintf('\n[5/5] Generating MATLAB-style figures...\n');

cols = [0.85 0.33 0.10; 0 0.45 0.74; 0.93 0.69 0.13; 0.49 0.18 0.56];
epochs_axis = 1:N_EPOCHS;

% Figure 1: Training & Validation
figure('Position', [100, 100, 1100, 480]);
subplot(1,2,1); hold on;
for i = 1:4
    r = results.(nms{i});
    semilogy(epochs_axis, r.train_losses, 'Color', cols(i,:), 'LineWidth', 1.2, 'DisplayName', dns{i});
end
xlabel('Epoch'); ylabel('MSE Loss'); title('Training Loss');
xlim([1, N_EPOCHS]); legend('Location', 'best'); grid on;

subplot(1,2,2); hold on;
for i = 1:4
    r = results.(nms{i});
    semilogy(epochs_axis, r.val_losses, 'Color', cols(i,:), 'LineWidth', 1.8, ...
        'DisplayName', sprintf('%s (%.2e)', dns{i}, r.val_losses(end)));
end
xlabel('Epoch'); ylabel('MSE Loss'); title('Validation Loss');
xlim([1, N_EPOCHS]); legend('Location', 'best'); grid on;
sgtitle('Figure 1: Training Curves (150 epochs, identical settings)', 'FontWeight', 'bold', 'FontSize', 13);
saveas(gcf, 'fig/Fig1_TrainingCurves.png');
close;
fprintf('  Fig1 saved.\n');

% Figure 2: Loss ratio relative to Unedited
figure('Position', [100, 100, 750, 500]);
ref_loss = results.unedited.val_losses;
hold on;
for i = 2:4
    r = results.(nms{i});
    ratio = r.val_losses ./ max(ref_loss, 1e-15);
    semilogy(epochs_axis, ratio, 'Color', cols(i,:), 'LineWidth', 1.8, ...
        'DisplayName', sprintf('%s / Unedited', dns{i}));
end
yline(1.0, 'k--', 'LineWidth', 1.0);
xlabel('Epoch'); ylabel('Val Loss Ratio'); title('Relative Performance');
xlim([1, N_EPOCHS]); legend('Location', 'best'); grid on;
sgtitle('Figure 2: Validation Loss Ratio -- Edited / Unedited', 'FontWeight', 'bold', 'FontSize', 13);
saveas(gcf, 'fig/Fig2_LossRatio.png');
close;
fprintf('  Fig2 saved.\n');

% Figure 3: Bar chart -- Final metrics
figure('Position', [100, 100, 1200, 450]);
vals_rmse = zeros(1,4); vals_params = zeros(1,4); vals_time = zeros(1,4);
for i = 1:4
    r = results.(nms{i});
    vals_rmse(i) = r.test_rmse; vals_params(i) = r.n_learnable; vals_time(i) = r.train_time;
end

subplot(1,3,1); b = bar(vals_rmse); b.FaceColor = 'flat';
for i=1:4, b.CData(i,:)=cols(i,:); end
set(gca, 'XTickLabel', dns, 'YScale', 'log');
title('(a) Single-Step Test RMSE');
for i=1:4, text(i, vals_rmse(i)*1.08, sprintf('%.4f', vals_rmse(i)), 'HorizontalAlignment', 'center', 'FontSize', 9); end

subplot(1,3,2); b = bar(vals_params); b.FaceColor = 'flat';
for i=1:4, b.CData(i,:)=cols(i,:); end
set(gca, 'XTickLabel', dns); title('(b) Learnable Parameters');
for i=1:4, text(i, vals_params(i)+max(vals_params)*0.03, num2str(vals_params(i)), 'HorizontalAlignment', 'center', 'FontSize', 9); end

subplot(1,3,3); b = bar(vals_time); b.FaceColor = 'flat';
for i=1:4, b.CData(i,:)=cols(i,:); end
set(gca, 'XTickLabel', dns); title('(c) Training Time');
for i=1:4, text(i, vals_time(i)+max(vals_time)*0.03, sprintf('%.0fs', vals_time(i)), 'HorizontalAlignment', 'center', 'FontSize', 9); end
sgtitle('Figure 3: Performance Metrics', 'FontWeight', 'bold', 'FontSize', 13);
saveas(gcf, 'fig/Fig3_Metrics.png');
close;
fprintf('  Fig3 saved.\n');

% Figure 4: Sparsity comparison
figure('Position', [100, 100, 1200, 450]);
vals_tot = zeros(1,4); vals_learn = zeros(1,4); vals_sp = zeros(1,4);
for i = 1:4
    r = results.(nms{i});
    vals_tot(i) = r.n_total; vals_learn(i) = r.n_learnable; vals_sp(i) = r.sparsity*100;
end
subplot(1,3,1); b = bar(vals_tot); b.FaceColor = 'flat';
for i=1:4, b.CData(i,:)=cols(i,:); end
set(gca, 'XTickLabel', dns); title('(a) Total Connections');
for i=1:4, text(i, vals_tot(i)+max(vals_tot)*0.02, num2str(vals_tot(i)), 'HorizontalAlignment', 'center', 'FontSize', 8); end
subplot(1,3,2); b = bar(vals_learn); b.FaceColor = 'flat';
for i=1:4, b.CData(i,:)=cols(i,:); end
set(gca, 'XTickLabel', dns); title('(b) Learnable Parameters');
for i=1:4, text(i, vals_learn(i)+max(vals_learn)*0.02, num2str(vals_learn(i)), 'HorizontalAlignment', 'center', 'FontSize', 8); end
subplot(1,3,3); b = bar(vals_sp); b.FaceColor = 'flat';
for i=1:4, b.CData(i,:)=cols(i,:); end
set(gca, 'XTickLabel', dns); title('(c) Weight Sparsity (%)');
for i=1:4, text(i, vals_sp(i)+1.5, sprintf('%.1f%%', vals_sp(i)), 'HorizontalAlignment', 'center', 'FontSize', 8); end
sgtitle('Figure 4: Model Complexity', 'FontWeight', 'bold', 'FontSize', 13);
saveas(gcf, 'fig/Fig4_Complexity.png');
close;
fprintf('  Fig4 saved.\n');

% Figure 5: Weight matrices
figure('Position', [100, 100, 1600, 450]);
titles_w = {'(a) Unedited', '(b) PIM-Edited', '(c) TKM-Edited', '(d) PIM+TKM'};
for i = 1:4
    subplot(1,4,i);
    m = models.(nms{i});
    W_eff = abs(m.A_value + m.A_uncertain .* m.W_learn);
    n_show = min(200, size(W_eff,2));
    imagesc(W_eff(:, 1:n_show)); colormap('hot'); colorbar;
    xlabel('Hidden Neuron'); title(titles_w{i});
    if i==1, ylabel('Output Dim'); end
end
sgtitle('Figure 5: Effective Weight Matrix |W| (first 200 hidden neurons)', 'FontWeight', 'bold', 'FontSize', 13);
saveas(gcf, 'fig/Fig5_WeightMatrix.png');
close;
fprintf('  Fig5 saved.\n');

% Figure 6: Convergence speed
figure('Position', [100, 100, 800, 500]); hold on;
for i = 1:4
    r = results.(nms{i});
    final_vl = r.val_losses(end);
    threshold = final_vl * 1.1;
    idx_conv = find(r.val_losses <= threshold, 1, 'first');
    if isempty(idx_conv), idx_conv = N_EPOCHS; end
    semilogy(epochs_axis, r.val_losses, 'Color', cols(i,:), 'LineWidth', 1.8, 'DisplayName', dns{i});
    xline(idx_conv, ':', 'Color', cols(i,:), 'LineWidth', 1.2);
end
xlabel('Epoch'); ylabel('Validation MSE'); title('Convergence Speed');
xlim([1, N_EPOCHS]); legend('Location', 'best'); grid on;
sgtitle('Figure 6: Convergence Analysis', 'FontWeight', 'bold', 'FontSize', 13);
saveas(gcf, 'fig/Fig6_Convergence.png');
close;
fprintf('  Fig6 saved.\n');

fprintf('\n%s\n', repmat('=', 1, 70));
fprintf('CONTROLLED EXPERIMENT COMPLETE\n');
fprintf('  All models: %d epochs, batch=%d, lr=%.3f\n', N_EPOCHS, BATCH_SIZE, LEARNING_RATE);
fprintf('  All models: %dD input, %d monomials, %d train samples\n', dim_input, n_mono, N_TRAIN);
fprintf('%s\n', repmat('=', 1, 70));

end
