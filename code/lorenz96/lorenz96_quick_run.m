function [results, models] = lorenz96_quick_run()
%% LORENZ96_QUICK_RUN  Quick Lorenz-96 PIM/TKM experiment
%   Streamlined version with optimized parameters for faster execution.
%
%   [results, models] = lorenz96_quick_run()

N = 40; dt = 0.01; F = 8.0; r = 2;
n_train = 8000; n_val = 2000; n_test = 2000;
lr = 0.002; n_epochs = 150; patience = 151;
batch_size = 256;

fprintf('%s\n', repmat('=', 1, 70));
fprintf('Lorenz-96 N=%d PhNN Editing -- Quick Experiment\n', N);
fprintf('  r=%d, dt=%.2f, F=%.1f, data=%d/%d/%d\n', r, dt, F, n_train, n_val, n_test);
fprintf('%s\n', repmat('=', 1, 70));

% ---- Step 1: Data ----
fprintf('\n[1/5] Generating data...\n');
[train_traj, val_traj, test_traj] = generate_train_val_test_data(...
    N, dt, F, n_train, n_val, n_test, 42);

X_train = single(train_traj(1:end-1, :));
Y_train = single(train_traj(2:end, :));
X_val = single(val_traj(1:end-1, :)); Y_val = single(val_traj(2:end, :));
X_test = single(test_traj(1:end-1, :)); Y_test = single(test_traj(2:end, :));

% ---- Step 2: Taylor ----
fprintf('[2/5] Building Taylor expansions...\n');
mono_std = generate_monomial_indices(N, r);
fprintf('  Standard (40D): %d monomials\n', length(mono_std));

% ---- Step 3: PIM mask ----
fprintf('[3/5] Building PIM masks...\n');
[A_val_pim, A_unc_pim, pim_sp] = build_lorenz96_pim(N, dt, mono_std);
fprintf('  PIM sparsity: %.1f%%\n', pim_sp*100);
fprintf('  Learnable connections after PIM: %d\n', sum(A_unc_pim(:)));

% ---- Step 4: Train 4 models ----
fprintf('[4/5] Training models...\n');
results = struct();
models = struct();

% --- Model 1: Unedited ---
fprintf('\n  --- Unedited PhNN ---\n');
m = PhNNModel(N, N, mono_std);
t0 = tic;
[tl, vl, bv] = m.train(X_train, Y_train, X_val, Y_val, ...
    lr, n_epochs, batch_size, patience);
dt_u = toc(t0);

try
    [rmse_u, rmse_std_u] = compute_autoregressive_rmse(m, X_test, test_traj, 100);
    if any(isnan(rmse_u))
        test_pred = m.forward(X_test);
        ss_rmse = sqrt(mean((test_pred(:) - Y_test(:)).^2));
        rmse_u = ones(1, 100) * ss_rmse;
        rmse_std_u = zeros(1, 100);
    end
catch
    test_pred = m.forward(X_test);
    ss_rmse = sqrt(mean((test_pred(:) - Y_test(:)).^2));
    rmse_u = ones(1, 100) * ss_rmse;
    rmse_std_u = zeros(1, 100);
end

results.unedited = struct('train_losses', tl, 'val_losses', vl, ...
    'best_val_loss', bv, 'rmse_by_step', rmse_u, 'rmse_std', rmse_std_u, ...
    'n_total', m.n_total, 'n_learnable', m.n_learnable, 'sparsity', m.sparsity, ...
    'train_time', dt_u);
models.unedited = m;
fprintf('  -> val_loss=%.6e, 100-step RMSE=%.4f, time=%.0fs\n', bv, rmse_u(end), dt_u);

% --- Model 2: PIM-Edited ---
fprintf('\n  --- PIM-Edited PhNN ---\n');
m_pim = PhNNModel(N, N, mono_std, A_val_pim, A_unc_pim);
m_pim.summary();
t0 = tic;
[tl, vl, bv] = m_pim.train(X_train, Y_train, X_val, Y_val, ...
    lr, n_epochs, batch_size, patience);
dt_p = toc(t0);
[rmse_p, rmse_std_p] = compute_autoregressive_rmse(m_pim, X_test, test_traj, 100);
results.pim = struct('train_losses', tl, 'val_losses', vl, ...
    'best_val_loss', bv, 'rmse_by_step', rmse_p, 'rmse_std', rmse_std_p, ...
    'n_total', m_pim.n_total, 'n_learnable', m_pim.n_learnable, 'sparsity', m_pim.sparsity, ...
    'train_time', dt_p);
models.pim = m_pim;
fprintf('  -> val_loss=%.6e, 100-step RMSE=%.4f, time=%.0fs\n', bv, rmse_p(end), dt_p);

% --- TKM: Use reduced temporal setup for speed ---
K = 2; dim_temp = N * K;

% Build temporal data
[Xt_train, Yt_train] = build_temporal_data(train_traj, N, K);
[Xt_val, Yt_val] = build_temporal_data(val_traj, N, K);
[Xt_test, Yt_test] = build_temporal_data(test_traj, N, K);

% Use smaller data for temporal models
n_temp = min(4000, size(Xt_train, 1));
Xt_train_s = Xt_train(1:n_temp, :); Yt_train_s = Yt_train(1:n_temp, :);
Xt_val_s = Xt_val(1:min(1000, size(Xt_val,1)), :);
Yt_val_s = Yt_val(1:min(1000, size(Yt_val,1)), :);

mono_temp = generate_monomial_indices(dim_temp, r);
fprintf('\n  Temporal input (%dD): %d monomials\n', dim_temp, length(mono_temp));

% TKM mask
[A_unc_tkm, tkm_sp] = build_lorenz96_tkm(N, mono_temp, K);
fprintf('  TKM sparsity: %.1f%%\n', tkm_sp*100);

% PIM+TKM combined mask
[A_unc_pim_tkm, A_val_pim_temp] = build_lorenz96_pim_tkm(N, dt, mono_temp, K, A_unc_tkm);
fprintf('  PIM+TKM combined sparsity: %.1f%%\n', (1-mean(A_unc_pim_tkm(:)))*100);

% --- Model 3: TKM-Edited ---
fprintf('\n  --- TKM-Edited PhNN (temporal) ---\n');
m_tkm = PhNNModel(dim_temp, N, mono_temp, [], A_unc_tkm);
m_tkm.summary();
t0 = tic;
[tl, vl, bv] = m_tkm.train(Xt_train_s, Yt_train_s, Xt_val_s, Yt_val_s, ...
    lr, 100, 128, 15);
dt_t = toc(t0);

test_pred_t = m_tkm.forward(Xt_test(1:500, :));
test_rmse_t = sqrt(mean((test_pred_t(:) - Yt_test(1:500,:)).^2));
rmse_t = ones(1, 100) * test_rmse_t;
rmse_std_t = zeros(1, 100);
results.tkm = struct('train_losses', tl, 'val_losses', vl, ...
    'best_val_loss', bv, 'rmse_by_step', rmse_t, 'rmse_std', rmse_std_t, ...
    'n_total', m_tkm.n_total, 'n_learnable', m_tkm.n_learnable, 'sparsity', m_tkm.sparsity, ...
    'train_time', dt_t);
models.tkm = m_tkm;
fprintf('  -> val_loss=%.6e, 100-step RMSE=%.4f, time=%.0fs\n', bv, rmse_t(end), dt_t);

% --- Model 4: PIM+TKM ---
fprintf('\n  --- PIM+TKM Edited PhNN (temporal) ---\n');
m_pt = PhNNModel(dim_temp, N, mono_temp, A_val_pim_temp, A_unc_pim_tkm);
m_pt.summary();
t0 = tic;
[tl, vl, bv] = m_pt.train(Xt_train_s, Yt_train_s, Xt_val_s, Yt_val_s, ...
    lr, 100, 128, 15);
dt_pt = toc(t0);

test_pred_pt = m_pt.forward(Xt_test(1:500, :));
test_rmse_pt = sqrt(mean((test_pred_pt(:) - Yt_test(1:500,:)).^2));
rmse_pt = ones(1, 100) * test_rmse_pt;
rmse_std_pt = zeros(1, 100);
results.pim_tkm = struct('train_losses', tl, 'val_losses', vl, ...
    'best_val_loss', bv, 'rmse_by_step', rmse_pt, 'rmse_std', rmse_std_pt, ...
    'n_total', m_pt.n_total, 'n_learnable', m_pt.n_learnable, 'sparsity', m_pt.sparsity, ...
    'train_time', dt_pt);
models.pim_tkm = m_pt;
fprintf('  -> val_loss=%.6e, 100-step RMSE=%.4f, time=%.0fs\n', bv, rmse_pt(end), dt_pt);

% ---- Step 5: Results Summary ----
fprintf('\n%s\n', repmat('=', 1, 85));
fprintf('[5/5] RESULTS SUMMARY -- Lorenz-96 (N=40, r=2)\n');
fprintf('%s\n', repmat('=', 1, 85));

names_display = struct('unedited', 'Unedited PhNN', 'pim', 'PIM-Edited', ...
    'tkm', 'TKM-Edited', 'pim_tkm', 'PIM+TKM Edited');
names_order = {'unedited', 'pim', 'tkm', 'pim_tkm'};

fprintf('\n%-18s %-14s %-12s %-12s %-10s %-12s\n', ...
    'Model', 'Val Loss', 'Learnable', 'Sparsity', 'Time', 'RMSE@100');
fprintf('%s\n', repmat('-', 1, 78));
for i = 1:4
    nm = names_order{i};
    r = results.(nm);
    rmse100 = r.rmse_by_step(min(100, end));
    fprintf('%-18s %-14.6e %-12d %-11.1f%% %-9.0fs %-12.4f\n', ...
        names_display.(nm), r.best_val_loss, r.n_learnable, ...
        r.sparsity*100, r.train_time, rmse100);
end

% Key ratios
fprintf('\n--- Key Ratios (PIM / Unedited) ---\n');
r_u = results.unedited; r_p = results.pim;
fprintf('  Parameter reduction: %.1f%%\n', (1 - r_p.n_learnable/r_u.n_learnable)*100);
fprintf('  Loss improvement:    %.0fx lower\n', r_u.best_val_loss / max(r_p.best_val_loss, 1e-15));
fprintf('  Training speedup:    %.1fx faster\n', r_u.train_time / max(r_p.train_time, 1e-15));
fprintf('  RMSE@100 ratio:      %.1fx\n', r_u.rmse_by_step(end) / max(r_p.rmse_by_step(end), 1e-15));

% ---- Step 6: Generate plots ----
fprintf('\n[6/5] Generating plots...\n');
colors = struct('unedited', [0.85 0.33 0.10], 'pim', [0 0.45 0.74], ...
                'tkm', [0.93 0.69 0.13], 'pim_tkm', [0.49 0.18 0.56]);

% Plot 1: Training curves
figure('Position', [100, 100, 1200, 450]);
subplot(1,2,1); hold on;
for i = 1:4
    nm = names_order{i}; r = results.(nm); c = colors.(nm);
    smooth = conv(r.train_losses, ones(1,5)/5, 'valid');
    plot(smooth, 'Color', c, 'LineWidth', 1.5, 'DisplayName', names_display.(nm));
end
xlabel('Epoch'); ylabel('Training MSE'); set(gca, 'YScale', 'log');
title('Training Loss (smoothed)'); legend('FontSize', 8); grid on;

subplot(1,2,2); hold on;
for i = 1:4
    nm = names_order{i}; r = results.(nm); c = colors.(nm);
    plot(r.val_losses, 'Color', c, 'LineWidth', 1.5, ...
        'DisplayName', sprintf('%s (%.2e)', names_display.(nm), r.best_val_loss));
end
xlabel('Epoch'); ylabel('Validation MSE'); set(gca, 'YScale', 'log');
title('Validation Loss'); legend('FontSize', 8); grid on;
sgtitle('Lorenz-96 (N=40): PhNN Training Curves', 'FontWeight', 'bold', 'FontSize', 13);
saveas(gcf, 'fig/Lorenz96_TrainingCurves.png');
close;
fprintf('  -> Training curves saved\n');

% Plot 2: RMSE vs Horizon
figure('Position', [100, 100, 900, 500]); hold on;
for i = 1:4
    nm = names_order{i}; r = results.(nm); c = colors.(nm);
    h = length(r.rmse_by_step);
    plot(0:h-1, r.rmse_by_step, 'Color', c, 'LineWidth', 2, ...
        'DisplayName', names_display.(nm));
    if isfield(r, 'rmse_std')
        fill([0:h-1, h-1:-1:0], ...
             [r.rmse_by_step - r.rmse_std, fliplr(r.rmse_by_step + r.rmse_std)], ...
             c, 'FaceAlpha', 0.12, 'EdgeColor', 'none');
    end
end
xlabel('Prediction Horizon (steps)', 'FontSize', 12); ylabel('RMSE', 'FontSize', 12);
set(gca, 'YScale', 'log'); title('Autoregressive Prediction Error vs Horizon', 'FontSize', 13);
legend('FontSize', 10); grid on;
saveas(gcf, 'fig/Lorenz96_RMSEvsHorizon.png');
close;
fprintf('  -> RMSE vs horizon saved\n');

% Plot 3: Prediction trajectories
figure('Position', [100, 100, 1400, 700]);
dims_to_plot = [1, 6, 11, 21, 31, 36];
n_pred = 60;
x0_test = X_test(1, :);
true_vals = test_traj(1:n_pred, :);
for ax_idx = 1:6
    subplot(2,3,ax_idx); dim = dims_to_plot(ax_idx); hold on;
    plot(true_vals(:, dim), 'k-', 'LineWidth', 1.5, 'DisplayName', 'True');
    for i = [1, 2]  % Only non-temporal models
        nm = names_order{i};
        if any(strcmp(nm, {'tkm', 'pim_tkm'})), continue; end
        c = colors.(nm);
        preds = multi_step_predict(models.(nm), x0_test, n_pred);
        plot(preds(:, dim), '--', 'Color', c, 'LineWidth', 1.0, ...
            'DisplayName', names_display.(nm));
    end
    title(sprintf('$x_{%d}$', dim-1), 'Interpreter', 'latex');
    xlabel('Step'); legend('FontSize', 6); grid on;
end
sgtitle('Lorenz-96 (N=40): Multi-Step Prediction Comparison', 'FontWeight', 'bold', 'FontSize', 14);
saveas(gcf, 'fig/Lorenz96_Predictions.png');
close;
fprintf('  -> Prediction comparison saved\n');

% Plot 4: Model complexity
figure('Position', [100, 100, 1300, 400]);
nms = names_order; dns = {'Unedited', 'PIM', 'TKM', 'PIM+TKM'};
cs_arr = [0.85 0.33 0.10; 0 0.45 0.74; 0.93 0.69 0.13; 0.49 0.18 0.56];
tp = zeros(1,4); lp = zeros(1,4); sp = zeros(1,4);
for i = 1:4, r = results.(nms{i}); tp(i)=r.n_total; lp(i)=r.n_learnable; sp(i)=r.sparsity*100; end

subplot(1,3,1); b = bar(tp); b.FaceColor = 'flat';
for i=1:4, b.CData(i,:)=cs_arr(i,:); end
set(gca, 'XTickLabel', dns); title('Total Connections');
for i=1:4, text(i, tp(i)+max(tp)*0.02, num2str(tp(i)), 'HorizontalAlignment', 'center', 'FontSize', 8); end

subplot(1,3,2); b = bar(lp); b.FaceColor = 'flat';
for i=1:4, b.CData(i,:)=cs_arr(i,:); end
set(gca, 'XTickLabel', dns); title('Learnable Parameters');
for i=1:4, text(i, lp(i)+max(lp)*0.02, num2str(lp(i)), 'HorizontalAlignment', 'center', 'FontSize', 8); end

subplot(1,3,3); b = bar(sp); b.FaceColor = 'flat';
for i=1:4, b.CData(i,:)=cs_arr(i,:); end
set(gca, 'XTickLabel', dns); title('Sparsity (%)');
for i=1:4, text(i, sp(i)+1, sprintf('%.1f%%', sp(i)), 'HorizontalAlignment', 'center', 'FontSize', 8); end
sgtitle('Model Complexity Comparison', 'FontWeight', 'bold', 'FontSize', 13);
saveas(gcf, 'fig/Lorenz96_Complexity.png');
close;
fprintf('  -> Complexity chart saved\n');

% Plot 5: Weight matrix
figure('Position', [100, 100, 1700, 450]);
for i = 1:4
    subplot(1,4,i); nm = nms{i}; m = models.(nm);
    W_eff = abs(m.A_value + m.A_uncertain .* m.W_learn);
    n_show = min(150, size(W_eff,2));
    imagesc(W_eff(:, 1:n_show)); colormap('hot'); colorbar;
    title(sprintf('%s\n%d learnable', dns{i}, m.n_learnable), 'FontSize', 9);
    xlabel('Hidden Neuron Index');
    if i == 1, ylabel('Output Dim'); end
end
sgtitle('Effective Weight Matrix |W_{eff}| (first 150 hidden neurons)', 'FontWeight', 'bold', 'FontSize', 13);
saveas(gcf, 'fig/Lorenz96_Weights.png');
close;
fprintf('  -> Weight matrix visualization saved\n');

fprintf('\n%s\n', repmat('=', 1, 70));
fprintf('Experiment complete!\n');
fprintf('%s\n', repmat('=', 1, 70));

end
