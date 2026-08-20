function [results, models] = experiment_sindy()
%% EXPERIMENT_SINDY  SINDy vs PhNN comparison on Lorenz-96
%   All methods use IDENTICAL Taylor library, data, and evaluation.
%
%   [results, models] = experiment_sindy()

N = 40; dt = 0.01; F = 8.0; r = 2; K = 2;
N_TRAIN = 6000; N_VAL = 2000; N_TEST = 3000;
EPOCHS = 150; BATCH = 256; LR = 0.001; SEED = 42;

fprintf('%s\n', repmat('=', 1, 70));
fprintf('SINDy vs PhNN COMPARISON -- Lorenz-96 (N=40)\n');
fprintf('  All methods: SAME Taylor library, SAME data, SAME evaluation\n');
fprintf('%s\n', repmat('=', 1, 70));

% ---- Data ----
fprintf('\n[1/6] Generating data...\n');
[train_traj, val_traj, test_traj] = generate_train_val_test_data(...
    N, dt, F, N_TRAIN, N_VAL, N_TEST, SEED);

% Standard (40D): x(k) -> x(k+1)
X_tr = single(train_traj(1:end-1, :)); Y_tr = single(train_traj(2:end, :));
X_va = single(val_traj(1:end-1, :));   Y_va = single(val_traj(2:end, :));
X_te = single(test_traj(1:end-1, :));  Y_te = single(test_traj(2:end, :));

% Temporal (80D): [x(k), x(k-1)] -> x(k+1)
[Xt_tr, Yt_tr] = build_temporal_data(train_traj, N, K);
[Xt_va, Yt_va] = build_temporal_data(val_traj,   N, K);
[Xt_te, Yt_te] = build_temporal_data(test_traj,  N, K);

% ---- Taylor libraries ----
fprintf('\n[2/6] Building Taylor libraries...\n');
mono_std = generate_monomial_indices(N, r);
mono_temp = generate_monomial_indices(N*K, r);
fprintf('  Standard (40D): %d monomials\n', length(mono_std));
fprintf('  Temporal (80D): %d monomials\n', length(mono_temp));

fprintf('  Building library matrix (standard)...\n');
Theta_std_tr = taylor_expand(X_tr, mono_std);
Theta_std_va = taylor_expand(X_va, mono_std);
Theta_std_te = taylor_expand(X_te, mono_std);

fprintf('  Building library matrix (temporal)...\n');
Theta_tmp_tr = taylor_expand(Xt_tr, mono_temp);
Theta_tmp_va = taylor_expand(Xt_va, mono_temp);
Theta_tmp_te = taylor_expand(Xt_te, mono_temp);

% PIM mask
[A_val_pim, A_unc_pim, ~] = build_lorenz96_pim(N, dt, mono_std);

% ============================================================
% Method 1: SINDy on standard input
% ============================================================
fprintf('\n[3/6] SINDy (standard 40D input) -- Ridge + Threshold...\n');
t0 = tic;
[best_th, Xi_sindy, best_rmse_s, cv_results] = sindy_threshold_ridge(...
    Theta_std_tr, Y_tr, Theta_std_va, Y_va, logspace(-3, 0.5, 20), 0.01);
dt_s = toc(t0);

n_nz_sindy = nnz(Xi_sindy);
Y_pred_sindy = Theta_std_te * Xi_sindy;
rmse_sindy = sqrt(mean((Y_pred_sindy(:) - Y_te(:)).^2));
sp_sindy = 1.0 - n_nz_sindy / numel(Xi_sindy);

fprintf('  Best threshold: %.4f, nonzero coeffs: %d/%d\n', best_th, n_nz_sindy, numel(Xi_sindy));
fprintf('  Sparsity: %.1f%%, Test RMSE: %.6e, Time: %.0fs\n', sp_sindy*100, rmse_sindy, dt_s);

% ============================================================
% Method 2: SINDy on temporal input
% ============================================================
fprintf('\n[4/6] SINDy (temporal 80D input) -- Ridge + Threshold...\n');
n_cv = min([3000, size(Theta_tmp_tr, 1), size(Theta_tmp_va, 1)]);
t0 = tic;
[best_th_t, Xi_sindy_t, best_rmse_st, ~] = sindy_threshold_ridge(...
    Theta_tmp_tr(1:n_cv,:), Yt_tr(1:n_cv,:), ...
    Theta_tmp_va(1:n_cv,:), Yt_va(1:n_cv,:), ...
    logspace(-2, 1, 10), 0.1);
dt_st = toc(t0);

n_nz_sindy_t = nnz(Xi_sindy_t);
Y_pred_sindy_t = Theta_tmp_te * Xi_sindy_t;
rmse_sindy_t = sqrt(mean((Y_pred_sindy_t(:) - Yt_te(:)).^2));
sp_sindy_t = 1.0 - n_nz_sindy_t / numel(Xi_sindy_t);

fprintf('  Best threshold: %.4f, nonzero coeffs: %d/%d\n', best_th_t, n_nz_sindy_t, numel(Xi_sindy_t));
fprintf('  Sparsity: %.1f%%, Test RMSE: %.6e, Time: %.0fs\n', sp_sindy_t*100, rmse_sindy_t, dt_st);

% ============================================================
% Method 3: PhNN models (standard 40D input)
% ============================================================
fprintf('\n[5/6] PhNN models (standard 40D input)...\n');
results = struct();

% PhNN Unedited (40D)
fprintf('  Training PhNN Unedited...\n');
m_u = PhNNModel(N, N, mono_std);
t0 = tic;
[tl_u, vl_u, ~] = m_u.train(X_tr, Y_tr, X_va, Y_va, LR, EPOCHS, BATCH, EPOCHS+1);
dt_u = toc(t0);
yp = m_u.forward(X_te); rmse_u = sqrt(mean((yp(:) - Y_te(:)).^2));
results.unedited = struct('tl', tl_u, 'vl', vl_u, 'rmse', rmse_u, ...
    'params', m_u.n_learnable, 'sparsity', m_u.sparsity, 'time', dt_u);
fprintf('  Val loss=%.4e, Test RMSE=%.4e, Params=%d, Time=%.0fs\n', ...
    vl_u(end), rmse_u, m_u.n_learnable, dt_u);

% PhNN PIM (40D)
fprintf('  Training PhNN + PIM...\n');
m_p = PhNNModel(N, N, mono_std, A_val_pim, A_unc_pim);
t0 = tic;
[tl_p, vl_p, ~] = m_p.train(X_tr, Y_tr, X_va, Y_va, LR, EPOCHS, BATCH, EPOCHS+1);
dt_p = toc(t0);
yp = m_p.forward(X_te); rmse_p = sqrt(mean((yp(:) - Y_te(:)).^2));
results.pim = struct('tl', tl_p, 'vl', vl_p, 'rmse', rmse_p, ...
    'params', m_p.n_learnable, 'sparsity', m_p.sparsity, 'time', dt_p);
fprintf('  Val loss=%.4e, Test RMSE=%.4e, Params=%d, Time=%.0fs\n', ...
    vl_p(end), rmse_p, m_p.n_learnable, dt_p);

models = struct('unedited', m_u, 'pim', m_p);

% ============================================================
% Summary table
% ============================================================
fprintf('\n%s\n', repmat('=', 1, 100));
fprintf('[6/6] COMPLETE COMPARISON -- SINDy vs PhNN on Lorenz-96 (N=40)\n');
fprintf('%s\n', repmat('=', 1, 100));

all_results = struct(...
    'sindy',          struct('rmse', rmse_sindy,    'params', n_nz_sindy,    'sparsity', sp_sindy,    'time', dt_s,  'vl', best_rmse_s), ...
    'sindy_temporal', struct('rmse', rmse_sindy_t,  'params', n_nz_sindy_t,  'sparsity', sp_sindy_t,  'time', dt_st, 'vl', best_rmse_st), ...
    'unedited',       struct('rmse', rmse_u,        'params', m_u.n_learnable, 'sparsity', m_u.sparsity, 'time', dt_u, 'vl', vl_u(end)), ...
    'pim',            struct('rmse', rmse_p,        'params', m_p.n_learnable, 'sparsity', m_p.sparsity, 'time', dt_p, 'vl', vl_p(end)));

labels_list = struct('sindy', 'SINDy (auto sparse)', 'sindy_temporal', 'SINDy (temporal)', ...
    'unedited', 'PhNN Unedited', 'pim', 'PhNN + PIM');

fprintf('\n%-28s %-10s %-12s %-12s %-14s %-14s %-8s\n', ...
    'Method', 'Input', 'Learnable', 'Sparsity', 'Val Loss', 'Test RMSE', 'Time');
fprintf('%s\n', repmat('-', 1, 100));
names_all = {'sindy', 'sindy_temporal', 'unedited', 'pim'};
for i = 1:4
    nm = names_all{i}; r = all_results.(nm);
    input_label = '40D';
    if contains(nm, 'temporal'), input_label = '80D'; end
    fprintf('%-28s %-10s %-12d %-11.1f%% %-14.6e %-14.6e %-7.0fs\n', ...
        labels_list.(nm), input_label, r.params, r.sparsity*100, r.vl, r.rmse, r.time);
end

fprintf('\n  SINDy auto-sparsity:         %.1f%% (%d terms selected)\n', sp_sindy*100, n_nz_sindy);
fprintf('  PIM physics-guided sparsity:  %.1f%% (%d params, prior knowledge)\n', m_p.sparsity*100, m_p.n_learnable);
fprintf('  SINDy Test RMSE:              %.6e\n', rmse_sindy);
fprintf('  PhNN+PIM Test RMSE:           %.6e\n', rmse_p);

% Structure Recovery Analysis
fprintf('\n--- Structure Recovery Analysis ---\n');
n_spurious = 0; n_missed = 0; n_total_true = 0;
for i = 1:N
    i_m2 = mod(i-3, N)+1; i_m1 = mod(i-2, N)+1; i_p1 = mod(i, N)+1;
    relevant = [i_m2, i_m1, i, i_p1];
    true_terms_i = [];
    for h = 1:length(mono_std)
        if all(ismember(mono_std{h}, relevant))
            true_terms_i(end+1) = h; %#ok<AGROW>
        end
    end
    n_total_true = n_total_true + length(true_terms_i);

    for h = true_terms_i
        if Xi_sindy(h, i) == 0, n_missed = n_missed + 1; end
    end

    for h = 1:length(mono_std)
        if Xi_sindy(h, i) ~= 0 && ~all(ismember(mono_std{h}, relevant))
            n_spurious = n_spurious + 1;
        end
    end
end

fprintf('  True relevant terms (per physics): %d\n', n_total_true);
fprintf('  SINDy selected terms (non-zero):   %d\n', n_nz_sindy);
fprintf('  Spurious selections:               %d\n', n_spurious);
fprintf('  Missed true terms:                 %d\n', n_missed);
if n_spurious + n_missed == 0
    fprintf('  PERFECT structure recovery!\n');
elseif n_spurious < 10 && n_missed < 10
    fprintf('  APPROXIMATE structure recovery (few errors)\n');
else
    fprintf('  SINDy has %d spurious + %d missed terms\n', n_spurious, n_missed);
end

% ============================================================
% FIGURES
% ============================================================
fprintf('\nGenerating figures...\n');

% Figure S1: Coefficients
figure('Position', [100, 100, 1400, 480]);
subplot(1,3,1);
imagesc(abs(Xi_sindy)); colormap('hot'); colorbar;
xlabel('Output Dim'); ylabel('Monomial');
title(sprintf('(a) SINDy |Xi| (%d nonzero, %.1f%% sparse)', n_nz_sindy, sp_sindy*100));

subplot(1,3,2);
imagesc(A_unc_pim); colormap('parula'); colorbar;
xlabel('Output Dim'); ylabel('Monomial');
title(sprintf('(b) PIM Mask (%d learnable, %.1f%% sparse)', sum(A_unc_pim(:)), m_p.sparsity*100));

subplot(1,3,3);
Wu = abs(m_u.W_learn);
imagesc(Wu); colormap('hot'); colorbar;
xlabel('Output Dim'); ylabel('Monomial');
title(sprintf('(c) PhNN Unedited |W| (%d params)', m_u.n_learnable));
sgtitle('Fig S1: Coefficient Structure -- SINDy vs PIM vs Unedited', 'FontWeight', 'bold', 'FontSize', 13);
saveas(gcf, 'fig/FigS1_SINDy_Coefficients.png');
close;
fprintf('  FigS1 saved.\n');

% Figure S2: RMSE bar chart
figure('Position', [100, 100, 900, 500]);
names_plot = {'sindy', 'sindy_temporal', 'unedited', 'pim'};
colors_plot = [0.47 0.67 0.19; 0.64 0.08 0.18; 0.85 0.33 0.10; 0 0.45 0.74];
display_plot = {'SINDy (40D)', 'SINDy (80D)', 'PhNN Unedited', 'PhNN + PIM'};
rmses = zeros(1,4);
for i = 1:4, rmses(i) = all_results.(names_plot{i}).rmse; end
b = bar(rmses); b.FaceColor = 'flat';
for i = 1:4, b.CData(i,:) = colors_plot(i,:); end
set(gca, 'XTickLabel', display_plot, 'YScale', 'log');
for i = 1:4
    text(i, rmses(i)*1.3, sprintf('%.4f', rmses(i)), ...
        'HorizontalAlignment', 'center', 'FontSize', 10);
end
ylabel('Test RMSE (log)'); title('Prediction Accuracy Comparison');
sgtitle('Fig S2: SINDy vs PhNN -- Same Data, Same Library', 'FontWeight', 'bold', 'FontSize', 13);
saveas(gcf, 'fig/FigS2_RMSE_Comparison.png');
close;
fprintf('  FigS2 saved.\n');

% Figure S3: CV
figure('Position', [100, 100, 800, 500]);
ths = zeros(1, length(cv_results)); rms_cv = zeros(1, length(cv_results)); nzs = zeros(1, length(cv_results));
for i = 1:length(cv_results)
    ths(i) = cv_results{i}{1}; rms_cv(i) = cv_results{i}{2}; nzs(i) = cv_results{i}{3};
end
yyaxis left;
semilogx(ths, rms_cv, 'b-o', 'LineWidth', 1.5, 'MarkerSize', 5);
ylabel('Val RMSE', 'Color', 'b');
yyaxis right;
semilogx(ths, nzs, 'r-s', 'LineWidth', 1.5, 'MarkerSize', 5);
xline(best_th, 'k--', 'LineWidth', 1.5);
xlabel('Threshold'); ylabel('Nonzero Coeffs', 'Color', 'r');
title('SINDy Threshold Cross-Validation'); grid on;
sgtitle('Fig S3: SINDy Hyperparameter Selection', 'FontWeight', 'bold', 'FontSize', 13);
saveas(gcf, 'fig/FigS3_SINDy_CV.png');
close;
fprintf('  FigS3 saved.\n');

% Figure S4: Terms per output
figure('Position', [100, 100, 800, 500]);
nz_per = sum(abs(Xi_sindy) > 0, 1);
bar(0:N-1, nz_per, 'FaceColor', [0.47 0.67 0.19], 'EdgeColor', 'k');
hold on;
yline(14, 'k--', 'LineWidth', 1.5);
yline(mean(nz_per), '-', 'Color', [0.47 0.67 0.19], 'LineWidth', 1.5);
xlabel('Output Dim'); ylabel('# Selected Terms');
title('SINDy: Terms per Output');
legend({'Expected: 14', sprintf('SINDy mean: %.1f', mean(nz_per))});
grid on;
sgtitle('Fig S4: SINDy Sparsity Pattern', 'FontWeight', 'bold', 'FontSize', 13);
saveas(gcf, 'fig/FigS4_SINDy_PerOutput.png');
close;
fprintf('  FigS4 saved.\n');

% Figure S5: Coefficient recovery
figure('Position', [100, 100, 1300, 480]);
true_c = zeros(length(mono_std), 1);
sindy_c_avg = zeros(length(mono_std), 1);
for i = 1:N
    i_m1 = mod(i-2,N)+1; i_m2 = mod(i-3,N)+1; i_p1 = mod(i,N)+1;
    for h = 1:length(mono_std)
        midx = mono_std{h};
        if length(midx)==1 && midx(1)==i,        true_c(h)=1.0-dt;
        elseif length(midx)==2 && midx(1)==i_m1 && midx(2)==i_p1, true_c(h)=dt;
        elseif length(midx)==2 && midx(1)==i_m1 && midx(2)==i_m2, true_c(h)=-dt;
        end
    end
    sindy_c_avg = sindy_c_avg + abs(Xi_sindy(:, i));
end
sindy_c_avg = sindy_c_avg / N;
W_pim_avg = mean(abs(m_p.A_value + m_p.A_uncertain .* m_p.W_learn), 1)';

n_top = min(80, sum(true_c~=0)*3);
[~, idx_s] = sort(sindy_c_avg, 'descend');
[~, idx_p] = sort(W_pim_avg, 'descend');
idx_s = idx_s(1:n_top); idx_p = idx_p(1:n_top);
xr = 1:n_top; w = 0.35;

subplot(1,2,1);
bar(xr-w/2, abs(true_c(idx_s)), w, 'FaceColor', 'k', 'DisplayName', 'True (Euler)'); hold on;
bar(xr+w/2, sindy_c_avg(idx_s), w, 'FaceColor', [0.47 0.67 0.19], 'DisplayName', 'SINDy');
xlabel('Monomial Rank'); ylabel('|Coefficient|'); title('(a) SINDy vs True'); legend; grid on;

subplot(1,2,2);
bar(xr-w/2, abs(true_c(idx_p)), w, 'FaceColor', 'k', 'DisplayName', 'True (Euler)'); hold on;
bar(xr+w/2, W_pim_avg(idx_p), w, 'FaceColor', [0 0.45 0.74], 'DisplayName', 'PhNN+PIM');
xlabel('Monomial Rank'); ylabel('|Coefficient|'); title('(b) PhNN+PIM vs True'); legend; grid on;
sgtitle('Fig S5: Coefficient Recovery', 'FontWeight', 'bold', 'FontSize', 13);
saveas(gcf, 'fig/FigS5_CoefficientRecovery.png');
close;
fprintf('  FigS5 saved.\n');

fprintf('\nAll SINDy comparison figures saved to fig/.\n');
fprintf('%s\n', repmat('=', 1, 70));

% Save experiment + figure data for offline plotting
figdata = struct();
figdata.Xi_sindy = Xi_sindy;
figdata.n_nz_sindy = n_nz_sindy;
figdata.sp_sindy = sp_sindy;
figdata.A_unc_pim = A_unc_pim;
figdata.all_results = all_results;
figdata.cv_results = cv_results;
figdata.best_th = best_th;
figdata.mono_std = mono_std;
figdata.N = N;
figdata.dt = dt;
out_dir = fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))), 'results');  % <repo>/results
if ~exist(out_dir, 'dir'), mkdir(out_dir); end
save(fullfile(out_dir, 'sindy_results.mat'), 'results', 'models', 'figdata');
fprintf('  Results saved to results/sindy_results.mat\n');

end

%% ========================================================================
%  SINDy: Sequential Thresholded Least Squares
%  ========================================================================

function [best_th, best_Xi, best_rmse, cv_results] = sindy_threshold_ridge(...
    Theta_train, Y_train, Theta_val, Y_val, thresholds, alpha)
% Pure NumPy SINDy: Ridge regression + hard thresholding + re-solve.
    if nargin < 6, alpha = 0.01; end

    [~, n_features] = size(Theta_train);
    n_targets = size(Y_train, 2);
    I = eye(n_features);

    % Full ridge regression
    G = Theta_train' * Theta_train + alpha * I;
    R = Theta_train' * Y_train;
    Xi_full = G \ R;

    best_th = NaN; best_rmse = inf; best_Xi = [];
    cv_results = cell(1, length(thresholds));

    for t_idx = 1:length(thresholds)
        th = thresholds(t_idx);
        Xi = Xi_full;
        mask = abs(Xi) < th;
        Xi(mask) = 0;

        % Per-output re-solve
        for j = 1:n_targets
            keep = ~mask(:, j);
            n_keep = sum(keep);
            if n_keep == 0 || n_keep >= size(Theta_train, 1)
                continue;
            end
            Th_sub = Theta_train(:, keep);
            Xi(keep, j) = Th_sub \ Y_train(:, j);
        end

        Y_pred = Theta_val * Xi;
        rmse = sqrt(mean((Y_pred(:) - Y_val(:)).^2));
        n_nz = nnz(Xi);
        cv_results{t_idx} = {th, rmse, n_nz};

        if rmse < best_rmse
            best_rmse = rmse; best_th = th; best_Xi = Xi;
        end
    end
end
