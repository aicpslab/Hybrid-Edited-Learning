%% ========================================================================
% SINDy vs PhNN COMPARISON -- Lorenz-96 N=40
% ========================================================================
% Standard SINDy (Brunton et al., PNAS 2016) using STLSQ on the SAME
% Taylor library as the PhNN. Same data, same evaluation.
%
% Key question: Can data-driven sparse regression (SINDy) match
%                physics-guided editing (PIM)?
%
% Author:  Yang Yejiang (Southwest Minzu University)
% ========================================================================

clear; close all; clc;

%% ========================================================================
% SECTION 1: PARAMETERS
% ========================================================================
N = 40; dt = 0.01; F = 8.0; r = 2;
n_train = 6000; n_val = 2000; n_test = 3000;
seed = 42;

fprintf('=============================================================\n');
fprintf('SINDy vs PhNN -- Lorenz-96 (N=%d, r=%d)\n', N, r);
fprintf('  Same Taylor library, same data, same evaluation\n');
fprintf('=============================================================\n\n');

%% ========================================================================
% SECTION 2: DATA & TAYLOR LIBRARY
% ========================================================================
fprintf('[Step 1/5] Generating data & Taylor library...\n');

% Generate data
[X_train, Y_train, X_val, Y_val, X_test, Y_test] = ...
    generate_lorenz96_data(N, dt, F, n_train, n_val, n_test, seed);

% Taylor library (same 860 monomials as PhNN)
mono = generate_monomials(N, r);
n_mono = length(mono);
fprintf('  Monomials: %d\n', n_mono);

% Build library matrices
fprintf('  Building library matrix (train: %d x %d)...\n', n_train, n_mono);
Theta_tr = taylor_expand(double(X_train), mono);  % (6000, 860)
Theta_va = taylor_expand(double(X_val), mono);
Theta_te = taylor_expand(double(X_test), mono);
fprintf('  Library built.\n');

%% ========================================================================
% SECTION 3: STANDARD SINDy (STLSQ)
% ========================================================================
fprintf('\n[Step 2/5] Standard SINDy (STLSQ) -- CV over threshold...\n');

% CV over thresholds
thresholds = logspace(-2, 0, 10);
best_lambda = 0; best_rmse = inf; best_Xi = []; best_nnz = 0;
cv_results = zeros(length(thresholds), 3);

fprintf('  Testing %d thresholds...\n', length(thresholds));
tic;
for idx = 1:length(thresholds)
    lambda = thresholds(idx);
    Xi = sindy_stlsq(Theta_tr, double(Y_train), lambda, 5);
    Yp = Theta_va * Xi;
    rmse_val = sqrt(mean((Yp(:) - Y_val(:)).^2));
    nnz_val = nnz(Xi);
    cv_results(idx, :) = [lambda, rmse_val, nnz_val];

    fprintf('    lambda=%.4f: nnz=%d, val_rmse=%.4f\n', lambda, nnz_val, rmse_val);

    if rmse_val < best_rmse
        best_rmse = rmse_val; best_lambda = lambda;
        best_Xi = Xi; best_nnz = nnz_val;
    end
end
t_sindy = toc;

Xi_sindy = best_Xi;
n_nz_sindy = nnz(Xi_sindy);
sp_sindy = 1.0 - n_nz_sindy / numel(Xi_sindy);
Yp_te_sindy = Theta_te * Xi_sindy;
rmse_sindy = sqrt(mean((Yp_te_sindy(:) - Y_test(:)).^2));

fprintf('\n  Best lambda=%.4f, nnz=%d/%d, sparsity=%.1f%%\n', ...
        best_lambda, n_nz_sindy, numel(Xi_sindy), sp_sindy*100);
fprintf('  Test RMSE=%.6e, Time=%.1fs\n', rmse_sindy, t_sindy);

%% ========================================================================
% SECTION 4: STRUCTURE RECOVERY ANALYSIS
% ========================================================================
fprintf('\n[Step 3/5] Structure recovery analysis...\n');

n_spurious = 0; n_missed = 0; n_true_total = 0;
for i = 1:N
    % Relevant inputs for output i (cyclic)
    im2 = mod(i-3, N)+1; im1 = mod(i-2, N)+1; ip1 = mod(i, N)+1;
    relevant_set = [im2, im1, i, ip1];

    true_terms_i = [];
    for h = 1:n_mono
        if all(ismember(mono{h}, relevant_set))
            true_terms_i(end+1) = h; %#ok<AGROW>
        end
    end
    n_true_total = n_true_total + length(true_terms_i);

    % Missed: true term has zero coefficient
    for h = true_terms_i
        if Xi_sindy(h, i) == 0, n_missed = n_missed + 1; end
    end

    % Spurious: non-zero term outside true set
    for h = 1:n_mono
        if Xi_sindy(h, i) ~= 0 && ~ismember(h, true_terms_i)
            n_spurious = n_spurious + 1;
        end
    end
end

precision = (n_nz_sindy - n_spurious) / max(n_nz_sindy, 1) * 100;
recall = (n_true_total - n_missed) / max(n_true_total, 1) * 100;

fprintf('  True relevant terms (per physics): %d\n', n_true_total);
fprintf('  SINDy selected (non-zero):         %d\n', n_nz_sindy);
fprintf('  Spurious (outside neighbor set):   %d\n', n_spurious);
fprintf('  Missed (should be non-zero):       %d\n', n_missed);
fprintf('  Precision: %.1f%%, Recall: %.1f%%\n', precision, recall);

%% ========================================================================
% SECTION 5: PhNN MODELS (for comparison)
% ========================================================================
fprintf('\n[Step 4/5] Training PhNN models for comparison...\n');

% PIM mask
[A_value_pim, A_uncertain_pim, pim_sparsity] = build_pim_lorenz96(N, dt, mono);

fprintf('\n  --- PhNN Unedited ---\n');
W_init = single(randn(N, n_mono) * 0.01);
b_init = single(zeros(N, 1));
A_val_zero = single(zeros(N, n_mono));
A_unc_all = single(ones(N, n_mono));

tic;
[W_u, b_u, ~, vl_u, bv_u] = phnn_train(...
    X_train, Y_train, X_val, Y_val, mono, ...
    A_val_zero, A_unc_all, W_init, b_init, ...
    0.001, 200, 256, true);
t_u = toc;
Yp_u = phnn_forward(X_test, mono, A_val_zero, A_unc_all, W_u, b_u);
rmse_u = sqrt(mean((Yp_u(:) - Y_test(:)).^2));

fprintf('  -> Val loss=%.4e, RMSE=%.4e, Params=%d, Time=%.0fs\n', ...
        bv_u, rmse_u, sum(A_unc_all(:)), t_u);

fprintf('\n  --- PhNN + PIM ---\n');
W_init_pim = single(randn(N, n_mono) * 0.01);

tic;
[W_p, b_p, ~, vl_p, bv_p] = phnn_train(...
    X_train, Y_train, X_val, Y_val, mono, ...
    A_value_pim, A_uncertain_pim, W_init_pim, b_init, ...
    0.001, 200, 256, true);
t_p = toc;
Yp_p = phnn_forward(X_test, mono, A_value_pim, A_uncertain_pim, W_p, b_p);
rmse_p = sqrt(mean((Yp_p(:) - Y_test(:)).^2));
n_pim_learnable = sum(A_uncertain_pim(:));

fprintf('  -> Val loss=%.4e, RMSE=%.4e, Params=%d, Time=%.0fs\n', ...
        bv_p, rmse_p, n_pim_learnable, t_p);

%% ========================================================================
% SECTION 6: RESULTS SUMMARY
% ========================================================================
fprintf('\n[Step 5/5] Results Summary & Figures...\n');
fprintf('=================================================================\n');
fprintf('SINDy vs PhNN -- Lorenz-96 (N=%d, r=%d)\n', N, r);
fprintf('  Same Taylor library (%d monomials), same data\n', n_mono);
fprintf('=================================================================\n');

fprintf('\n%-22s %-12s %-10s %-14s %-10s %-10s\n', ...
        'Method', 'Nonzero', 'Sparsity', 'Test RMSE', 'Precision', 'Recall');
fprintf('%s\n', repmat('-', 1, 85));
fprintf('%-22s %-12d %-9.1f%% %-14.6e %-9.1f%% %-9.1f%%\n', ...
        'SINDy (STLSQ)', n_nz_sindy, sp_sindy*100, rmse_sindy, precision, recall);
fprintf('%-22s %-12d %-9.1f%% %-14.6e %-9s %-9s\n', ...
        'PhNN Unedited', sum(A_unc_all(:)), 0.0, rmse_u, '--', '--');
fprintf('%-22s %-12d %-9.1f%% %-14.6e %-9s %-9s\n', ...
        'PhNN + PIM', n_pim_learnable, pim_sparsity*100, rmse_p, '100.0%', '100.0%');

fprintf('\n  Key Ratios:\n');
fprintf('    PIM RMSE / SINDy RMSE: %.4f\n', rmse_p / max(rmse_sindy, 1e-15));
fprintf('    SINDy RMSE / PIM RMSE: %.1fx\n', rmse_sindy / max(rmse_p, 1e-15));
fprintf('    PIM loss / SINDy loss: %.4f\n', bv_p / max(best_rmse, 1e-15));

%% ========================================================================
% FIGURES
% ========================================================================

% --- Figure S1: SINDy Coefficient Matrix vs PIM Mask vs Unedited Weights ---
figure('Position', [100, 100, 1700, 500]);

subplot(1,3,1);
imagesc(abs(Xi_sindy));
colormap hot; colorbar;
vmax = prctile(abs(Xi_sindy(:)), 98);
if vmax > 0, caxis([0, vmax]); end
xlabel('Output Dimension'); ylabel('Monomial Index');
title(sprintf('(a) SINDy (STLSQ) |Xi|   \\lambda=%.2e\n%d nonzero, sparsity=%.1f%%', ...
       best_lambda, n_nz_sindy, sp_sindy*100), 'FontWeight', 'bold');

subplot(1,3,2);
imagesc(A_uncertain_pim);
colormap gray; colorbar;
xlabel('Output Dimension'); ylabel('Monomial Index');
title(sprintf('(b) PIM Mask (Physics Prior)\n%d learnable, sparsity=%.1f%%', ...
       n_pim_learnable, pim_sparsity*100), 'FontWeight', 'bold');

subplot(1,3,3);
imagesc(abs(W_u));
colormap hot; colorbar;
vmax_w = prctile(abs(W_u(:)), 98);
if vmax_w > 0, caxis([0, vmax_w]); end
xlabel('Output Dimension'); ylabel('Monomial Index');
title(sprintf('(c) PhNN Unedited |W|\n%d params, 0%% sparse', sum(A_unc_all(:))), ...
       'FontWeight', 'bold');

sgtitle('Figure S1: Coefficient Structure -- SINDy vs PIM vs Unedited PhNN', ...
       'FontWeight', 'bold', 'FontSize', 13);
saveas(gcf, 'fig/FigS1_SINDy_Coefficients.png');
fprintf('  FigS1_SINDy_Coefficients.png saved.\n');

% --- Figure S2: RMSE Comparison Bar Chart ---
figure('Position', [100, 100, 800, 500]);
methods = {'SINDy (STLSQ)', 'PhNN Unedited', 'PhNN + PIM'};
rmse_vals = [rmse_sindy, rmse_u, rmse_p];
b = bar(rmse_vals);
b.FaceColor = 'flat';
b.CData = [0.47 0.67 0.19; 0.85 0.33 0.10; 0 0.45 0.74];
set(gca, 'XTickLabel', methods, 'YScale', 'log');
ylabel('Test RMSE (log scale)');
title('Prediction Accuracy: SINDy vs PhNN');
grid on;
for i = 1:length(rmse_vals)
    text(i, rmse_vals(i)*1.2, sprintf('%.4f', rmse_vals(i)), ...
         'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 10);
end
sgtitle('Figure S2: Test RMSE -- Same Library, Same Data', ...
       'FontWeight', 'bold', 'FontSize', 13);
saveas(gcf, 'fig/FigS2_RMSE_Comparison.png');
fprintf('  FigS2_RMSE_Comparison.png saved.\n');

% --- Figure S3: SINDy CV Curve ---
figure('Position', [100, 100, 800, 500]);
yyaxis left;
semilogx(cv_results(:,1), cv_results(:,2), 'b-o', 'LineWidth', 1.5, 'MarkerSize', 5);
xlabel('Threshold \lambda'); ylabel('Validation RMSE');
xline(best_lambda, 'k--', 'LineWidth', 1.5);
yyaxis right;
semilogx(cv_results(:,1), cv_results(:,3), 'r-s', 'LineWidth', 1.5, 'MarkerSize', 5);
ylabel('Number of Nonzero Coefficients');
title(sprintf('SINDy (STLSQ): Threshold CV   Best \\lambda=%.4f', best_lambda));
legend({'Val RMSE', sprintf('Best \\lambda=%.4f', best_lambda), 'Nonzero Coeffs'}, ...
       'Location', 'best');
grid on;
sgtitle('Figure S3: SINDy Hyperparameter Selection', ...
       'FontWeight', 'bold', 'FontSize', 13);
saveas(gcf, 'fig/FigS3_SINDy_CV.png');
fprintf('  FigS3_SINDy_CV.png saved.\n');

% --- Figure S4: SINDy Terms per Output ---
figure('Position', [100, 100, 800, 500]);
nz_per_output = sum(abs(Xi_sindy) > 0, 1);
bar(1:N, nz_per_output, 'FaceColor', [0.47 0.67 0.19], 'EdgeColor', 'k', 'LineWidth', 0.5);
hold on;
yline(14, 'k--', 'LineWidth', 1.5);  % Expected: 14 terms (4 neighbors, r=2)
yline(mean(nz_per_output), '-', 'LineWidth', 1.5, 'Color', [0.47 0.67 0.19]);
hold off;
xlabel('Output Dimension'); ylabel('# Selected Terms');
title(sprintf('SINDy (STLSQ): Terms per Output (mean=%.1f)', mean(nz_per_output)));
legend({'Per-output count', 'Expected: 14 (4 neighbors, r=2)', ...
       sprintf('SINDy mean: %.1f', mean(nz_per_output))}, 'Location', 'best');
grid on;
sgtitle('Figure S4: SINDy Sparsity Pattern per Output', ...
       'FontWeight', 'bold', 'FontSize', 13);
saveas(gcf, 'fig/FigS4_SINDy_PerOutput.png');
fprintf('  FigS4_SINDy_PerOutput.png saved.\n');

% --- Figure S5: Coefficient Recovery ---
figure('Position', [100, 100, 1500, 500]);

% True coefficients for Lorenz-96 (Euler discretization)
true_coeffs = zeros(n_mono, 1);
sindy_avg = zeros(n_mono, 1);
for i = 1:N
    im1 = mod(i-2, N)+1; im2 = mod(i-3, N)+1; ip1 = mod(i, N)+1;
    for h = 1:n_mono
        midx = mono{h};
        if length(midx) == 1 && midx(1) == i
            true_coeffs(h) = 1.0 - dt;
        elseif length(midx) == 2
            if (midx(1) == im1 && midx(2) == ip1) || (midx(1) == ip1 && midx(2) == im1)
                true_coeffs(h) = dt;
            elseif (midx(1) == im1 && midx(2) == im2) || (midx(1) == im2 && midx(2) == im1)
                true_coeffs(h) = -dt;
            end
        end
    end
    sindy_avg = sindy_avg + abs(Xi_sindy(:, i));
end
sindy_avg = sindy_avg / N;

W_pim = abs(A_value_pim + A_uncertain_pim .* double(W_p));
pim_avg = mean(W_pim, 1)';  % Average over 40 outputs -> (n_mono, 1)

n_top = min(80, sum(true_coeffs ~= 0) * 3);
[~, idx_s] = sort(sindy_avg, 'descend');
[~, idx_p] = sort(pim_avg, 'descend');
idx_s = idx_s(1:n_top); idx_p = idx_p(1:n_top);

subplot(1,2,1);
bar_width = 0.35; xr = 1:n_top;
bar(xr - bar_width/2, abs(true_coeffs(idx_s)), bar_width, 'k', 'DisplayName', 'True (Euler)');
hold on;
bar(xr + bar_width/2, sindy_avg(idx_s), bar_width, ...
    'FaceColor', [0.47 0.67 0.19], 'DisplayName', 'SINDy (STLSQ)');
hold off;
xlabel('Monomial Rank'); ylabel('|Coefficient|');
title('(a) SINDy vs True Coefficients', 'FontWeight', 'bold');
legend('Location', 'best'); grid on;

subplot(1,2,2);
bar(xr - bar_width/2, abs(true_coeffs(idx_p)), bar_width, 'k', 'DisplayName', 'True (Euler)');
hold on;
bar(xr + bar_width/2, pim_avg(idx_p), bar_width, ...
    'FaceColor', [0 0.45 0.74], 'DisplayName', 'PhNN+PIM');
hold off;
xlabel('Monomial Rank'); ylabel('|Coefficient|');
title('(b) PhNN+PIM vs True Coefficients', 'FontWeight', 'bold');
legend('Location', 'best'); grid on;

sgtitle('Figure S5: Coefficient Recovery Accuracy', ...
       'FontWeight', 'bold', 'FontSize', 13);
saveas(gcf, 'fig/FigS5_CoefficientRecovery.png');
fprintf('  FigS5_CoefficientRecovery.png saved.\n');

%% ========================================================================
% FINAL
% ========================================================================
fprintf('\n=============================================================\n');
fprintf('SINDy COMPARISON COMPLETE\n');
fprintf('  SINDy sparsity: %.1f%% (auto) vs PIM sparsity: %.1f%% (physics)\n', ...
        sp_sindy*100, pim_sparsity*100);
fprintf('  SINDy RMSE / PIM RMSE: %.1fx\n', rmse_sindy / max(rmse_p, 1e-15));
fprintf('  SINDy precision: %.1f%%, recall: %.1f%%\n', precision, recall);
fprintf('  All figures saved.\n');
fprintf('=============================================================\n');

%% ========================================================================
% LOCAL FUNCTIONS
% ========================================================================

function Xi = sindy_stlsq(Theta, Y, lambda, max_iter)
% Standard SINDy STLSQ (Brunton et al., PNAS 2016)
%   Initial LS -> threshold -> re-solve per output -> iterate
    if nargin < 4, max_iter = 10; end
    [M, P] = size(Theta); D = size(Y, 2); %#ok<ASGLU>
    Xi = Theta \ Y;
    for it = 1:max_iter
        small = abs(Xi) < lambda;
        if ~any(small(:)), break; end
        Xi(small) = 0;
        for j = 1:D
            keep = find(~small(:, j));
            nk = length(keep);
            if nk == 0 || nk > 200, continue; end
            Xi(keep, j) = Theta(:, keep) \ Y(:, j);
        end
    end
end
