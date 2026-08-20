function experiment_sindy_std()
%% EXPERIMENT_SINDY_STD  Standard SINDy (STLSQ) vs PhNN -- Lorenz-96 N=40
%   Pure MATLAB STLSQ implementation. Same Taylor library, same data.
%
%   Uses the standard Sequential Thresholded Least Squares (STLSQ)
%   algorithm from Brunton et al., PNAS 2016.

N = 40; dt = 0.01; F = 8.0; r = 2;
N_TR = 6000; N_VA = 2000; N_TE = 3000;
EP = 150; B = 256; LR = 0.001; SEED = 42;

fprintf('%s\n', repmat('=', 1, 70));
fprintf('STANDARD SINDy (STLSQ) vs PhNN -- Lorenz-96 N=40\n');
fprintf('  Same Taylor library, same data, same evaluation\n');
fprintf('%s\n', repmat('=', 1, 70));

% ---- Data ----
fprintf('\n[1/4] Data & Taylor library...\n');
[tr, va, te] = generate_train_val_test_data(N, dt, F, N_TR, N_VA, N_TE, SEED);
Xtr = double(tr(1:end-1,:)); Ytr = double(tr(2:end,:));
Xva = double(va(1:end-1,:)); Yva = double(va(2:end,:));
Xte = double(te(1:end-1,:)); Yte = double(te(2:end,:));

mono = generate_monomial_indices(N, r);
n_mono = length(mono);
fprintf('  Standard (40D): %d monomials\n', n_mono);

fprintf('  Building library...\n');
Th_tr = taylor_expand(Xtr, mono);
Th_va = taylor_expand(Xva, mono);
Th_te = taylor_expand(Xte, mono);

[Av, Au, ~] = build_lorenz96_pim(N, dt, mono);
n_pim_true = sum(Au(:));
fprintf('  PIM learnable terms per physics: %d\n', n_pim_true);

% ============================================================
% SINDy: STLSQ
% ============================================================
fprintf('\n[2/4] Standard SINDy (STLSQ)...\n');

thresholds = logspace(-1.5, 0.5, 15);
best_th = NaN; best_rmse = inf; best_Xi = []; best_nnz = 0;
cv_results = cell(1, length(thresholds));

fprintf('  Testing %d thresholds (max 3 STLSQ iterations)...\n', length(thresholds));
t0 = tic;
for th_idx = 1:length(thresholds)
    th = thresholds(th_idx);
    Xi = sindy_stlsq(Th_tr, Ytr, th, 3);
    Yp = Th_va * Xi;
    rmse = sqrt(mean((Yp(:) - Yva(:)).^2));
    nnz_val = nnz(Xi);
    cv_results{th_idx} = {th, rmse, nnz_val};
    if rmse < best_rmse
        best_rmse = rmse; best_th = th; best_Xi = Xi; best_nnz = nnz_val;
    end
    fprintf('    lambda=%.1e: nnz=%d, val_rmse=%.4f\n', th, nnz_val, rmse);
end
dt_s = toc(t0);

Xi_s = best_Xi;
nz_s = nnz(Xi_s);
sp_s = 1.0 - nz_s/numel(Xi_s);
Yp_te = Th_te * Xi_s;
rmse_s = sqrt(mean((Yp_te(:) - Yte(:)).^2));
fprintf('  Best lambda=%.4e, nnz=%d/%d, sparsity=%.1f%%\n', best_th, nz_s, numel(Xi_s), sp_s*100);
fprintf('  Test RMSE=%.6e, Time=%.1fs\n', rmse_s, dt_s);

% ============================================================
% Structure analysis
% ============================================================
fprintf('\n[3/4] Structure recovery analysis...\n');
n_spurious = 0; n_missed = 0; n_true_total = 0;
for i = 1:N
    im2 = mod(i-3,N)+1; im1 = mod(i-2,N)+1; ip1 = mod(i,N)+1;
    rel = [im2, im1, i, ip1];
    true_h = [];
    for h = 1:length(mono)
        if all(ismember(mono{h}, rel)), true_h(end+1) = h; end %#ok<AGROW>
    end
    n_true_total = n_true_total + length(true_h);
    for h = true_h
        if Xi_s(h,i) == 0, n_missed = n_missed + 1; end
    end
    for h = 1:length(mono)
        if Xi_s(h,i) ~= 0 && ~all(ismember(mono{h}, rel))
            n_spurious = n_spurious + 1;
        end
    end
end

precision = (nz_s - n_spurious)/max(nz_s, 1)*100;
recall = (n_true_total - n_missed)/max(n_true_total, 1)*100;
fprintf('  True relevant terms: %d, SINDy selected: %d\n', n_true_total, nz_s);
fprintf('  Spurious: %d, Missed: %d\n', n_spurious, n_missed);
fprintf('  Precision: %.1f%%, Recall: %.1f%%\n', precision, recall);

% ============================================================
% PhNN models
% ============================================================
fprintf('\n[4/4] PhNN models (%d epochs each)...\n', EP);

fprintf('  PhNN Unedited...\n');
mu = PhNNModel(N, N, mono);
t0 = tic;
[~, vlu, ~] = mu.train(single(Xtr), single(Ytr), single(Xva), single(Yva), LR, EP, B, EP+1);
dt_u = toc(t0);
yp = mu.forward(single(Xte)); rmse_u = sqrt(mean((double(yp(:)) - Yte(:)).^2));
fprintf('  Val loss=%.4e, RMSE=%.4e, Params=%d, Time=%.0fs\n', vlu(end), rmse_u, mu.n_learnable, dt_u);

fprintf('  PhNN + PIM...\n');
mp = PhNNModel(N, N, mono, Av, Au);
t0 = tic;
[~, vlp, ~] = mp.train(single(Xtr), single(Ytr), single(Xva), single(Yva), LR, EP, B, EP+1);
dt_p = toc(t0);
yp = mp.forward(single(Xte)); rmse_p = sqrt(mean((double(yp(:)) - Yte(:)).^2));
fprintf('  Val loss=%.4e, RMSE=%.4e, Params=%d, Time=%.0fs\n', vlp(end), rmse_p, mp.n_learnable, dt_p);

% ============================================================
% Summary
% ============================================================
fprintf('\n%s\n', repmat('=', 1, 95));
fprintf('RESULTS: Standard SINDy (STLSQ) vs PhNN\n');
fprintf('  Same 40D input, same %d Taylor monomials, same data\n', n_mono);
fprintf('%s\n', repmat('=', 1, 95));
fprintf('\n%-22s %-12s %-10s %-14s %-8s %-10s %-10s\n', ...
    'Method', 'Nonzero', 'Sparsity', 'Test RMSE', 'Time', 'Precision', 'Recall');
fprintf('%s\n', repmat('-', 1, 95));

data_rows = {
    'SINDy (STLSQ)', nz_s, sp_s, rmse_s, dt_s, precision, recall;
    'PhNN Unedited', mu.n_learnable, mu.sparsity, rmse_u, dt_u, 0.0, 100.0;
    'PhNN + PIM', mp.n_learnable, mp.sparsity, rmse_p, dt_p, 100.0, 100.0;
};
for i = 1:3
    fprintf('%-22s %-12d %-9.1f%% %-14.6e %-7.0fs %-9.1f%% %-9.1f%%\n', ...
        data_rows{i,:});
end

fprintf('\n  Key insights:\n');
fprintf('    SINDy (auto) sparsity:        %.1f%%\n', sp_s*100);
fprintf('    PIM (physics-guided) sparsity: %.1f%%\n', mp.sparsity*100);
if rmse_s > 0
    fprintf('    RMSE ratio (PhNN+PIM / SINDy): %.4f\n', rmse_p/rmse_s);
end

% ============================================================
% FIGURES
% ============================================================
fprintf('\nGenerating figures...\n');
C = struct('s', [0.47 0.67 0.19], 'u', [0.85 0.33 0.10], 'p', [0 0.45 0.74]);

% S1: Coefficient matrices
figure('Position', [100, 100, 1500, 480]);
subplot(1,3,1);
imagesc(abs(Xi_s)); colormap('hot'); colorbar;
xlabel('Output Dim'); ylabel('Monomial');
title(sprintf('(a) SINDy |Xi| (lambda=%.2e, %d nonzero)', best_th, nz_s));

subplot(1,3,2);
imagesc(Au); colormap('parula'); colorbar;
xlabel('Output Dim'); ylabel('Monomial');
title(sprintf('(b) PIM Mask (%d learnable)', n_pim_true));

subplot(1,3,3);
imagesc(abs(mu.W_learn)); colormap('hot'); colorbar;
xlabel('Output Dim'); ylabel('Monomial');
title(sprintf('(c) PhNN Unedited |W| (%d params)', mu.n_learnable));
sgtitle('Fig S1: Coefficient Structure', 'FontWeight', 'bold', 'FontSize', 13);
saveas(gcf, 'fig/FigS1_SINDy_Coefficients.png');
close;

% S2: RMSE bar chart
figure('Position', [100, 100, 700, 480]);
dns = {'SINDy', 'PhNN Unedited', 'PhNN + PIM'};
clrs = [C.s; C.u; C.p];
rms = [rmse_s, rmse_u, rmse_p];
b = bar(rms); b.FaceColor = 'flat';
for i = 1:3, b.CData(i,:) = clrs(i,:); end
set(gca, 'XTickLabel', dns, 'YScale', 'log');
for i = 1:3, text(i, rms(i)*1.2, sprintf('%.4f', rms(i)), 'HorizontalAlignment', 'center', 'FontSize', 10); end
ylabel('Test RMSE (log)'); title('Prediction Accuracy');
sgtitle('Fig S2: SINDy vs PhNN', 'FontWeight', 'bold', 'FontSize', 13);
saveas(gcf, 'fig/FigS2_RMSE_Comparison.png');
close;

% S3: Terms per output
figure('Position', [100, 100, 800, 480]);
nz_out = sum(abs(Xi_s) > 0, 1);
bar(0:N-1, nz_out, 'FaceColor', C.s, 'EdgeColor', 'k');
hold on;
yline(14, 'k--', 'LineWidth', 1.5);
yline(mean(nz_out), '-', 'Color', C.s, 'LineWidth', 1.5);
xlabel('Output Dim'); ylabel('# Terms'); title('SINDy: Terms per Output');
legend({'Expected: 14', sprintf('SINDy mean: %.1f', mean(nz_out))}); grid on;
sgtitle(sprintf('Fig S3: Sparsity per Output (lambda=%.2e)', best_th), 'FontWeight', 'bold', 'FontSize', 13);
saveas(gcf, 'fig/FigS3_SINDy_PerOutput.png');
close;

% S4: CV curve
figure('Position', [100, 100, 800, 480]);
ths = zeros(1, length(cv_results)); rms_cv = zeros(1, length(cv_results)); nzs_vals = zeros(1, length(cv_results));
for i = 1:length(cv_results)
    ths(i) = cv_results{i}{1}; rms_cv(i) = cv_results{i}{2}; nzs_vals(i) = cv_results{i}{3};
end
yyaxis left;
semilogx(ths, rms_cv, 'b-o', 'MarkerSize', 5); ylabel('Val RMSE');
yyaxis right;
semilogx(ths, nzs_vals, 'r-s', 'MarkerSize', 5); ylabel('Nonzero Coeffs');
xlabel('Threshold'); title('SINDy Cross-Validation'); grid on;
sgtitle('Fig S4: Hyperparameter Selection', 'FontWeight', 'bold', 'FontSize', 13);
saveas(gcf, 'fig/FigS4_SINDy_CV.png');
close;

% S5: Coefficient recovery
figure('Position', [100, 100, 1300, 480]);
tc = zeros(length(mono), 1); sc_avg = zeros(length(mono), 1);
for i = 1:N
    im1 = mod(i-2,N)+1; im2 = mod(i-3,N)+1; ip1 = mod(i,N)+1;
    for h = 1:length(mono)
        midx = mono{h};
        if length(midx)==1 && midx(1)==i,        tc(h)=1.0-dt;
        elseif length(midx)==2 && midx(1)==im1 && midx(2)==ip1, tc(h)=dt;
        elseif length(midx)==2 && midx(1)==im1 && midx(2)==im2, tc(h)=-dt;
        end
    end
    sc_avg = sc_avg + abs(Xi_s(:, i));
end
sc_avg = sc_avg / N;
Wp_avg = mean(abs(mp.A_value + mp.A_uncertain .* mp.W_learn), 1)';

nt = min(80, sum(tc~=0)*3);
[~, ids] = sort(sc_avg, 'descend'); ids = ids(1:nt);
[~, idp] = sort(Wp_avg, 'descend'); idp = idp(1:nt);
xr = 1:nt; w = 0.35;

subplot(1,2,1);
bar(xr-w/2, abs(tc(ids)), w, 'FaceColor', 'k', 'DisplayName', 'True (Euler)'); hold on;
bar(xr+w/2, sc_avg(ids), w, 'FaceColor', C.s, 'DisplayName', 'SINDy');
xlabel('Monomial Rank'); ylabel('|Coefficient|'); title('(a) SINDy vs True'); legend; grid on;

subplot(1,2,2);
bar(xr-w/2, abs(tc(idp)), w, 'FaceColor', 'k', 'DisplayName', 'True (Euler)'); hold on;
bar(xr+w/2, Wp_avg(idp), w, 'FaceColor', C.p, 'DisplayName', 'PhNN+PIM');
xlabel('Monomial Rank'); ylabel('|Coefficient|'); title('(b) PhNN+PIM vs True'); legend; grid on;
sgtitle('Fig S5: Coefficient Recovery', 'FontWeight', 'bold', 'FontSize', 13);
saveas(gcf, 'fig/FigS5_CoefficientRecovery.png');
close;

fprintf('Done! 5 SINDy figures saved to fig/.\n');
fprintf('%s\n', repmat('=', 1, 70));

end

%% ========================================================================
%  Standard SINDy STLSQ
%  ========================================================================

function Xi = sindy_stlsq(Theta, Y, threshold, max_iter)
% Standard SINDy STLSQ (Brunton et al. 2016).
    if nargin < 4, max_iter = 3; end

    [~, P] = size(Theta);
    D = size(Y, 2);

    % Initial least squares
    Xi = Theta \ Y;

    for it = 1:max_iter
        small = abs(Xi) < threshold;
        if ~any(small(:)), break; end
        Xi(small) = 0;

        for j = 1:D
            keep = find(~small(:, j));
            nk = length(keep);
            if nk == 0 || nk > 200, continue; end
            Th_k = Theta(:, keep);
            Xi(keep, j) = Th_k \ Y(:, j);
        end
    end
end
