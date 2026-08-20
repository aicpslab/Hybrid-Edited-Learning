function [model, cross_weights, within_weights, ratios] = tkm_validation()
%% TKM_VALIDATION  Train unedited PhNN on temporal data and analyze weights
%   Verifies that cross-temporal weights are negligible, empirically
%   justifying TKM pruning for first-order Markov systems.
%
%   [model, cross_w, within_w, ratios] = tkm_validation()

N = 40; dt = 0.01; F = 8.0; r = 2; K = 2;
dim_temporal = N * K;

fprintf('%s\n', repmat('=', 1, 60));
fprintf('TKM Validation: Cross-Temporal Weight Analysis\n');
fprintf('  Lorenz-96 N=%d, r=%d, K=%d temporal steps\n', N, r, K);
fprintf('  Input dim: %d\n', dim_temporal);
fprintf('%s\n', repmat('=', 1, 60));

% ---- Generate data ----
fprintf('\n[1/6] Generating Lorenz-96 data...\n');
[train_traj, val_traj, test_traj] = generate_train_val_test_data(...
    N, dt, F, 8000, 2000, 2000, 123);

[Xt_train, Yt_train] = build_temporal_data(train_traj, N, K);
[Xt_val,   Yt_val]   = build_temporal_data(val_traj,   N, K);
[Xt_test,  Yt_test]  = build_temporal_data(test_traj,  N, K);

n_samples = 4000;
Xt_train_s = Xt_train(1:n_samples, :); Yt_train_s = Yt_train(1:n_samples, :);
Xt_val_s   = Xt_val(1:1000, :);        Yt_val_s   = Yt_val(1:1000, :);

fprintf('  Train: %d, Val: %d, Test: %d\n', size(Xt_train_s,1), size(Xt_val_s,1), size(Xt_test,1));

% ---- Build Taylor expansion ----
fprintf('\n[2/6] Building Taylor expansion (r=2, dim=80)...\n');
mono_temp = generate_monomial_indices(dim_temporal, r);
n_mono = length(mono_temp);
fprintf('  Monomials: %d\n', n_mono);

% ---- Classify hidden neurons ----
cross_temporal_mask = false(1, n_mono);
within_temporal_mask = false(1, n_mono);
for h = 1:n_mono
    midx = mono_temp{h};
    time_steps = unique(floor((midx - 1) / N));  % 0-based time steps
    if length(time_steps) > 1
        cross_temporal_mask(h) = true;
    else
        within_temporal_mask(h) = true;
    end
end

n_cross = sum(cross_temporal_mask);
n_within = sum(within_temporal_mask);
fprintf('  Cross-temporal neurons: %d (%.1f%%)\n', n_cross, n_cross/n_mono*100);
fprintf('  Within-temporal neurons: %d (%.1f%%)\n', n_within, n_within/n_mono*100);

% ---- Train unedited PhNN on temporal data ----
fprintf('\n[3/6] Training unedited PhNN on temporal data...\n');
model = PhNNModel(dim_temporal, N, mono_temp);
model.summary();

model.train(Xt_train_s, Yt_train_s, Xt_val_s, Yt_val_s, ...
    0.002, 100, 256, 50);

% ---- Analyze learned weights ----
fprintf('\n[4/6] Analyzing cross-temporal vs within-temporal weights...\n');
W_eff = model.A_value + model.A_uncertain .* model.W_learn;

cross_weights = abs(W_eff(:, cross_temporal_mask));
cross_weights = cross_weights(:);
within_weights = abs(W_eff(:, within_temporal_mask));
within_weights = within_weights(:);

mean_cross = mean(cross_weights);
mean_within = mean(within_weights);
median_cross = median(cross_weights);
median_within = median(within_weights);
std_cross = std(cross_weights);
std_within = std(within_weights);

fprintf('\n  Cross-temporal weights:\n');
fprintf('    Mean:   %.6e\n', mean_cross);
fprintf('    Median: %.6e\n', median_cross);
fprintf('    Std:    %.6e\n', std_cross);
fprintf('\n  Within-temporal weights:\n');
fprintf('    Mean:   %.6e\n', mean_within);
fprintf('    Median: %.6e\n', median_within);
fprintf('    Std:    %.6e\n', std_within);
fprintf('\n  Ratio (cross / within):\n');
fprintf('    Mean ratio:   %.4f\n', mean_cross/(mean_within + 1e-10));
fprintf('    Median ratio: %.4f\n', median_cross/(median_within + 1e-10));

% ---- Weight sparsity analysis ----
fprintf('\n  Fraction of weights below threshold:\n');
for th = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    frac_cross = mean(cross_weights < th) * 100;
    frac_within = mean(within_weights < th) * 100;
    fprintf('    |W| < %.0e:  cross=%.1f%%,  within=%.1f%%\n', th, frac_cross, frac_within);
end

% ---- Per-output analysis ----
fprintf('\n  Per-output ratio (first 8 and last 3 outputs):\n');
ratios = zeros(N, 1);
for i = 1:N
    cw = mean(abs(W_eff(i, cross_temporal_mask)));
    ww = mean(abs(W_eff(i, within_temporal_mask)));
    ratios(i) = cw / (ww + 1e-10);
    if i <= 8 || i >= N-2
        fprintf('    Output %2d:  cross=%.6e,  within=%.6e,  ratio=%.4f\n', ...
            i-1, cw, ww, ratios(i));
    end
end
fprintf('  Mean ratio across all outputs: %.4f\n', mean(ratios));
fprintf('  Min ratio: %.4f,  Max ratio: %.4f\n', min(ratios), max(ratios));

% ---- Test error comparison ----
fprintf('\n[5/6] Comparing test error: full model vs TKM-pruned...\n');
test_pred_full = model.forward(Xt_test(1:500, :));
test_rmse_full = sqrt(mean((test_pred_full(:) - Yt_test(1:500,:)).^2));

% Create TKM-pruned version
A_val_tkm = W_eff;
A_val_tkm(:, cross_temporal_mask) = 0;
A_zero = zeros(N, n_mono, 'single');

model_tkm = PhNNModel(dim_temporal, N, mono_temp, single(A_val_tkm), A_zero);
test_pred_tkm = model_tkm.forward(Xt_test(1:500, :));
test_rmse_tkm = sqrt(mean((test_pred_tkm(:) - Yt_test(1:500,:)).^2));

fprintf('  Full model (unedited):    RMSE = %.6f\n', test_rmse_full);
fprintf('  TKM-pruned (cross=0):     RMSE = %.6f\n', test_rmse_tkm);
if test_rmse_full > 1e-10
    fprintf('  RMSE change:              %+.2f%%\n', ...
        (test_rmse_tkm - test_rmse_full)/test_rmse_full*100);
end

% ---- Generate figures ----
fprintf('\n[6/6] Generating TKM validation plots...\n');

figure('Position', [100, 100, 1300, 900]);

% (a) Histogram of weight magnitudes
subplot(2, 2, 1);
bins = logspace(-8, 1, 80);
h1 = histogram(within_weights, bins, 'Normalization', 'pdf', ...
    'FaceColor', [0 0.45 0.74], 'FaceAlpha', 0.6, 'DisplayName', ...
    sprintf('Within-temporal (mean=%.4f, n=%d)', mean_within, n_within));
hold on;
h2 = histogram(cross_weights, bins, 'Normalization', 'pdf', ...
    'FaceColor', [0.85 0.33 0.10], 'FaceAlpha', 0.6, 'DisplayName', ...
    sprintf('Cross-temporal (mean=%.4f, n=%d)', mean_cross, n_cross));
set(gca, 'XScale', 'log');
xlabel('|Weight|', 'FontSize', 11);
ylabel('Density', 'FontSize', 11);
title('Distribution of Learned Weight Magnitudes');
legend('FontSize', 9, 'Location', 'northeast');
grid on;

% (b) Per-output weight ratio
subplot(2, 2, 2);
bar(0:N-1, ratios, 'FaceColor', [0 0.45 0.74], 'EdgeColor', 'k');
hold on;
yline(1.0, 'r--', 'LineWidth', 1.5);
yline(mean(ratios), 'g-', 'LineWidth', 1.5);
xlabel('Output Dimension', 'FontSize', 11);
ylabel('Cross / Within Mean |W| Ratio', 'FontSize', 11);
title('Cross-Temporal Weight Significance per Output Dimension');
legend({'Equal (ratio=1)', sprintf('Mean = %.4f', mean(ratios))}, 'FontSize', 9);
grid on;

% (c) Cumulative distribution
subplot(2, 2, 3);
sorted_cross = sort(cross_weights);
sorted_within = sort(within_weights);
plot(linspace(0, 100, length(cross_weights)), sorted_cross, ...
    'Color', [0.85 0.33 0.10], 'LineWidth', 2, ...
    'DisplayName', sprintf('Cross-temporal (n=%d)', n_cross));
hold on;
plot(linspace(0, 100, length(within_weights)), sorted_within, ...
    'Color', [0 0.45 0.74], 'LineWidth', 2, ...
    'DisplayName', sprintf('Within-temporal (n=%d)', n_within));
xlabel('Percentile', 'FontSize', 11);
ylabel('|Weight|', 'FontSize', 11);
title('Cumulative Distribution of Weight Magnitudes');
legend('FontSize', 9);
grid on;

% (d) Summary stats bar chart
subplot(2, 2, 4);
stats_labels = {'Mean', 'Median', 'Std', '90th %ile', '99th %ile'};
cross_stats = [mean_cross, median_cross, std_cross, ...
    prctile(cross_weights, 90), prctile(cross_weights, 99)];
within_stats = [mean_within, median_within, std_within, ...
    prctile(within_weights, 90), prctile(within_weights, 99)];
x = 1:5; w = 0.35;
b1 = bar(x - w/2, within_stats, w, 'FaceColor', [0 0.45 0.74], ...
    'EdgeColor', 'k', 'DisplayName', 'Within-temporal');
hold on;
b2 = bar(x + w/2, cross_stats, w, 'FaceColor', [0.85 0.33 0.10], ...
    'EdgeColor', 'k', 'DisplayName', 'Cross-temporal');
set(gca, 'XTick', x, 'XTickLabel', stats_labels);
ylabel('|Weight| Value');
title('Weight Magnitude Statistics Comparison');
legend('FontSize', 9);
grid on;

sgtitle(['TKM Validation: Cross-Temporal vs Within-Temporal Weights', newline, ...
    '(Unedited PhNN on Lorenz-96 N=40, Temporal Input [x(k),x(k-1)], r=2)'], ...
    'FontWeight', 'bold', 'FontSize', 14);
saveas(gcf, 'fig/Lorenz96_TKM_Validation.png');
close;

% ---- Final verdict ----
fprintf('\n%s\n', repmat('=', 1, 60));
fprintf('TKM VALIDATION VERDICT\n');
fprintf('%s\n', repmat('=', 1, 60));
ratio_mean = mean_cross / (mean_within + 1e-10);
ratio_median = median_cross / (median_within + 1e-10);
pct_near_zero = mean(cross_weights < 1e-3) * 100;

fprintf('\n  Cross/Within mean weight ratio:   %.4f\n', ratio_mean);
fprintf('  Cross/Within median weight ratio: %.4f\n', ratio_median);
fprintf('  %% cross-temporal weights < 1e-3:  %.1f%%\n', pct_near_zero);

if test_rmse_full > 1e-10
    rmse_change = (test_rmse_tkm - test_rmse_full) / test_rmse_full * 100;
    fprintf('  RMSE change from TKM pruning:     %+.2f%%\n', rmse_change);
end

if ratio_mean < 0.1
    fprintf('\n  VERDICT: PASS\n');
    fprintf('  Cross-temporal weights are >10x smaller than within-temporal.\n');
    fprintf('  TKM pruning is EMPIRICALLY JUSTIFIED for Lorenz-96.\n');
    fprintf('  The first-order Markov property is confirmed.\n');
elseif ratio_mean < 0.5
    fprintf('\n  VERDICT: PARTIAL\n');
    fprintf('  Cross-temporal weights are noticeably smaller but not negligible.\n');
    fprintf('  TKM pruning should be applied with caution.\n');
else
    fprintf('\n  VERDICT: FAIL\n');
    fprintf('  Cross-temporal weights are comparable to within-temporal.\n');
    fprintf('  TKM pruning may introduce significant approximation error.\n');
end

end
