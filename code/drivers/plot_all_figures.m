function plot_all_figures()
%% PLOT_ALL_FIGURES  Regenerate the 12 report figures from saved .mat data.
%   Loads results/*.mat (produced by save_all_results.m) and writes
%   fig/FigN_EnglishName.png at 300 DPI. No simulation is re-run.
%
%   Figure order matches the order figures appear in the report:
%     Fig1  Lorenz96vsOscillator (cross-experiment)
%     Fig2  WeightMatrices
%     Fig3  TrainingCurves
%     Fig4  Complexity
%     Fig5  Predictions
%     Fig6  SINDyCoefficients
%     Fig7  RMSEComparison
%     Fig8  OscillatorTraining
%     Fig9  OscillatorRMSE
%     Fig10 Regulation
%     Fig11 FinalNorm
%     Fig12 Ablation
%
%   Figures removed from the paper (generation commented out below):
%     RMSEvsHorizon (old Fig4), SINDyCV (old Fig8),
%     CoefficientRecovery (old Fig9), SINDyPerOutput (old Fig11)

    base = fileparts(fileparts(fileparts(mfilename('fullpath'))));  % repo root (two folders up from code/drivers)
    addpath(base);
    res_dir = fullfile(base, 'results');
    fig_dir = fullfile(base, 'fig');
    if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end

    L = load(fullfile(res_dir, 'lorenz96_results.mat'));   % .results .models .test_traj
    S = load(fullfile(res_dir, 'sindy_results.mat'));      % .results .models .figdata
    O = load(fullfile(res_dir, 'oscillator_results.mat')); % .results
    C = load(fullfile(res_dir, 'control_results.mat'));    % .results_ctrl .models

    fig1_cross_experiment(L.results, O.results, fig_dir);
    fig2_weight_matrices(L.models, fig_dir);
    fig3_training_curves(L.results, fig_dir);
    % fig4_rmse_horizon(L.results, fig_dir);              % removed from paper
    fig4_complexity(L.results, fig_dir);
    fig5_predictions(L.models, L.test_traj, fig_dir);
    fig6_sindy_coeffs(S.figdata, S.models, fig_dir);
    % fig8_sindy_cv(S.figdata, fig_dir);                  % removed from paper
    % fig9_coeff_recovery(S.figdata, S.models, fig_dir);  % removed from paper
    fig7_rmse_comparison(S.figdata, fig_dir);
    % fig11_sindy_peroutput(S.figdata, fig_dir);          % removed from paper
    fig8_oscillator_training(O.results, fig_dir);
    fig9_oscillator_rmse(O.results, fig_dir);
    fig10_regulation(C.results_ctrl, fig_dir);
    fig11_finalnorm(C.results_ctrl, fig_dir);
    fig12_ablation(C.results_ctrl, fig_dir);

    fprintf('\nAll 12 figures written to %s\n', fig_dir);
end

%% ========================================================================
%  Shared helpers
%  ========================================================================

function save_fig(name, fig_dir)
    % Hide the axes hover-toolbar so it is never captured by exportgraphics.
    axs = findall(gcf, 'Type', 'axes');
    for k = 1:numel(axs)
        try
            axs(k).Toolbar.Visible = 'off';
        catch
        end
    end
    out = fullfile(fig_dir, name);
    exportgraphics(gcf, out, 'Resolution', 300);
    close(gcf);
    fprintf('  saved %s\n', name);
end

function set_log10_axis(ax)
    % Force a clean log10 y-axis: a tick at every decade, labelled 10^{n}.
    ax.YScale = 'log';
    yl = ylim(ax);
    decades = ceil(log10(yl(1))) : floor(log10(yl(2)));
    ax.YTick = 10.^decades;
    ax.YTickLabel = arrayfun(@(d) sprintf('10^{%d}', d), decades, 'UniformOutput', false);
    ax.YMinorTick = 'on';
end

function C = l96_colors()
    C = struct('unedited', [0.85 0.33 0.10], 'pim', [0 0.45 0.74], ...
               'tkm', [0.93 0.69 0.13], 'pim_tkm', [0.49 0.18 0.56]);
end

function L = l96_labels()
    L = struct('unedited', 'Unedited PhNN', 'pim', 'PIM-Edited PhNN', ...
               'tkm', 'TKM-Edited PhNN', 'pim_tkm', 'PIM+TKM Edited PhNN');
end

function C = ctrl_colors()
    C = struct('lqr', [0 0 0], 'unedited', [0.85 0.33 0.10], ...
               'pim', [0 0.45 0.74], 'tkm', [0.93 0.69 0.13], ...
               'pim_tkm', [0.49 0.18 0.56], 'random', [0.64 0.08 0.18], ...
               'mlp', [0.47 0.67 0.19]);
end

function L = ctrl_labels()
    L = struct('lqr', 'LQR (Ground Truth)', 'unedited', 'Unedited PhNN', ...
               'pim', 'PIM-Edited PhNN', 'tkm', 'TKM-Edited PhNN', ...
               'pim_tkm', 'PIM+TKM PhNN', 'random', 'Random-Pruned PhNN', ...
               'mlp', 'MLP Baseline');
end

%% ========================================================================
%  Fig1: Cross-experiment (Lorenz-96 vs Oscillator)
%  ========================================================================

function fig1_cross_experiment(l96_res, osc_res, fig_dir)
    colors = [0.85 0.33 0.10; 0 0.45 0.74; 0.93 0.69 0.13; 0.49 0.18 0.56];
    order = {'unedited', 'pim', 'tkm', 'pim_tkm'};

    % Lorenz-96 val losses (consistent with tab:main headline results)
    l96_losses = zeros(1, 4);
    for i = 1:4, l96_losses(i) = l96_res.(order{i}).best_val_loss; end

    % Oscillator best val losses
    osc_losses = zeros(1, 4);
    for i = 1:4, osc_losses(i) = osc_res.(order{i}).bv; end

    figure('Position', [100 100 1200 480], 'Color', 'w');

    subplot(1, 2, 1);
    b = bar(l96_losses); b.FaceColor = 'flat';
    for i = 1:4, b.CData(i, :) = colors(i, :); end
    set(gca, 'XTickLabel', {'Unedited', 'PIM', 'TKM', 'PIM+TKM'}, 'YScale', 'log');
    ylabel('Val Loss (log)'); title('(a) Lorenz-96 (dynamics only)'); grid on;
    for i = 1:4
        v = l96_losses(i);
        if v < 0.1
            text(i, v * 1.3, sprintf('%.2e', v), 'HorizontalAlignment', 'center', 'FontSize', 9);
        else
            text(i, v * 1.3, sprintf('%.2f', v), 'HorizontalAlignment', 'center', 'FontSize', 9);
        end
    end

    subplot(1, 2, 2);
    b = bar(osc_losses); b.FaceColor = 'flat';
    for i = 1:4, b.CData(i, :) = colors(i, :); end
    set(gca, 'XTickLabel', {'Unedited', 'PIM', 'TKM', 'PIM+TKM'}, 'YScale', 'log');
    ylabel('Val Loss (log)'); title('(b) Oscillator Network (40D + 5D control)'); grid on;
    for i = 1:4
        text(i, osc_losses(i) * 1.3, sprintf('%.2e', osc_losses(i)), ...
            'HorizontalAlignment', 'center', 'FontSize', 9);
    end

    sgtitle('Cross-Experiment: PIM Effect (Same Ring Topology)', 'FontWeight', 'bold', 'FontSize', 13);
    save_fig('Fig1_Lorenz96vsOscillator.png', fig_dir);
end

%% ========================================================================
%  Fig2: Weight matrices (2x2 layout to match caption)
%  ========================================================================

function fig2_weight_matrices(models, fig_dir)
    names_order = {'unedited', 'pim', 'tkm', 'pim_tkm'};
    display_names = {'Unedited PhNN', 'PIM-Edited', 'TKM-Edited', 'PIM+TKM'};

    figure('Position', [100 100 900 700], 'Color', 'w');
    for ax_idx = 1:4
        subplot(2, 2, ax_idx);
        model = models.(names_order{ax_idx});
        W_eff = abs(model.A_value + model.A_uncertain .* model.W_learn);
        n_show = min(100, size(W_eff, 2));
        imagesc(W_eff(:, 1:n_show)); colormap('hot'); colorbar;
        title(sprintf('%s (%d learnable)', display_names{ax_idx}, model.n_learnable), 'FontSize', 10);
        xlabel('Taylor Monomial Index'); ylabel('Output Dimension');
    end
    sgtitle('Effective Weight Matrix |W_{eff}| (first 100 monomials)', 'FontWeight', 'bold', 'FontSize', 13);
    save_fig('Fig2_WeightMatrices.png', fig_dir);
end

%% ========================================================================
%  Fig3: Training curves
%  ========================================================================

function fig3_training_curves(results, fig_dir)
    colors = l96_colors();
    labels = l96_labels();

    figure('Position', [100 100 1200 450], 'Color', 'w');
    subplot(1, 2, 1); hold on;
    names = fieldnames(results);
    for i = 1:numel(names)
        name = names{i}; r = results.(name); c = colors.(name);
        smoothed = conv(r.train_losses, ones(1, 10) / 10, 'valid');
        plot(smoothed, 'Color', c, 'LineWidth', 2, 'DisplayName', [labels.(name) ' (train)']);
    end
    xlabel('Epoch'); ylabel('Training MSE Loss');
    set(gca, 'YScale', 'log'); title('Training Loss (smoothed)');
    legend('Location', 'best', 'FontSize', 8); grid on;

    subplot(1, 2, 2); hold on;
    for i = 1:numel(names)
        name = names{i}; r = results.(name); c = colors.(name);
        plot(r.val_losses, 'Color', c, 'LineWidth', 2, ...
            'DisplayName', sprintf('%s (val=%.4e)', labels.(name), r.best_val_loss));
    end
    xlabel('Epoch'); ylabel('Validation MSE Loss');
    set(gca, 'YScale', 'log'); title('Validation Loss');
    legend('Location', 'best', 'FontSize', 8); grid on;

    sgtitle('Training and Validation Curves', 'FontWeight', 'bold', 'FontSize', 13);
    save_fig('Fig3_TrainingCurves.png', fig_dir);
end

%% ========================================================================
%  [REMOVED] old Fig4: RMSE vs horizon
%  ========================================================================

function fig4_rmse_horizon(results, fig_dir)
    colors = l96_colors();
    labels = l96_labels();

    figure('Position', [100 100 900 500], 'Color', 'w'); hold on;
    names = fieldnames(results);
    for i = 1:numel(names)
        name = names{i}; r = results.(name); c = colors.(name);
        rmse_vals = r.rmse_by_step;
        if all(isnan(rmse_vals)), continue; end
        % truncate at the last finite step (diverged models produce NaN)
        last_fin = find(~isnan(rmse_vals), 1, 'last');
        rmse_vals = rmse_vals(1:last_fin);
        steps = 0:last_fin-1;
        plot(steps, rmse_vals, 'Color', c, 'LineWidth', 2, 'DisplayName', labels.(name));
        if isfield(r, 'rmse_std')
            std_vals = r.rmse_std(1:last_fin);
            std_vals(isnan(std_vals)) = 0;
            lo = max(rmse_vals - std_vals, 1e-10);
            hi = rmse_vals + std_vals;
            fill([steps, fliplr(steps)], [lo, fliplr(hi)], c, ...
                'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');
        end
    end
    xlabel('Prediction Horizon (steps)', 'FontSize', 12);
    ylabel('RMSE', 'FontSize', 12);
    set(gca, 'YScale', 'log');
    title('Autoregressive Prediction Error vs Horizon (Lorenz-96, N=40)', 'FontSize', 13);
    legend('Location', 'best', 'FontSize', 10); grid on;
    save_fig('Fig4_RMSEvsHorizon.png', fig_dir);
end

%% ========================================================================
%  Fig4: Model complexity
%  ========================================================================

function fig4_complexity(results, fig_dir)
    names_order = {'unedited', 'pim', 'tkm', 'pim_tkm'};
    display_names = {'Unedited', 'PIM-Edited', 'TKM-Edited', 'PIM+TKM'};
    cs = [0.85 0.33 0.10; 0 0.45 0.74; 0.93 0.69 0.13; 0.49 0.18 0.56];

    total = zeros(1, 4); learn = zeros(1, 4); sp = zeros(1, 4);
    for i = 1:4
        r = results.(names_order{i});
        total(i) = r.n_total; learn(i) = r.n_learnable; sp(i) = r.sparsity * 100;
    end

    figure('Position', [100 100 1300 400], 'Color', 'w');
    subplot(1, 3, 1);
    b = bar(total, 'FaceAlpha', 0.7); b.FaceColor = 'flat';
    for i = 1:4, b.CData(i, :) = cs(i, :); end
    set(gca, 'XTickLabel', display_names); title('Weight Matrix Size'); ylabel('Count');
    for i = 1:4, text(i, total(i) + max(total) * 0.02, num2str(total(i)), 'HorizontalAlignment', 'center', 'FontSize', 9); end

    subplot(1, 3, 2);
    b = bar(learn, 'FaceAlpha', 0.7); b.FaceColor = 'flat';
    for i = 1:4, b.CData(i, :) = cs(i, :); end
    set(gca, 'XTickLabel', display_names); title('Learnable Parameters'); ylabel('Count');
    for i = 1:4, text(i, learn(i) + max(learn) * 0.02, num2str(learn(i)), 'HorizontalAlignment', 'center', 'FontSize', 9); end

    subplot(1, 3, 3);
    b = bar(sp, 'FaceAlpha', 0.7); b.FaceColor = 'flat';
    for i = 1:4, b.CData(i, :) = cs(i, :); end
    set(gca, 'XTickLabel', display_names); title('Sparsity (%)'); ylabel('%');
    for i = 1:4, text(i, sp(i) + 1, sprintf('%.1f%%', sp(i)), 'HorizontalAlignment', 'center', 'FontSize', 9); end

    sgtitle('Model Complexity Comparison (Lorenz-96, N=40)', 'FontWeight', 'bold', 'FontSize', 13);
    save_fig('Fig4_Complexity.png', fig_dir);
end

%% ========================================================================
%  Fig5: Multi-step prediction (temporal models omitted, as in the report)
%  ========================================================================

function fig5_predictions(models, test_traj, fig_dir)
    colors = struct('unedited', [0.85 0.33 0.10], 'pim', [0 0.45 0.74], ...
                    'tkm', [0.93 0.69 0.13], 'pim_tkm', [0.49 0.18 0.56]);
    labels = struct('unedited', 'Unedited', 'pim', 'PIM', 'tkm', 'TKM', 'pim_tkm', 'PIM+TKM');

    dims_to_plot = [1 6 11 21 31 36];
    n_steps = 100;
    x0 = single(test_traj(1, :));

    figure('Position', [100 100 1400 700], 'Color', 'w');
    for ax_idx = 1:6
        subplot(2, 3, ax_idx);
        dim = dims_to_plot(ax_idx); hold on;
        plot(test_traj(1:n_steps, dim), 'k-', 'LineWidth', 1.5, 'DisplayName', 'True');
        names = fieldnames(models);
        for i = 1:numel(names)
            name = names{i};
            if any(strcmp(name, {'tkm', 'pim_tkm', 'unedited'})), continue; end
            model = models.(name);
            preds = multi_step_predict(model, x0, n_steps);
            plot(preds(:, dim), '--', 'Color', colors.(name), 'LineWidth', 1.2, 'DisplayName', labels.(name));
        end
        title(sprintf('$x_{%d}$', dim - 1), 'Interpreter', 'latex');
        xlabel('Step'); ylabel('Value');
        legend('Location', 'best', 'FontSize', 7); grid on;
    end
    sgtitle('Multi-Step Autoregressive Prediction (Lorenz-96, N=40)', 'FontWeight', 'bold', 'FontSize', 14);
    save_fig('Fig5_Predictions.png', fig_dir);
end

function predictions = multi_step_predict(model, x0, n_steps)
    predictions = zeros(n_steps, model.dim_out);
    x_current = x0(:)';
    for t = 1:n_steps
        x_next = model.forward(x_current);
        predictions(t, :) = x_next;
        x_current = x_next;
    end
end

%% ========================================================================
%  Fig6: SINDy coefficient structure
%  ========================================================================

function fig6_sindy_coeffs(fd, models, fig_dir)
    Xi_sindy = fd.Xi_sindy; n_nz = fd.n_nz_sindy; sp = fd.sp_sindy;
    A_unc_pim = fd.A_unc_pim;
    m_u = models.unedited; m_p = models.pim;

    figure('Position', [100 100 1400 480], 'Color', 'w');
    subplot(1, 3, 1);
    imagesc(abs(Xi_sindy)); colormap('hot'); colorbar;
    xlabel('Output Dim'); ylabel('Monomial');
    title(sprintf('(a) SINDy |Xi| (%d nonzero, %.1f%% sparse)', n_nz, sp * 100));

    subplot(1, 3, 2);
    imagesc(A_unc_pim); colormap('parula'); colorbar;
    xlabel('Output Dim'); ylabel('Monomial');
    title(sprintf('(b) PIM Mask (%d learnable, %.1f%% sparse)', sum(A_unc_pim(:)), m_p.sparsity * 100));

    subplot(1, 3, 3);
    imagesc(abs(m_u.W_learn)); colormap('hot'); colorbar;
    xlabel('Output Dim'); ylabel('Monomial');
    title(sprintf('(c) PhNN Unedited |W| (%d params)', m_u.n_learnable));

    sgtitle('Coefficient Structure: SINDy vs PIM vs Unedited', 'FontWeight', 'bold', 'FontSize', 13);
    save_fig('Fig6_SINDyCoefficients.png', fig_dir);
end

%% ========================================================================
%  [REMOVED] old Fig8: SINDy CV curve
%  ========================================================================

function fig8_sindy_cv(fd, fig_dir)
    cv = fd.cv_results; best_th = fd.best_th;
    ths = zeros(1, numel(cv)); rms_cv = zeros(1, numel(cv)); nzs = zeros(1, numel(cv));
    for i = 1:numel(cv)
        ths(i) = cv{i}{1}; rms_cv(i) = cv{i}{2}; nzs(i) = cv{i}{3};
    end

    figure('Position', [100 100 800 500], 'Color', 'w');
    yyaxis left;
    semilogx(ths, rms_cv, 'b-o', 'LineWidth', 1.5, 'MarkerSize', 5);
    ylabel('Val RMSE', 'Color', 'b');
    yyaxis right;
    semilogx(ths, nzs, 'r-s', 'LineWidth', 1.5, 'MarkerSize', 5);
    ylabel('Nonzero Coeffs', 'Color', 'r');
    xline(best_th, 'k--', 'LineWidth', 1.5);
    xlabel('Threshold'); title('SINDy Threshold Cross-Validation'); grid on;
    sgtitle('SINDy Hyperparameter Selection', 'FontWeight', 'bold', 'FontSize', 13);
    save_fig('Fig8_SINDyCV.png', fig_dir);
end

%% ========================================================================
%  [REMOVED] old Fig9: Coefficient recovery
%  ========================================================================

function fig9_coeff_recovery(fd, models, fig_dir)
    mono_std = fd.mono_std; Xi_sindy = fd.Xi_sindy;
    N = fd.N; dt = fd.dt;
    m_p = models.pim;

    true_c = zeros(length(mono_std), 1);
    sindy_c_avg = zeros(length(mono_std), 1);
    for i = 1:N
        i_m1 = mod(i-2, N) + 1; i_m2 = mod(i-3, N) + 1; i_p1 = mod(i, N) + 1;
        for h = 1:length(mono_std)
            midx = mono_std{h};
            if length(midx) == 1 && midx(1) == i
                true_c(h) = 1.0 - dt;
            elseif length(midx) == 2 && midx(1) == i_m1 && midx(2) == i_p1
                true_c(h) = dt;
            elseif length(midx) == 2 && midx(1) == i_m1 && midx(2) == i_m2
                true_c(h) = -dt;
            end
        end
        sindy_c_avg = sindy_c_avg + abs(Xi_sindy(:, i));
    end
    sindy_c_avg = sindy_c_avg / N;
    W_pim_avg = mean(abs(m_p.A_value + m_p.A_uncertain .* m_p.W_learn), 1)';

    n_top = min(80, sum(true_c ~= 0) * 3);
    [~, idx_s] = sort(sindy_c_avg, 'descend');
    [~, idx_p] = sort(W_pim_avg, 'descend');
    idx_s = idx_s(1:n_top); idx_p = idx_p(1:n_top);
    xr = 1:n_top; w = 0.35;

    figure('Position', [100 100 1300 480], 'Color', 'w');
    subplot(1, 2, 1);
    bar(xr - w/2, abs(true_c(idx_s)), w, 'FaceColor', 'k', 'DisplayName', 'True (Euler)'); hold on;
    bar(xr + w/2, sindy_c_avg(idx_s), w, 'FaceColor', [0.47 0.67 0.19], 'DisplayName', 'SINDy');
    xlabel('Monomial Rank'); ylabel('|Coefficient|'); title('(a) SINDy vs True'); legend; grid on;

    subplot(1, 2, 2);
    bar(xr - w/2, abs(true_c(idx_p)), w, 'FaceColor', 'k', 'DisplayName', 'True (Euler)'); hold on;
    bar(xr + w/2, W_pim_avg(idx_p), w, 'FaceColor', [0 0.45 0.74], 'DisplayName', 'PhNN+PIM');
    xlabel('Monomial Rank'); ylabel('|Coefficient|'); title('(b) PhNN+PIM vs True'); legend; grid on;

    sgtitle('Coefficient Recovery', 'FontWeight', 'bold', 'FontSize', 13);
    save_fig('Fig9_CoefficientRecovery.png', fig_dir);
end

%% ========================================================================
%  Fig7: RMSE comparison bar
%  ========================================================================

function fig7_rmse_comparison(fd, fig_dir)
    allr = fd.all_results;
    names_plot = {'sindy', 'sindy_temporal', 'unedited', 'pim'};
    colors_plot = [0.47 0.67 0.19; 0.64 0.08 0.18; 0.85 0.33 0.10; 0 0.45 0.74];
    display_plot = {'SINDy (40D)', 'SINDy (80D)', 'PhNN Unedited', 'PhNN + PIM'};

    rmses = zeros(1, 4);
    for i = 1:4, rmses(i) = allr.(names_plot{i}).rmse; end

    figure('Position', [100 100 900 500], 'Color', 'w');
    b = bar(rmses); b.FaceColor = 'flat';
    for i = 1:4, b.CData(i, :) = colors_plot(i, :); end
    set(gca, 'XTickLabel', display_plot, 'YScale', 'log');
    for i = 1:4
        text(i, rmses(i) * 1.3, sprintf('%.4f', rmses(i)), 'HorizontalAlignment', 'center', 'FontSize', 10);
    end
    ylabel('Test RMSE (log)'); title('Prediction Accuracy Comparison'); grid on;
    sgtitle('SINDy vs PhNN: Same Data, Same Library', 'FontWeight', 'bold', 'FontSize', 13);
    save_fig('Fig7_RMSEComparison.png', fig_dir);
end

%% ========================================================================
%  [REMOVED] old Fig11: SINDy terms per output
%  ========================================================================

function fig11_sindy_peroutput(fd, fig_dir)
    Xi = fd.Xi_sindy;
    nz_per = sum(abs(Xi) > 0, 1);
    N = size(Xi, 2);

    figure('Position', [100 100 800 500], 'Color', 'w');
    bar(0:N-1, nz_per, 'FaceColor', [0.47 0.67 0.19], 'EdgeColor', 'k'); hold on;
    yline(14, 'k--', 'LineWidth', 1.5);
    yline(mean(nz_per), '-', 'Color', [0.47 0.67 0.19], 'LineWidth', 1.5);
    xlabel('Output Dim'); ylabel('# Selected Terms'); title('SINDy: Terms per Output');
    legend({'Expected: 14', sprintf('SINDy mean: %.1f', mean(nz_per))}); grid on;
    sgtitle('SINDy Sparsity Pattern', 'FontWeight', 'bold', 'FontSize', 13);
    save_fig('Fig11_SINDyPerOutput.png', fig_dir);
end

%% ========================================================================
%  Fig8: Oscillator training curves
%  ========================================================================

function fig8_oscillator_training(osc_res, fig_dir)
    keys_plot = {'unedited', 'pim', 'tkm', 'pim_tkm', 'mlp'};
    colors_plot = [0.85 0.33 0.10; 0 0.45 0.74; 0.93 0.69 0.13; ...
                   0.49 0.18 0.56; 0.47 0.67 0.19];
    EPOCHS = 150;

    figure('Position', [100 100 1200 480], 'Color', 'w');
    subplot(1, 2, 1); hold on; subplot(1, 2, 2); hold on;
    for i = 1:numel(keys_plot)
        r = osc_res.(keys_plot{i}); c = colors_plot(i, :);
        subplot(1, 2, 1);
        smooth = conv(r.tl, ones(1, 8) / 8, 'valid');
        semilogy(smooth, 'Color', c, 'LineWidth', 1.2, 'DisplayName', r.label);
        subplot(1, 2, 2);
        semilogy(r.vl, 'Color', c, 'LineWidth', 1.5, 'DisplayName', sprintf('%s (%.2e)', r.label, r.bv));
    end
    subplot(1, 2, 1); xlabel('Epoch'); ylabel('MSE (log)'); title('Training Loss');
    xlim([0 EPOCHS]); set_log10_axis(gca); legend('FontSize', 7); grid on;
    subplot(1, 2, 2); xlabel('Epoch'); ylabel('MSE (log)'); title('Validation Loss');
    xlim([0 EPOCHS]); set_log10_axis(gca); legend('FontSize', 7); grid on;
    sgtitle('Oscillator Network: Training Curves (40D state + 5D control)', 'FontWeight', 'bold', 'FontSize', 13);
    save_fig('Fig8_OscillatorTraining.png', fig_dir);
end

%% ========================================================================
%  Fig9: Oscillator RMSE
%  ========================================================================

function fig9_oscillator_rmse(osc_res, fig_dir)
    keys_plot = {'unedited', 'pim', 'tkm', 'pim_tkm', 'random', 'mlp'};
    colors_plot = [0.85 0.33 0.10; 0 0.45 0.74; 0.93 0.69 0.13; ...
                   0.49 0.18 0.56; 0.64 0.08 0.18; 0.47 0.67 0.19];

    rms = zeros(1, numel(keys_plot));
    for i = 1:numel(keys_plot), rms(i) = osc_res.(keys_plot{i}).rmse; end

    figure('Position', [100 100 1000 480], 'Color', 'w');
    b = bar(rms); b.FaceColor = 'flat';
    for i = 1:numel(keys_plot), b.CData(i, :) = colors_plot(i, :); end
    set(gca, 'XTickLabel', cellfun(@(x) osc_res.(x).label, keys_plot, 'UniformOutput', false), 'YScale', 'log');
    for i = 1:numel(rms)
        text(i, rms(i) * 1.1, sprintf('%.4f', rms(i)), 'HorizontalAlignment', 'center', 'FontSize', 8);
    end
    ylabel('Test RMSE (log)'); title('Oscillator Network: Prediction Accuracy'); grid on;
    sgtitle('RMSE Comparison', 'FontWeight', 'bold', 'FontSize', 13);
    save_fig('Fig9_OscillatorRMSE.png', fig_dir);
end

%% ========================================================================
%  Fig10: Closed-loop regulation
%  ========================================================================

function fig10_regulation(ctrl_res, fig_dir)
    COLORS = ctrl_colors(); labels = ctrl_labels();

    figure('Position', [100 100 1300 550], 'Color', 'w');
    subplot(1, 2, 1); hold on;
    for k = {'lqr', 'pim', 'pim_tkm', 'unedited'}
        r = ctrl_res.(k{1}); c = COLORS.(k{1});
        steps = 0:length(r.mean) - 1;
        semilogy(steps, r.mean, 'Color', c, 'LineWidth', 2.0, 'DisplayName', labels.(k{1}));
        fill([steps, fliplr(steps)], [max(r.mean - r.std, 1e-10), fliplr(r.mean + r.std)], ...
            c, 'FaceAlpha', 0.12, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    end
    xlabel('Time Step'); ylabel('||x|| (log)'); title('(a) Edited vs Unedited vs LQR');
    legend('FontSize', 7); grid on; yline(1.0, ':', 'Color', [0.5 0.5 0.5], 'HandleVisibility', 'off');

    subplot(1, 2, 2); hold on;
    for k = {'lqr', 'pim', 'mlp', 'random'}
        r = ctrl_res.(k{1}); c = COLORS.(k{1});
        steps = 0:length(r.mean) - 1;
        semilogy(steps, r.mean, 'Color', c, 'LineWidth', 2.0, 'DisplayName', labels.(k{1}));
        fill([steps, fliplr(steps)], [max(r.mean - r.std, 1e-10), fliplr(r.mean + r.std)], ...
            c, 'FaceAlpha', 0.12, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    end
    xlabel('Time Step'); ylabel('||x|| (log)'); title('(b) PIM vs Baselines');
    legend('FontSize', 7); grid on; yline(1.0, ':', 'Color', [0.5 0.5 0.5], 'HandleVisibility', 'off');

    sgtitle('Closed-Loop Regulation', 'FontWeight', 'bold', 'FontSize', 13);
    save_fig('Fig10_Regulation.png', fig_dir);
end

%% ========================================================================
%  Fig11: Final state norm
%  ========================================================================

function fig11_finalnorm(ctrl_res, fig_dir)
    COLORS = ctrl_colors();
    keys_bar = {'pim', 'pim_tkm', 'tkm', 'unedited', 'mlp', 'random'};
    dns_bar = {'PIM', 'PIM+TKM', 'TKM', 'Unedited', 'MLP', 'Random'};

    n_bar = numel(keys_bar);
    finals = zeros(1, n_bar); cols_bar = zeros(n_bar, 3);
    for i = 1:n_bar
        r = ctrl_res.(keys_bar{i});
        finals(i) = r.final_val;
        cols_bar(i, :) = COLORS.(keys_bar{i});
    end

    figure('Position', [100 100 1000 480], 'Color', 'w');
    b = bar(finals); b.FaceColor = 'flat';
    for i = 1:n_bar, b.CData(i, :) = cols_bar(i, :); end
    set(gca, 'XTickLabel', dns_bar);
    ylim([0, max(finals) * 1.15]);
    for i = 1:n_bar
        text(i, finals(i) * 1.02, sprintf('%.2f', finals(i)), ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
            'FontSize', 11, 'FontWeight', 'bold');
    end
    ylabel('Final ||x||'); title('Regulation Performance (lower is better)'); grid on;
    sgtitle('Final State Norm: Lower is Better', 'FontWeight', 'bold', 'FontSize', 13);
    save_fig('Fig11_FinalNorm.png', fig_dir);
end

%% ========================================================================
%  Fig12: Ablation study
%  ========================================================================

function fig12_ablation(ctrl_res, fig_dir)
    COLORS = ctrl_colors();
    abl_keys = {'unedited', 'tkm', 'pim_tkm', 'pim'};
    abl_dns = {'Unedited', 'TKM', 'PIM+TKM', 'PIM'};

    abl_vals = zeros(1, 4); abl_cols = zeros(4, 3);
    for i = 1:4
        abl_vals(i) = ctrl_res.(abl_keys{i}).final_val;
        abl_cols(i, :) = COLORS.(abl_keys{i});
    end

    figure('Position', [100 100 850 480], 'Color', 'w');
    b = bar(abl_vals); b.FaceColor = 'flat';
    for i = 1:4, b.CData(i, :) = abl_cols(i, :); end
    set(gca, 'XTickLabel', abl_dns);
    for i = 1:4
        text(i, abl_vals(i) * 1.1, sprintf('%.4f', abl_vals(i)), 'HorizontalAlignment', 'center', 'FontSize', 10);
    end
    ylabel('Final ||x||'); title('Component Contributions'); grid on;
    for i = 2:4
        imp = (abl_vals(1) - abl_vals(i)) / abl_vals(1) * 100;
        text(i, abl_vals(i) * 1.5, sprintf('%.1f%%', imp), 'HorizontalAlignment', 'center', 'FontSize', 8, 'Color', [0 0.5 0]);
    end
    sgtitle('Ablation Study', 'FontWeight', 'bold', 'FontSize', 13);
    save_fig('Fig12_Ablation.png', fig_dir);
end
