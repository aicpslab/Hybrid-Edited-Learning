%% OSCILLATOR_SHOOT_EFFECTIVE  Long-horizon shooting MPC with LQR warm-start.
%   REPLACES oscillator_shoot_long.m as the canonical long-horizon control
%   demonstration.
%
%   The plain greedy random-shooting MPC (H=5, no warm-start) provably cannot
%   regulate even with a PERFECT model: it plateaus at ||x||~2-4 over 300-1000
%   steps (see diag_shooting.m).  The fixed controller:
%     * LQR warm-start candidate u = -K_lqr*x in every candidate set
%       (ALL controllers share the SAME warm-start; only the model differs),
%     * longer prediction horizon H = 10.
%   With an accurate model the MPC then tracks the LQR action and regulates
%   below ||x||<1; with an inaccurate model (N=16, ordinary hybrid) the
%   mis-evaluated rollouts reject the warm-start candidate and control
%   degrades -- so model discrimination is preserved.
%
%   Loads trained models from results/oscillator_hybrid_results.mat.
%   Controllers: LQR (true), single PIM (r=2), Hybrid r=1 at N=2/4/8/16,
%   Ordinary Hybrid (unedited).  Identical initial states, cost, saturation,
%   warm-start; only the model differs.
%
%   Saves: results/oscillator_shoot_effective.mat, fig/OscHyb_ShootEffective.png

clear; close all;
S = load(fullfile('results', 'oscillator_hybrid_results.mat'));
r = S.results;

dim_state = 2 * r.meta.N_MASSES; dim_control = r.meta.M_ACTUATORS;
dim_input = dim_state + dim_control; dim_output = dim_state;
Q_lqr = eye(dim_state) * 0.1; R_lqr = eye(dim_control) * 0.01;

N_STEPS = 500; N_TRIALS = 30; N_CAND = 400; H = 10; UB = 2.0; LAM = 0.01;
rng(777); x0_list = randn(N_TRIALS, dim_state) * 3.0;
x0norm_mean = mean(sqrt(sum(x0_list.^2, 2)));
fprintf('  initial ||x_0|| (mean over %d trials) = %.3f\n', N_TRIALS, x0norm_mean);

osc = setup_oscillator(r.meta.N_MASSES, r.meta.M_ACTUATORS);
K_lqr = design_lqr(osc.A_mat, osc.B_mat, Q_lqr, R_lqr);
m_pim = r.res_pim.model;
pred_pim = @(X, U) m_pim.forward(single([X U]));
pred_ord = @(X, U) osc_hyb_predict(r.ord_hybrid.models0, r.ord_hybrid.boxes0, ...
    r.pca_info, r.ord_hybrid.means_x, r.ord_hybrid.means_y, X, U);

% Controller list: lqr + pim + each hybrid in r.hybrid (N=2/4/8/16) + ordhyb
configs = {'lqr', 'pim'};
labels  = {'LQR (optimum)', 'Single PIM (r=2, N=1)'};
for s = 1:numel(r.hybrid)
    configs{end+1} = sprintf('hyb%d', r.hybrid(s).N); %#ok<AGROW>
    labels{end+1}  = sprintf('Hybrid r=1, N=%d', r.hybrid(s).N); %#ok<AGROW>
end
configs{end+1} = 'ordhyb'; %#ok<AGROW>
labels{end+1}  = sprintf('Ordinary Hybrid (unedited N=%d)', r.ord_hybrid.N); %#ok<AGROW>
n_c = numel(configs);

preds = cell(1, n_c); preds{2} = pred_pim;
for s = 1:numel(r.hybrid)
    h = r.hybrid(s);
    preds{2 + s} = @(X, U) osc_hyb_predict(h.models, h.boxes, r.pca_info, ...
        h.means_x, h.means_y, X, U);
end
preds{end} = pred_ord;

res = struct();
res.configs = {configs};
res.labels  = {labels};
res.method  = struct('H', H, 'n_cand', N_CAND, 'warm_start', 'LQR', ...
    'n_steps', N_STEPS, 'n_trials', N_TRIALS, 'lam', LAM, 'ub', UB);
for k = 1:n_c
    fin = zeros(N_TRIALS, 1); step2one = NaN(N_TRIALS, 1);
    mean_traj = zeros(N_TRIALS, N_STEPS + 1);
    for tr = 1:N_TRIALS
        if strcmp(configs{k}, 'lqr')
            x = double(x0_list(tr, :)');
            traj = zeros(N_STEPS + 1, dim_state); traj(1, :) = x';
            for t = 1:N_STEPS
                u = -K_lqr * x; u = max(min(u, UB), -UB);
                x = osc_step(osc, x, u); traj(t + 1, :) = x';
            end
        else
            [traj, ~] = osc_shoot(preds{k}, osc, x0_list(tr, :)', N_STEPS, ...
                N_CAND, H, UB, LAM, K_lqr, dim_state, dim_control, 1000 + tr);
        end
        norms = sqrt(sum(traj.^2, 2));
        fin(tr) = norms(end); mean_traj(tr, :) = norms';
        f2 = find(norms < 1.0, 1);
        if ~isempty(f2), step2one(tr) = f2 - 1; end   % NaN = never reached
    end
    n_ok = sum(~isnan(step2one));
    res.(configs{k}) = struct(...
        'final', mean(fin), 'std', std(fin), ...
        'traj_mean', mean(mean_traj, 1), 'traj_std', std(mean_traj, 0, 1), ...
        'x0norm', x0norm_mean, 'reduction', mean(fin) / x0norm_mean, ...
        'n_reach1', n_ok, 'mean_step2one', mean(step2one(~isnan(step2one)), 'omitnan'), ...
        'final_all', fin);
end

%% Report
fprintf('\n%s\n', repmat('=', 1, 100));
fprintf('LONG-HORIZON SHOOTING MPC + LQR WARM-START (%d steps, %d trials, H=%d, true plant)\n', ...
    N_STEPS, N_TRIALS, H);
fprintf('%s\n', repmat('=', 1, 100));
fprintf('%-32s %-12s %-12s %-12s %-12s %-12s\n', 'Controller', 'Final ||x||', '||x||@120', '||x||@300', 'Reduction', 'Reach<1 (mean step)');
fprintf('%s\n', repmat('-', 1, 100));
for k = 1:n_c
    R = res.(configs{k});
    fprintf('%-32s %-12.4f %-12.3f %-12.3f %-12.1f%% %-12s\n', labels{k}, ...
        R.final, R.traj_mean(121), R.traj_mean(301), ...
        (1 - R.reduction) * 100, ...
        sprintf('%d/%d @%.0f', R.n_reach1, N_TRIALS, R.mean_step2one));
end
fprintf('  (initial ||x_0|| = %.3f; "Reach<1" = trials that drove ||x|| below 1, mean first-passage step)\n', x0norm_mean);

%% Figure
figure('Position', [100, 100, 1300, 500]);
steps = 0:N_STEPS;
hold on;
cols = [[0 0 0]; [0 0.45 0.74]; [0.49 0.18 0.56]; [0.30 0.75 0.30]; ...
        [0.85 0.33 0.10]; [0.85 0.60 0.20]; [0.25 0.35 0.60]; [0.60 0.30 0.30]];
for k = 1:n_c
    R = res.(configs{k});
    plot(steps, R.traj_mean, 'Color', cols(k, :), 'LineWidth', 2.0, 'DisplayName', labels{k});
    fill([steps, fliplr(steps)], [max(R.traj_mean - R.traj_std, 1e-12), fliplr(R.traj_mean + R.traj_std)], ...
        cols(k, :), 'FaceAlpha', 0.10, 'EdgeColor', 'none');
end
plot([0 N_STEPS], [1 1], 'k--', 'LineWidth', 1, 'DisplayName', '||x||=1 (effective bound)');
set(gca, 'YScale', 'log');
xlabel('Control step'); ylabel('||x|| (log)');
title(sprintf('Long-horizon shooting MPC + LQR warm-start (H=%d, %d steps): effective regulation', H, N_STEPS));
legend('Location', 'northeast', 'FontSize', 7); grid on;
sgtitle('Effective regulation: Hybrid r=1 N=2/4/8 vs Single r=2', 'FontWeight', 'bold', 'FontSize', 13);
if ~exist('fig', 'dir'), mkdir('fig'); end
saveas(gcf, 'fig/OscHyb_ShootEffective.png'); close;

if ~exist('results', 'dir'), mkdir('results'); end
save(fullfile('results', 'oscillator_shoot_effective.mat'), 'res');
fprintf('\n  Figure: fig/OscHyb_ShootEffective.png\n');
fprintf('  Data  : results/oscillator_shoot_effective.mat\n');

%% =========================================================================
%  Helpers (same as oscillator_hybrid_control.m / oscillator_shoot_long.m)
%  =========================================================================
function [b_idx] = assign_mode(F, boxes)
    n = size(F, 1); nb = numel(boxes);
    d = zeros(n, nb);
    for b = 1:nb
        lo = boxes{b}.lo; hi = boxes{b}.hi;
        tmp = max(lo - F, F - hi);
        tmp = max(tmp, [], 2);
        d(:, b) = tmp;
    end
    [~, b_idx] = min(d, [], 2);
end

function Xp = osc_hyb_predict(models, boxes, pca_info, means_x, means_y, X, U)
    XU = single([X U]);
    feat = (double(XU) - pca_info.mu) ./ pca_info.rangev * pca_info.Q';
    b = assign_mode(feat, boxes);
    Xp = zeros(size(X), 'single');
    for i = 1:numel(models)
        m = (b == i);
        if any(m)
            if isempty(models{i})
                Xp(m, :) = single(X(m, :));
            else
                Xp(m, :) = models{i}.forward(single(XU(m, :) - means_x(i, :))) + means_y(i, :);
            end
        end
    end
end

function [traj, U] = osc_shoot(pred, osc, x0, n_steps, n_cand, H, ub, lam, K_ws, dim_state, dim_control, rngseed)
    % K_ws: warm-start gain; the candidate u = -K_ws*x is added to every set.
    rng(rngseed);
    x = double(x0(:));
    traj = zeros(n_steps + 1, dim_state); traj(1, :) = x';
    U = zeros(n_steps, dim_control);
    Uprev = single(zeros(1, dim_control));
    for t = 1:n_steps
        U_cand = single(-ub + 2*ub*rand(n_cand, dim_control));
        U_cand(1, :) = single(zeros(1, dim_control));   % no-control candidate
        U_cand(2, :) = Uprev;                            % persistence candidate
        if ~isempty(K_ws)
            u_ws = -K_ws * x; u_ws = max(min(u_ws, ub), -ub);
            U_cand(3, :) = single(u_ws(:)');              % LQR warm-start candidate
        end
        Xp = repmat(single(x'), n_cand, 1);
        J = zeros(n_cand, 1);
        for h = 1:H
            Xp = pred(Xp, U_cand);
            J = J + sum(single(Xp).^2, 2);
        end
        J = J + lam * sum(single(U_cand).^2, 2);
        [~, bi] = min(J);
        u = double(U_cand(bi, :));
        x = osc_step(osc, x, u);
        U(t, :) = u; traj(t + 1, :) = x';
        Uprev = U_cand(bi, :);
    end
end
