%% COMPARE_SINGLE_VS_HYBRID16  Detailed head-to-head comparison:
%   Degree-2 SINGLE (monolithic PIM-edited, N=1)  vs  Hybrid with degree-1
%   sub-PhNs at N=16, on the coupled oscillator network.
%
%   Metrics: model structure (monomials / parameters / sparsity),
%   training computation (epochs, cumulative wall time, epoch-steps),
%   inference computation (analytic FLOPs, measured forward latency),
%   closed-loop control computation (shooting MPC prediction time per step,
%   LQR gain-design time) and control performance (final ||x||, both the
%   fresh timed runs and the 30-trial reference from the full experiment).
%
%   Loads the trained models from results/oscillator_hybrid_results.mat (the
%   full run, deterministic rng), so every number here is consistent with the
%   published experiment.  All helper functions are reproduced from
%   oscillator_hybrid_control.m so the script is self-contained.
%
%   Saves: results/compare_single_vs_hybrid16.mat, fig/OscHyb_SingleVsHybrid16.png

clear; close all;
addpath(pwd);

%% =========================================================================
%  [1] Load full-run models
%  =========================================================================
S  = load(fullfile('results', 'oscillator_hybrid_results.mat'));
r  = S.results;
m_pim = r.res_pim.model;                      % degree-2 monolithic PIM (N=1)
i16   = find([r.hybrid.N] == 16, 1);
h16   = r.hybrid(i16);                        % Hybrid degree-1, N=16

dim_state = 2 * r.meta.N_MASSES; dim_control = r.meta.M_ACTUATORS;
dim_input = dim_state + dim_control; dim_output = dim_state;
Q_lqr = eye(dim_state) * 0.1; R_lqr = eye(dim_control) * 0.01;

fprintf('%s\n', repmat('=', 1, 100));
fprintf('HEAD-TO-HEAD: degree-2 SINGLE PIM (N=1)  vs  Hybrid degree-1 sub-PhNs (N=16)\n');
fprintf('%s\n', repmat('=', 1, 100));

%% =========================================================================
%  [2] Model structure (static computational size)
%  =========================================================================
nmono_m = r.meta.n_mono;      % 1080  (degree 2)
nmono_s = r.meta.n_mono_sub;  % 45    (degree 1)
nb16    = h16.N;              % 16 partitions

mono.n_name      = 'Single PIM (r=2, N=1)';
mono.n_monomials = nmono_m;
mono.n_total_w   = dim_output * nmono_m;      % full weight matrix entries
mono.n_learnable = m_pim.n_learnable;          % PIM-revealed (masked) params
mono.sparsity    = m_pim.sparsity;
mono.w_eff_nnz   = nnz(m_pim.A_value + m_pim.A_uncertain .* m_pim.W_learn);

h16.n_name      = 'Hybrid r=1, N=16';
h16.n_monomials = nmono_s;                     % per sub-PhN
h16.n_total_w   = dim_output * nmono_s * nb16; % stored across 16 sub-PhNs
h16.n_learnable = h16.n_learnable;             % cumulative learnable
h16.sparsity    = h16.sparsity;
h16.w_eff_nnz   = 0;
for i = 1:nb16
    h16.w_eff_nnz = h16.w_eff_nnz + nnz(h16.models{i}.A_value + ...
        h16.models{i}.A_uncertain .* h16.models{i}.W_learn);
end
h16.n_pca       = r.pca_info.n_p;              % PCA projection stored for switching
h16.pca_entries = r.pca_info.n_p * dim_input;  % feature projector

fprintf('\n--- [2] Model structure ---\n');
fprintf('%-28s %-20s %-20s\n', '', mono.n_name, h16.n_name);
fprintf('%-28s %-20d %-20d\n', 'Monomials (per model/sub)', mono.n_monomials, h16.n_monomials);
fprintf('%-28s %-20d %-20d\n', 'Weight-matrix entries', mono.n_total_w, h16.n_total_w);
fprintf('%-28s %-20d %-20d\n', 'Learnable (PIM-revealed)', mono.n_learnable, h16.n_learnable);
fprintf('%-28s %-20.2f %-20.2f\n', 'Effective W nonzeros', mono.w_eff_nnz, h16.w_eff_nnz);
fprintf('%-28s %-20.1f %-20.1f\n', 'PIM sparsity (%)', mono.sparsity*100, h16.sparsity*100);

%% =========================================================================
%  [3] Training computation  (cumulative across ALL sub-PhNs)
%  =========================================================================
mono.train_time_s  = r.res_pim.train_time;
mono.train_epochs  = r.res_pim.epochs_used;
h16.train_time_s   = h16.train_time;           % cumulative, all 16 sub-PhNs
h16.train_epochs   = h16.epochs_used;          % max over sub-PhNs
% Cumulative gradient steps: sum over sub-PhNs of epochs_i * ceil(n_i/256).
% n_i per partition is not stored, so re-derive by re-running the exact
% ME-bisect assignment on the regenerated training set (deterministic rng 42).
rng(42);
N_MASSES = r.meta.N_MASSES; M_ACTUATORS = r.meta.M_ACTUATORS;
osc0 = setup_oscillator(N_MASSES, M_ACTUATORS);
K0   = design_lqr(osc0.A_mat, osc0.B_mat, Q_lqr, R_lqr);
NS   = 12000; UB0 = r.meta.ub;
X_all = zeros(NS, dim_input, 'single'); Y_all = zeros(NS, dim_output, 'single');
for i = 1:NS
    x = randn(dim_state, 1) * 2.0;
    u = -K0 * x + randn(dim_control, 1) * 0.05;
    u = max(min(u, UB0), -UB0);
    X_all(i, :) = [x; u]'; Y_all(i, :) = osc_step(osc0, x, u)';
end
prm = randperm(NS); X_all = X_all(prm, :); Y_all = Y_all(prm, :);
n_tr = floor(NS * 0.7);
Xtr = X_all(1:n_tr, :); Ytr = Y_all(1:n_tr, :);
% PCA on the SAME train set
F_tr = (double(Xtr) - r.pca_info.mu) ./ r.pca_info.rangev * r.pca_info.Q';
boxes16 = h16.boxes;
b_tr = assign_mode(F_tr, boxes16);
n_i = accumarray(b_tr, 1, [nb16 1]);           % samples per partition
steps_i = ceil(n_i / 256);
h16.train_steps = sum(n_i .* 200);             % each sub-PhN ran all 200 epochs
h16.train_batches = sum(steps_i * 200);        % cumulative mini-batch steps
mono.train_batches = mono.train_epochs * ceil(n_tr / 256);

fprintf('\n--- [3] Training computation (cumulative over all sub-PhNs) ---\n');
fprintf('%-28s %-20s %-20s\n', '', mono.n_name, h16.n_name);
fprintf('%-28s %-20.3f %-20.3f\n', 'Wall time (s)', mono.train_time_s, h16.train_time_s);
fprintf('%-28s %-20d %-20d\n', 'Epochs (max over subs)', mono.train_epochs, h16.train_epochs);
fprintf('%-28s %-20d %-20d\n', 'Mini-batch steps (cumul.)', mono.train_batches, h16.train_batches);

%% =========================================================================
%  [4] Inference computation: analytic FLOPs + measured forward latency
%  =========================================================================
% Analytic per-input FLOPs (MAC = 2 flops)
n_p = r.pca_info.n_p;
flops_mono = 2*(nmono_m*dim_output) + (nmono_m - dim_input);   % matvec + deg-2 products
macs_pca   = n_p * dim_input;                                   % PCA projector
assign_ops = nb16*(30*3 + 29) + (nb16 - 1);                     % Chebyshev over boxes
macs_sub   = nmono_s * dim_output;                              % active sub-PhN matvec
flops_h16  = 2*macs_pca + assign_ops + 2*macs_sub + dim_output + dim_output;
mono.flops_fwd  = flops_mono;
h16.flops_fwd   = flops_h16;

% Measured latency (single input, single precision, mean over repeats)
N_MEAS = 2000; N_REP = 20; N_WARM = 3;
XU = rand(N_MEAS, dim_input, 'single');
Xm = XU(:, 1:dim_state); Um = XU(:, dim_state+1:end);
pred_pim = @(X, U) m_pim.forward(single([X U]));
pred_h16 = @(X, U) osc_hyb_predict(h16.models, h16.boxes, r.pca_info, ...
    h16.means_x, h16.means_y, X, U);
for w = 1:N_WARM, pred_pim(Xm, Um); pred_h16(Xm, Um); end
t = tic; for k = 1:N_REP, pred_pim(Xm, Um); end
mono.fwd_us_per_input = toc(t) / N_REP / N_MEAS * 1e6;
t = tic; for k = 1:N_REP, pred_h16(Xm, Um); end
h16.fwd_us_per_input = toc(t) / N_REP / N_MEAS * 1e6;

fprintf('\n--- [4] Inference computation (per single input) ---\n');
fprintf('%-28s %-20s %-20s\n', '', mono.n_name, h16.n_name);
fprintf('%-28s %-20.0f %-20.0f\n', 'Analytic FLOPs', mono.flops_fwd, h16.flops_fwd);
fprintf('%-28s %-20.3f %-20.3f\n', 'Measured latency (us)', mono.fwd_us_per_input, h16.fwd_us_per_input);
fprintf('%-28s %-20.2f %-20.2f\n', 'FLOP reduction (x)', mono.flops_fwd/h16.flops_fwd, 1);
fprintf('%-28s %-20.2f %-20.2f\n', 'Latency speedup (x)', mono.fwd_us_per_input/h16.fwd_us_per_input, 1);

%% =========================================================================
%  [5] Accuracy (reference from the full experiment, shared test set)
%  =========================================================================
acc_names = r.accuracy.names;                 % ordered: ued, pim, hyb2, hyb4, hyb8, hyb16, ord
rmse_all  = r.accuracy.rmse_test;
roll_all  = r.accuracy.roll;
mono.test_rmse  = rmse_all(2);
mono.rmse5      = roll_all(2, 5);
mono.rmse10     = roll_all(2, 10);
h16.test_rmse   = rmse_all(2 + 4);            % slot 6 = Hybrid N=16
h16.rmse5       = roll_all(6, 5);
h16.rmse10      = roll_all(6, 10);

fprintf('\n--- [5] Accuracy (shared test set, single-step + closed-loop rollout) ---\n');
fprintf('%-28s %-20s %-20s\n', '', mono.n_name, h16.n_name);
fprintf('%-28s %-20.4e %-20.4e\n', 'Single-step test RMSE', mono.test_rmse, h16.test_rmse);
fprintf('%-28s %-20.4f %-20.4f\n', 'Closed-loop rollout RMSE@5', mono.rmse5, h16.rmse5);
fprintf('%-28s %-20.4f %-20.4f\n', 'Closed-loop rollout RMSE@10', mono.rmse10, h16.rmse10);

%% =========================================================================
%  [6] Closed-loop control: performance + computation time
%  =========================================================================
fprintf('\n--- [6] Closed-loop control (true plant) ---\n');
osc = setup_oscillator(N_MASSES, M_ACTUATORS);
K_lqr = design_lqr(osc.A_mat, osc.B_mat, Q_lqr, R_lqr);

% 6.1 Certainty-equivalence LQR: gain-design time (model-dependent part)
t = tic; Wm = m_pim.A_value + m_pim.A_uncertain .* m_pim.W_learn;
K_pim = osc_get_lqr_gain(Wm, dim_state, dim_control, Q_lqr, R_lqr);
mono.lqr_gain_ms = toc(t) * 1e3;
t = tic; Wh = osc_hyb_linpart(h16.models, h16.boxes, r.pca_info, h16.means_x, dim_input);
K_h16 = osc_get_lqr_gain(Wh, dim_state, dim_control, Q_lqr, R_lqr);
h16.lqr_gain_ms = toc(t) * 1e3;
mono.lqr_final = r.ctrl_lqr.pim.final;
h16.lqr_final  = r.ctrl_lqr.hyb16.final;
mono.lqr_normK = norm(K_pim);
h16.lqr_normK  = norm(K_h16);

fprintf('  [6.1] Certainty-equivalence LQR (500 steps x 30 trials, ref.)\n');
fprintf('%-28s %-20s %-20s\n', '', mono.n_name, h16.n_name);
fprintf('%-28s %-20.4f %-20.4f\n', 'Final ||x||', mono.lqr_final, h16.lqr_final);
fprintf('%-28s %-20.4f %-20.4f\n', '||K_hat||', mono.lqr_normK, h16.lqr_normK);
fprintf('%-28s %-20.3f %-20.3f\n', 'Gain-design time (ms)', mono.lqr_gain_ms, h16.lqr_gain_ms);

% 6.2 Shooting MPC: timed run on identical trials, prediction time isolated
fprintf('\n  [6.2] Shooting MPC (400 cand/step, H=5, 60 steps; fresh timed run)\n');
N_TRIALS_MEAS = 6; N_STEPS_SHOOT = 60; N_CAND = 400; H_SHOOT = 5; UB = 2.0; LAM = 0.01;
rng(777); x0_list = randn(N_TRIALS_MEAS, dim_state) * 3.0;
for which = 1:2
    if which == 1
        pred = pred_pim; nm = 'mono';
    else
        pred = pred_h16; nm = 'h16';
    end
    fin = zeros(N_TRIALS_MEAS, 1); t_pred = 0; t_wall = 0;
    for tr = 1:N_TRIALS_MEAS
        tw = tic;
        [traj, ~, tp] = osc_shoot_timed(pred, osc, x0_list(tr, :)', N_STEPS_SHOOT, ...
            N_CAND, H_SHOOT, UB, LAM, dim_state, dim_control, 1000 + tr);
        t_wall = t_wall + toc(tw); t_pred = t_pred + tp;
        fin(tr) = norm(traj(end, :));
    end
    if which == 1
        mono.shoot_final6 = mean(fin);
        mono.shoot_pred_total_s = t_pred;
        mono.shoot_wall_total_s = t_wall;
    else
        h16.shoot_final6 = mean(fin);
        h16.shoot_pred_total_s = t_pred;
        h16.shoot_wall_total_s = t_wall;
    end
end
mono.shoot_pred_per_step_ms = mono.shoot_pred_total_s / (N_TRIALS_MEAS*N_STEPS_SHOOT) * 1e3;
h16.shoot_pred_per_step_ms  = h16.shoot_pred_total_s  / (N_TRIALS_MEAS*N_STEPS_SHOOT) * 1e3;
mono.shoot_pred_per_fwd_us  = mono.shoot_pred_total_s / (N_TRIALS_MEAS*N_STEPS_SHOOT*N_CAND*H_SHOOT) * 1e6;
h16.shoot_pred_per_fwd_us   = h16.shoot_pred_total_s  / (N_TRIALS_MEAS*N_STEPS_SHOOT*N_CAND*H_SHOOT) * 1e6;
mono.shoot_final = r.ctrl_shoot.pim.final;      % 30-trial reference
h16.shoot_final  = r.ctrl_shoot.hyb16.final;

fprintf('%-28s %-20s %-20s\n', '', mono.n_name, h16.n_name);
fprintf('%-28s %-20.4f %-20.4f\n', 'Final ||x|| (30-trial ref.)', mono.shoot_final, h16.shoot_final);
fprintf('%-28s %-20.4f %-20.4f\n', 'Final ||x|| (6-trial meas.)', mono.shoot_final6, h16.shoot_final6);
fprintf('%-28s %-20.3f %-20.3f\n', 'Total prediction time (s)', mono.shoot_pred_total_s, h16.shoot_pred_total_s);
fprintf('%-28s %-20.3f %-20.3f\n', 'Total wall time (s)', mono.shoot_wall_total_s, h16.shoot_wall_total_s);
fprintf('%-28s %-20.3f %-20.3f\n', 'Prediction per step (ms)', mono.shoot_pred_per_step_ms, h16.shoot_pred_per_step_ms);
fprintf('%-28s %-20.3f %-20.3f\n', 'Pred per rollout fwd (us)', mono.shoot_pred_per_fwd_us, h16.shoot_pred_per_fwd_us);

%% =========================================================================
%  [7] Summary ratios + figure + save
%  =========================================================================
ratio = struct( ...
    'flops',      mono.flops_fwd / h16.flops_fwd, ...
    'latency',    mono.fwd_us_per_input / h16.fwd_us_per_input, ...
    'train_time', mono.train_time_s / h16.train_time_s, ...
    'params',     mono.n_learnable / h16.n_learnable, ...
    'test_rmse',  mono.test_rmse / h16.test_rmse, ...
    'shoot_perstep', mono.shoot_pred_per_step_ms / h16.shoot_pred_per_step_ms, ...
    'shoot_final',   h16.shoot_final / mono.shoot_final);

fprintf('\n%s\n', repmat('=', 1, 100));
fprintf('[7] SUMMARY: single(mono) / hybrid16  ( >1 means mono is costlier/worse )\n');
fprintf('%s\n', repmat('=', 1, 100));
fprintf('  Analytic FLOPs per forward   : %.2fx  (mono %.0f vs hyb16 %.0f)\n', ratio.flops, mono.flops_fwd, h16.flops_fwd);
fprintf('  Measured forward latency     : %.2fx  (mono %.3f us vs hyb16 %.3f us)\n', ratio.latency, mono.fwd_us_per_input, h16.fwd_us_per_input);
fprintf('  Training wall time (cumul.)  : %.2fx  (mono %.3f s vs hyb16 %.3f s)\n', ratio.train_time, mono.train_time_s, h16.train_time_s);
fprintf('  Learnable parameters         : %.2fx  (mono %d vs hyb16 %d)\n', ratio.params, mono.n_learnable, h16.n_learnable);
fprintf('  Single-step test RMSE        : %.2fx  (mono %.3e vs hyb16 %.3e)\n', ratio.test_rmse, mono.test_rmse, h16.test_rmse);
fprintf('  Shooting pred per step       : %.2fx  (mono %.3f ms vs hyb16 %.3f ms)\n', ratio.shoot_perstep, mono.shoot_pred_per_step_ms, h16.shoot_pred_per_step_ms);
fprintf('  Shooting final ||x|| (hyb/mono): %.2fx  (mono %.3f vs hyb16 %.3f)\n', ratio.shoot_final, mono.shoot_final, h16.shoot_final);

% --- Figure: two-bar comparison across six cost/quality axes ---
figure('Position', [100, 100, 1350, 560]);
axes_titles = {'Learnable params', 'Test RMSE (log)', 'Forward latency (us)', ...
    'Train time (s)', 'Shoot pred/step (ms)', 'Shoot final ||x||'};
vals_m = [mono.n_learnable, mono.test_rmse, mono.fwd_us_per_input, ...
    mono.train_time_s, mono.shoot_pred_per_step_ms, mono.shoot_final];
vals_h = [h16.n_learnable, h16.test_rmse, h16.fwd_us_per_input, ...
    h16.train_time_s, h16.shoot_pred_per_step_ms, h16.shoot_final];
for p = 1:6
    subplot(2, 3, p);
    b = bar([vals_m(p), vals_h(p)], 0.6);
    b.FaceColor = 'flat'; b.CData = [0 0.45 0.74; 0.49 0.18 0.56];
    set(gca, 'XTickLabel', {'Single r=2', 'Hybrid r=1 N=16'});
    if p == 2, set(gca, 'YScale', 'log'); end
    title(axes_titles{p});
    grid on;
    text(1, vals_m(p)*1.05, sprintf('%.4g', vals_m(p)), 'HorizontalAlignment', 'center', 'FontSize', 7);
    text(2, vals_h(p)*1.05, sprintf('%.4g', vals_h(p)), 'HorizontalAlignment', 'center', 'FontSize', 7);
end
sgtitle('Degree-2 Single (N=1)  vs  Hybrid degree-1 N=16', 'FontWeight', 'bold', 'FontSize', 13);
if ~exist('fig', 'dir'), mkdir('fig'); end
saveas(gcf, 'fig/OscHyb_SingleVsHybrid16.png'); close;
fprintf('\n  Figure saved to fig/OscHyb_SingleVsHybrid16.png\n');

% --- Save detailed data ---
c = struct();
c.mono = mono; c.hyb16 = h16; c.ratio = ratio;
c.meta = struct('dim_state', dim_state, 'dim_control', dim_control, ...
    'n_trials_meas', N_TRIALS_MEAS, 'n_cand', N_CAND, 'H', H_SHOOT, ...
    'n_steps_shoot', N_STEPS_SHOOT, 'ub', UB, 'n_fwd_meas', N_MEAS, 'n_rep', N_REP);
if ~exist('results', 'dir'), mkdir('results'); end
save(fullfile('results', 'compare_single_vs_hybrid16.mat'), 'c');
fprintf('  Detailed data saved to results/compare_single_vs_hybrid16.mat\n');
fprintf('%s\n', repmat('=', 1, 100));
fprintf('COMPARISON COMPLETE\n');
fprintf('%s\n', repmat('=', 1, 100));

%% =========================================================================
%  Helper functions (reproduced from oscillator_hybrid_control.m)
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

function K_hat = osc_get_lqr_gain(W_eff, dim_state, dim_control, Q, R)
    A_hat = double(W_eff(:, 1:dim_state));
    B_hat = double(W_eff(:, dim_state+1:dim_state+dim_control));
    K_hat = zeros(dim_control, dim_state);
    if max(abs(eig(A_hat))) > 1.0 + 1e-6 || norm(B_hat, 'fro') < 1e-6
        return;
    end
    try
        K_hat = design_lqr(A_hat, B_hat, Q, R);
    catch
        K_hat = zeros(dim_control, dim_state);
    end
    if any(~isfinite(K_hat(:)))
        K_hat = zeros(dim_control, dim_state);
    end
end

function W_eff = osc_hyb_linpart(models, boxes, pca_info, means_x, dim_input)
    feat0 = (zeros(1, dim_input) - pca_info.mu) ./ pca_info.rangev * pca_info.Q';
    b0 = assign_mode(feat0, boxes);
    if isempty(models{b0})
        idx = find(~cellfun(@isempty, models));
        dc = zeros(1, numel(idx));
        for j = 1:numel(idx), dc(j) = norm(means_x(idx(j), :)); end
        [~, bi] = min(dc);
        b0 = idx(bi);
    end
    m0 = models{b0};
    W_eff = m0.A_value + m0.A_uncertain .* m0.W_learn;
end

function [traj, U, t_pred] = osc_shoot_timed(pred, osc, x0, n_steps, n_cand, H, ub, lam, dim_state, dim_control, rngseed)
    rng(rngseed);
    x = double(x0(:));
    traj = zeros(n_steps + 1, dim_state); traj(1, :) = x';
    U = zeros(n_steps, dim_control);
    Uprev = single(zeros(1, dim_control));
    t_pred = 0;
    for t = 1:n_steps
        U_cand = single(-ub + 2*ub*rand(n_cand, dim_control));
        U_cand(1, :) = single(zeros(1, dim_control));
        U_cand(2, :) = Uprev;
        Xp = repmat(single(x'), n_cand, 1);
        J = zeros(n_cand, 1);
        for h = 1:H
            tt = tic; Xp = pred(Xp, U_cand); t_pred = t_pred + toc(tt);
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
