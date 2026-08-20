function results = compare_single_vs_hybrid(Ns, out_tag)
%% COMPARE_SINGLE_VS_HYBRID  Head-to-head: degree-2 SINGLE PIM-edited PhNN
%   (N=1) vs Hybrids whose sub-PhNs use the degree-1 Taylor library, at the
%   requested partition counts Ns (e.g. Ns=[4,8]).
%
%   For every model it reports, on the coupled oscillator network:
%     [structure]    monomials, stored weight entries, learnable params,
%                    effective W nonzeros, PIM sparsity
%     [training]     cumulative wall time over ALL sub-PhNs, epochs,
%                    cumulative mini-batch steps
%     [inference]    analytic FLOPs / forward and measured latency / input
%     [accuracy]     single-step test RMSE and closed-loop rollout RMSE@5/@10
%                    (shared test set from the full experiment)
%     [control]      certainty-equivalence LQR (final ||x||, ||K||, gain-design
%                    time) and shooting MPC (final ||x|| 30-trial reference and
%                    fresh timed run, prediction time per step, per rollout
%                    forward, total wall time) on the true plant.
%
%   Loads the trained models from results/oscillator_hybrid_results.mat (the
%   full deterministic run), so all numbers are consistent with the
%   published experiment.  Self-contained: helper functions reproduced from
%   oscillator_hybrid_control.m.
%
%   results = compare_single_vs_hybrid(Ns, out_tag)
%   Saves: results/compare_single_vs_hybrid_<tag>.mat, fig/OscHyb_SingleVsHybrid<tag>.png

if nargin < 1 || isempty(Ns), Ns = [4, 8]; end
if nargin < 2 || isempty(out_tag)
    out_tag = strrep(num2str(Ns), ' ', '');   % e.g. "48"
end

%% =========================================================================
%  [1] Load full-run models
%  =========================================================================
S  = load(fullfile('results', 'oscillator_hybrid_results.mat'));
r  = S.results;
m_pim = r.res_pim.model;                      % degree-2 monolithic PIM (N=1)

dim_state = 2 * r.meta.N_MASSES; dim_control = r.meta.M_ACTUATORS;
dim_input = dim_state + dim_control; dim_output = dim_state;
Q_lqr = eye(dim_state) * 0.1; R_lqr = eye(dim_control) * 0.01;

Ns = Ns(:)';  Ns = sort(unique(Ns));
hybs = struct([]);
for n = 1:numel(Ns)
    idx = find([r.hybrid.N] == Ns(n), 1);
    if isempty(idx), error('No hybrid with N=%d in saved results', Ns(n)); end
    hybs(n).h = r.hybrid(idx);
end

fprintf('%s\n', repmat('=', 1, 108));
fprintf('HEAD-TO-HEAD: degree-2 SINGLE PIM (N=1)  vs  Hybrid degree-1 sub-PhNs (N=%s)\n', ...
    mat2str(Ns));
fprintf('%s\n', repmat('=', 1, 108));

%% =========================================================================
%  [2] Regenerate the (deterministic) training set once for batch-step counts
%  =========================================================================
N_MASSES = r.meta.N_MASSES; M_ACTUATORS = r.meta.M_ACTUATORS;
osc0 = setup_oscillator(N_MASSES, M_ACTUATORS);
K0   = design_lqr(osc0.A_mat, osc0.B_mat, Q_lqr, R_lqr);
NS   = 12000; UB0 = r.meta.ub;
rng(42);
X_all = zeros(NS, dim_input, 'single');
for i = 1:NS
    x = randn(dim_state, 1) * 2.0;
    u = -K0 * x + randn(dim_control, 1) * 0.05;
    u = max(min(u, UB0), -UB0);
    X_all(i, :) = [x; u]';
end
prm = randperm(NS); X_all = X_all(prm, :);
Xtr = X_all(1:floor(NS*0.7), :);
F_tr = (double(Xtr) - r.pca_info.mu) ./ r.pca_info.rangev * r.pca_info.Q';

%% =========================================================================
%  [3] Measure the degree-2 single model
%  =========================================================================
nmono_m = r.meta.n_mono;    % 1080
mono.N          = 1;
mono.n_name     = 'Single PIM (r=2, N=1)';
mono.n_monomials = nmono_m;
mono.n_total_w  = dim_output * nmono_m;
mono.n_learnable = m_pim.n_learnable;
mono.sparsity   = m_pim.sparsity;
mono.w_eff_nnz  = nnz(m_pim.A_value + m_pim.A_uncertain .* m_pim.W_learn);
mono.train_time_s = r.res_pim.train_time;
mono.train_epochs = r.res_pim.epochs_used;
mono.train_batches = mono.train_epochs * ceil(size(Xtr, 1) / 256);
mono.flops_fwd = 2*(nmono_m*dim_output) + (nmono_m - dim_input);
mono.test_rmse = r.accuracy.rmse_test(2);
mono.rmse5     = r.accuracy.roll(2, 5);
mono.rmse10    = r.accuracy.roll(2, 10);
t = tic; Wm = m_pim.A_value + m_pim.A_uncertain .* m_pim.W_learn;
K_pim = osc_get_lqr_gain(Wm, dim_state, dim_control, Q_lqr, R_lqr);
mono.lqr_gain_ms = toc(t) * 1e3;
mono.lqr_final = r.ctrl_lqr.pim.final;
mono.lqr_normK = norm(K_pim);
mono.shoot_final = r.ctrl_shoot.pim.final;

%% =========================================================================
%  [4] Measure each degree-1 hybrid
%  =========================================================================
n_p = r.pca_info.n_p;
for n = 1:numel(hybs)
    h  = hybs(n).h;
    Nb = h.N;
    hybs(n).N = Nb;
    hybs(n).n_name  = sprintf('Hybrid r=1, N=%d', Nb);
    hybs(n).n_monomials = r.meta.n_mono_sub;          % 45, per sub-PhN
    hybs(n).n_total_w  = dim_output * r.meta.n_mono_sub * Nb;
    hybs(n).n_learnable = h.n_learnable;
    hybs(n).sparsity   = h.sparsity;
    hybs(n).w_eff_nnz = 0;
    for i = 1:Nb
        hybs(n).w_eff_nnz = hybs(n).w_eff_nnz + nnz(h.models{i}.A_value + ...
            h.models{i}.A_uncertain .* h.models{i}.W_learn);
    end
    hybs(n).n_pca = n_p;
    hybs(n).pca_entries = n_p * dim_input;
    % training (cumulative)
    hybs(n).train_time_s = h.train_time;
    hybs(n).train_epochs = h.epochs_used;
    b_tr = assign_mode(F_tr, h.boxes);
    n_i = accumarray(b_tr, 1, [Nb 1]);
    hybs(n).train_batches = sum(ceil(n_i / 256) * h.epochs_used);
    % inference FLOPs
    assign_ops = Nb*(30*3 + 29) + (Nb - 1);
    hybs(n).flops_fwd = 2*(n_p*dim_input) + assign_ops + 2*(r.meta.n_mono_sub*dim_output) ...
                        + dim_output + dim_output;
    % accuracy
    slot = 2 + find([r.hybrid.N] == Nb, 1);
    hybs(n).test_rmse = r.accuracy.rmse_test(slot);
    hybs(n).rmse5     = r.accuracy.roll(slot, 5);
    hybs(n).rmse10    = r.accuracy.roll(slot, 10);
    % control: LQR
    t = tic; Wh = osc_hyb_linpart(h.models, h.boxes, r.pca_info, h.means_x, dim_input);
    K_h = osc_get_lqr_gain(Wh, dim_state, dim_control, Q_lqr, R_lqr);
    hybs(n).lqr_gain_ms = toc(t) * 1e3;
    hybs(n).lqr_final = r.ctrl_lqr.(sprintf('hyb%d', Nb)).final;
    hybs(n).lqr_normK = norm(K_h);
    hybs(n).shoot_final = r.ctrl_shoot.(sprintf('hyb%d', Nb)).final;
end

%% =========================================================================
%  [5] Measured forward latency (shared measurement protocol)
%  =========================================================================
N_MEAS = 2000; N_REP = 20; N_WARM = 3;
XU = rand(N_MEAS, dim_input, 'single');
Xm = XU(:, 1:dim_state); Um = XU(:, dim_state+1:end);
pred_pim = @(X, U) m_pim.forward(single([X U]));
pred_h   = cell(1, numel(hybs));
for n = 1:numel(hybs)
    h = hybs(n).h;
    pred_h{n} = @(X, U) osc_hyb_predict(h.models, h.boxes, r.pca_info, ...
        h.means_x, h.means_y, X, U);
end
for w = 1:N_WARM
    pred_pim(Xm, Um);
    for n = 1:numel(hybs), pred_h{n}(Xm, Um); end
end
t = tic; for k = 1:N_REP, pred_pim(Xm, Um); end
mono.fwd_us_per_input = toc(t) / N_REP / N_MEAS * 1e6;
for n = 1:numel(hybs)
    t = tic; for k = 1:N_REP, pred_h{n}(Xm, Um); end
    hybs(n).fwd_us_per_input = toc(t) / N_REP / N_MEAS * 1e6;
end

%% =========================================================================
%  [6] Timed closed-loop shooting MPC (identical trials, prediction isolated)
%  =========================================================================
osc = setup_oscillator(N_MASSES, M_ACTUATORS);
N_TRIALS_MEAS = 6; N_STEPS_SHOOT = 60; N_CAND = 400; H_SHOOT = 5; UB = 2.0; LAM = 0.01;
rng(777); x0_list = randn(N_TRIALS_MEAS, dim_state) * 3.0;

[mono.shoot_final6, mono.shoot_pred_total_s, mono.shoot_wall_total_s] = ...
    run_shoot_timed(pred_pim, osc, x0_list, N_TRIALS_MEAS, N_STEPS_SHOOT, ...
    N_CAND, H_SHOOT, UB, LAM, dim_state, dim_control);
for n = 1:numel(hybs)
    [hybs(n).shoot_final6, hybs(n).shoot_pred_total_s, hybs(n).shoot_wall_total_s] = ...
        run_shoot_timed(pred_h{n}, osc, x0_list, N_TRIALS_MEAS, N_STEPS_SHOOT, ...
        N_CAND, H_SHOOT, UB, LAM, dim_state, dim_control);
end
mono.shoot_pred_per_step_ms = mono.shoot_pred_total_s / (N_TRIALS_MEAS*N_STEPS_SHOOT) * 1e3;
mono.shoot_pred_per_fwd_us  = mono.shoot_pred_total_s / (N_TRIALS_MEAS*N_STEPS_SHOOT*N_CAND*H_SHOOT) * 1e6;
for n = 1:numel(hybs)
    hybs(n).shoot_pred_per_step_ms = hybs(n).shoot_pred_total_s / (N_TRIALS_MEAS*N_STEPS_SHOOT) * 1e3;
    hybs(n).shoot_pred_per_fwd_us  = hybs(n).shoot_pred_total_s / (N_TRIALS_MEAS*N_STEPS_SHOOT*N_CAND*H_SHOOT) * 1e6;
end

%% =========================================================================
%  [7] Report tables
%  =========================================================================
n_h = numel(hybs);
hdr = sprintf('%-30s %-18s', 'metric', mono.n_name);
for n = 1:n_h, hdr = sprintf('%s %-18s', hdr, hybs(n).n_name); end %#ok<AGROW>

fprintf('\n--- [A] Model structure ---\n'); fprintf('%s\n', hdr);
prow('Monomials (per model/sub)', mono.n_monomials, [hybs.n_monomials], '%d', hdr);
prow('Weight-matrix entries', mono.n_total_w, [hybs.n_total_w], '%d', hdr);
prow('Learnable (PIM-revealed)', mono.n_learnable, [hybs.n_learnable], '%d', hdr);
prow('Effective W nonzeros', mono.w_eff_nnz, [hybs.w_eff_nnz], '%d', hdr);
prow('PIM sparsity (%)', mono.sparsity*100, [hybs.sparsity]*100, '%.1f', hdr);

fprintf('\n--- [B] Training computation (cumulative over all sub-PhNs) ---\n');
fprintf('%s\n', hdr);
prow('Wall time (s)', mono.train_time_s, [hybs.train_time_s], '%.3f', hdr);
prow('Epochs (max over subs)', mono.train_epochs, [hybs.train_epochs], '%d', hdr);
prow('Mini-batch steps (cumul.)', mono.train_batches, [hybs.train_batches], '%d', hdr);

fprintf('\n--- [C] Inference computation (per single input) ---\n');
fprintf('%s\n', hdr);
prow('Analytic FLOPs', mono.flops_fwd, [hybs.flops_fwd], '%.0f', hdr);
prow('Measured latency (us)', mono.fwd_us_per_input, [hybs.fwd_us_per_input], '%.3f', hdr);
prow('FLOP speedup vs mono (x)', 1, mono.flops_fwd./[hybs.flops_fwd], '%.2f', hdr);
prow('Latency speedup vs mono (x)', 1, mono.fwd_us_per_input./[hybs.fwd_us_per_input], '%.2f', hdr);

fprintf('\n--- [D] Accuracy (shared test set) ---\n');
fprintf('%s\n', hdr);
prow('Single-step test RMSE', mono.test_rmse, [hybs.test_rmse], '%.4e', hdr);
prow('Rollout RMSE@5', mono.rmse5, [hybs.rmse5], '%.4f', hdr);
prow('Rollout RMSE@10', mono.rmse10, [hybs.rmse10], '%.4f', hdr);

fprintf('\n--- [E] Certainty-equivalence LQR (true plant, 500 steps x 30 trials) ---\n');
fprintf('%s\n', hdr);
prow('Final ||x||', mono.lqr_final, [hybs.lqr_final], '%.4f', hdr);
prow('||K_hat||', mono.lqr_normK, [hybs.lqr_normK], '%.4f', hdr);
prow('Gain-design time (ms)', mono.lqr_gain_ms, [hybs.lqr_gain_ms], '%.3f', hdr);

fprintf('\n--- [F] Shooting MPC (400 cand/step, H=5, 60 steps; true plant) ---\n');
fprintf('%s\n', hdr);
prow('Final ||x|| (30-trial ref.)', mono.shoot_final, [hybs.shoot_final], '%.4f', hdr);
prow('Final ||x|| (6-trial meas.)', mono.shoot_final6, [hybs.shoot_final6], '%.4f', hdr);
prow('Prediction total (s)', mono.shoot_pred_total_s, [hybs.shoot_pred_total_s], '%.3f', hdr);
prow('Wall total (s)', mono.shoot_wall_total_s, [hybs.shoot_wall_total_s], '%.3f', hdr);
prow('Prediction per step (ms)', mono.shoot_pred_per_step_ms, [hybs.shoot_pred_per_step_ms], '%.3f', hdr);
prow('Per rollout forward (us)', mono.shoot_pred_per_fwd_us, [hybs.shoot_pred_per_fwd_us], '%.3f', hdr);

%% =========================================================================
%  [8] Summary ratios + figure + save
%  =========================================================================
fprintf('\n%s\n', repmat('=', 1, 108));
fprintf('SUMMARY  (value for hybrid N / value for single; <1 = hybrid cheaper/better)\n');
fprintf('%s\n', repmat('=', 1, 108));
names = {'FLOPs/fwd', 'Latency/fwd', 'Train wall time', 'Learnable params', ...
         'Test RMSE', 'Shoot pred/step', 'Shoot final ||x||', 'LQR final ||x||'};
fprintf('%-22s', 'metric \\ N');
for n = 1:n_h, fprintf('%-16s', sprintf('N=%d', hybs(n).N)); end
fprintf('\n%s\n', repmat('-', 1, 22 + 16*n_h));
for i = 1:numel(names)
    switch i
        case 1, v = [hybs.flops_fwd]/mono.flops_fwd;
        case 2, v = [hybs.fwd_us_per_input]/mono.fwd_us_per_input;
        case 3, v = [hybs.train_time_s]/mono.train_time_s;
        case 4, v = [hybs.n_learnable]/mono.n_learnable;
        case 5, v = [hybs.test_rmse]/mono.test_rmse;
        case 6, v = [hybs.shoot_pred_per_step_ms]/mono.shoot_pred_per_step_ms;
        case 7, v = [hybs.shoot_final]/mono.shoot_final;
        case 8, v = [hybs.lqr_final]/mono.lqr_final;
    end
    fprintf('%-22s', names{i});
    for n = 1:n_h, fprintf('%-16.3f', v(n)); end
    fprintf('\n');
end

% --- Figure: three bars per panel (mono + each hybrid) ---
figure('Position', [100, 100, 1380, 560]);
axes_titles = {'Learnable params', 'Test RMSE (log)', 'Forward latency (us)', ...
    'Train time (s)', 'Shoot pred/step (ms)', 'Shoot final ||x||'};
vals = cell(6, 1);
vals{1} = [mono.n_learnable, [hybs.n_learnable]];
vals{2} = [mono.test_rmse, [hybs.test_rmse]];
vals{3} = [mono.fwd_us_per_input, [hybs.fwd_us_per_input]];
vals{4} = [mono.train_time_s, [hybs.train_time_s]];
vals{5} = [mono.shoot_pred_per_step_ms, [hybs.shoot_pred_per_step_ms]];
vals{6} = [mono.shoot_final, [hybs.shoot_final]];
xlabs = ['Single r=2', arrayfun(@(h) sprintf('Hyb r=1\nN=%d', h.N), hybs, 'UniformOutput', false)];
cols  = [[0 0.45 0.74]; repmat([0.49 0.18 0.56; 0.30 0.75 0.30; 0.85 0.33 0.10], 3, 1)];
for p = 1:6
    subplot(2, 3, p);
    v = vals{p}; nb = numel(v);
    b = bar(v, 0.6); b.FaceColor = 'flat';
    b.CData = cols(1:nb, :);
    set(gca, 'XTickLabel', xlabs);
    if p == 2, set(gca, 'YScale', 'log'); end
    title(axes_titles{p}); grid on;
    for k = 1:nb
        text(k, v(k)*1.05, sprintf('%.4g', v(k)), 'HorizontalAlignment', 'center', 'FontSize', 7);
    end
end
sgtitle(sprintf('Degree-2 Single (N=1) vs Hybrid degree-1 sub-PhNs (N=%s)', mat2str(Ns)), ...
    'FontWeight', 'bold', 'FontSize', 13);
if ~exist('fig', 'dir'), mkdir('fig'); end
saveas(gcf, sprintf('fig/OscHyb_SingleVsHybrid%s.png', out_tag)); close;

% --- Save ---
results = struct();
results.mono = mono;
results.hybrid = rmfield(hybs, 'h');
results.ratio = struct( ...
    'flops', [hybs.flops_fwd]/mono.flops_fwd, ...
    'latency', [hybs.fwd_us_per_input]/mono.fwd_us_per_input, ...
    'train_time', [hybs.train_time_s]/mono.train_time_s, ...
    'params', [hybs.n_learnable]/mono.n_learnable, ...
    'test_rmse', [hybs.test_rmse]/mono.test_rmse, ...
    'shoot_perstep', [hybs.shoot_pred_per_step_ms]/mono.shoot_pred_per_step_ms, ...
    'shoot_final', [hybs.shoot_final]/mono.shoot_final, ...
    'lqr_final', [hybs.lqr_final]/mono.lqr_final);
results.meta = struct('Ns', Ns, 'dim_state', dim_state, 'dim_control', dim_control, ...
    'n_trials_meas', N_TRIALS_MEAS, 'n_cand', N_CAND, 'H', H_SHOOT, ...
    'n_steps_shoot', N_STEPS_SHOOT, 'ub', UB, 'n_fwd_meas', N_MEAS, 'n_rep', N_REP);
if ~exist('results', 'dir'), mkdir('results'); end
save(fullfile('results', sprintf('compare_single_vs_hybrid_%s.mat', out_tag)), 'results');
fprintf('\n  Figure: fig/OscHyb_SingleVsHybrid%s.png\n', out_tag);
fprintf('  Data  : results/compare_single_vs_hybrid_%s.mat\n', out_tag);
fprintf('%s\n', repmat('=', 1, 108));
fprintf('COMPARISON COMPLETE (single r=2 vs hybrid r=1, N=%s)\n', mat2str(Ns));
fprintf('%s\n', repmat('=', 1, 108));

end

%% =========================================================================
%  Print one comparison row: label + mono value + each hybrid value
%  =========================================================================
function prow(label, v_mono, v_hybs, fmt, ~)
    line = sprintf(['%-30s %-18', strrep(fmt, '%', '')], label, v_mono);
    for n = 1:numel(v_hybs)
        line = [line sprintf([' %-18', strrep(fmt, '%', '')], v_hybs(n))]; %#ok<AGROW>
    end
    fprintf('%s\n', line);
end

%% =========================================================================
%  Timed shooting MPC over identical trials; returns final norms, prediction
%  wall time, total wall time
%  =========================================================================
function [fin, t_pred, t_wall] = run_shoot_timed(pred, osc, x0_list, n_trials, ...
    n_steps, n_cand, H, ub, lam, dim_state, dim_control)
    fin = zeros(n_trials, 1); t_pred = 0; t_wall = 0;
    for tr = 1:n_trials
        tw = tic;
        [traj, ~, tp] = osc_shoot_timed(pred, osc, x0_list(tr, :)', n_steps, ...
            n_cand, H, ub, lam, dim_state, dim_control, 1000 + tr);
        t_wall = t_wall + toc(tw); t_pred = t_pred + tp;
        fin(tr) = norm(traj(end, :));
    end
    fin = mean(fin);
end

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
