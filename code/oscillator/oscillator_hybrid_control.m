function results = oscillator_hybrid_control(opts)
%% OSCILLATOR_HYBRID_CONTROL  Hybrid framework on the coupled oscillator network
%   PCA + ME-bisecting + distributed PIM editing, applied to the spring-mass
%   network (ring topology, structurally homologous with Lorenz-96), followed
%   by a closed-loop regulation control verification over a sweep of the
%   partition count N.
%
%   Sub-neural-network experiment: the PIM-edited sub-PhNs (the Hybrid's
%   building blocks) are trained on a LOWER-ORDER Taylor library
%   (SUB_EXPANSION = 1) than the monolithic full model (EXPANSION = 2).
%   Because the oscillator dynamics are strictly linear (x' = A x + B u, a
%   second-order mechanical system in continuous time), the degree-1 library
%   already spans the exact map: the 1035 quadratic monomials of the
%   degree-2 library carry zero weight and are masked out by the PIM.  The
%   partition sweep (TARGETS) then verifies that the simpler sub-PhNs
%   reproduce the accuracy and the closed-loop control of the full model for
%   every partition count.
%
%   Control methodology follows the experiment report (Lorenz96_Experiment_
%   Report.md, Sec. 7): model-predictive shooting control (random candidates
%   per step, |u|<=2, 30 trials x 60 steps) with an LQR warm-start candidate
%   and horizon H=10 (the plain H=5 shooting controller provably cannot
%   regulate even with a perfect model -- see diag_shooting.m), plus
%   certainty-equivalence LQR (the paper's experiments.tex method) as a
%   secondary verification. The
%   only quantity that varies across controllers is the learned model used
%   for prediction / linear-part extraction; the true plant, the cost and
%   the input saturation are identical, so model accuracy is the sole driver
%   of control quality.
%
%   results = oscillator_hybrid_control()
%
%   Reuses: setup_oscillator, osc_step, design_lqr, build_oscillator_pim,
%   generate_monomial_indices, PhNNModel.
%   Saves:  results/oscillator_hybrid_results[_tag].mat, fig/OscHyb_*.png

%% =========================================================================
%  Parameters
%  =========================================================================
N_MASSES = 20; M_ACTUATORS = 5; EXPANSION = 2; SUB_EXPANSION = 1;
EPOCHS = 200; BATCH = 256; LR = 0.001; PATIENCE = 25;
LR_EDITED = 0.01;   % PIM-edited models are well-conditioned by construction
                    % (masks fix the known terms), so they tolerate a 10x
                    % larger Adam step; the unedited baselines require the
                    % conservative schedule (LR<=0.001) or they diverge.
VAR_THRESH = 0.90; MIN_SAMPLES = 100; MAX_PARTS = 16;
EPS_LIST = [0.85, 0.60, 0.45, 0.35, 0.28, 0.22, 0.17, 0.13, 0.10, 0.08, 0.06, 0.05, 0.04];
TARGETS = [2, 4, 8, 16];
N_SAMPLES = 12000;
OUT_TAG = '';

% Control
% Note: plain greedy random-shooting MPC (H=5, no warm-start) cannot regulate
% even with a perfect model (plateau ~2-4 over 300+ steps); the shooting MPC
% below uses an LQR warm-start candidate + H=10 so it genuinely regulates.
N_TRIALS = 30; N_STEPS_SHOOT = 60; N_CAND = 400; H_SHOOT = 10;
UB = 2.0; LAM = 0.01; N_STEPS_LQR = 500;

% Optional smoke-test overrides
if nargin >= 1 && ~isempty(opts)
    flds = {'N_SAMPLES','EPOCHS','BATCH','LR','LR_EDITED','N_TRIALS','N_CAND', ...
        'N_STEPS_SHOOT','H_SHOOT','N_STEPS_LQR','MIN_SAMPLES','VAR_THRESH', ...
        'EPS_LIST','TARGETS','PATIENCE','SUB_EXPANSION','EXPANSION','MAX_PARTS','OUT_TAG'};
    for f = 1:numel(flds)
        if isfield(opts, flds{f})
            eval(sprintf('%s = opts.%s;', flds{f}, flds{f}));
        end
    end
end

dim_state = 2 * N_MASSES; dim_control = M_ACTUATORS;
dim_input = dim_state + dim_control; dim_output = dim_state;
n_mono_full = nchoosek(dim_input + EXPANSION, EXPANSION) - 1;

fprintf('%s\n', repmat('=', 1, 78));
fprintf('Hybrid Framework on Coupled Oscillator Network + Closed-Loop Control\n');
fprintf('  %d masses -> %dD state, %d actuators -> %dD input (ring, homologous to L96)\n', ...
    N_MASSES, dim_state, M_ACTUATORS, dim_input);
fprintf('  Monolithic models: Taylor degree %d (%d monomials); sub-PhNs: degree %d (%d monomials)\n', ...
    EXPANSION, n_mono_full, SUB_EXPANSION, dim_input);
fprintf('  Control: shooting MPC (%d cand/step, H=%d, LQR warm-start) + certainty-equiv LQR; %d trials\n', ...
    N_CAND, H_SHOOT, N_TRIALS);
fprintf('%s\n', repmat('=', 1, 78));

%% =========================================================================
%  [1/7] Setup oscillator network + ground-truth LQR
%  =========================================================================
fprintf('\n[1/7] Setting up oscillator network and LQR...\n');
osc = setup_oscillator(N_MASSES, M_ACTUATORS);
Q_lqr = eye(dim_state) * 0.1; R_lqr = eye(dim_control) * 0.01;
K_lqr = design_lqr(osc.A_mat, osc.B_mat, Q_lqr, R_lqr);
fprintf('  ||K_lqr|| = %.3f (Q=0.1I, R=0.01I)\n', norm(K_lqr));

%% =========================================================================
%  [2/7] Generate data  (x,u) -> x'   with LQR-driven exploration
%  =========================================================================
fprintf('\n[2/7] Generating training data (random states + LQR control)...\n');
rng(42);
X_data = zeros(N_SAMPLES, dim_input, 'single');
Y_data = zeros(N_SAMPLES, dim_output, 'single');
for i = 1:N_SAMPLES
    x = randn(dim_state, 1) * 2.0;
    u = -K_lqr * x + randn(dim_control, 1) * 0.05;
    u = max(min(u, UB), -UB);
    x_next = osc_step(osc, x, u);
    X_data(i, :) = [x; u]';
    Y_data(i, :) = x_next';
end
idx = randperm(N_SAMPLES);
X_data = X_data(idx, :); Y_data = Y_data(idx, :);
n_tr = floor(N_SAMPLES * 0.7); n_va = floor(N_SAMPLES * 0.15);
Xtr = X_data(1:n_tr, :);       Ytr = Y_data(1:n_tr, :);
Xva = X_data(n_tr+1:n_tr+n_va, :); Yva = Y_data(n_tr+1:n_tr+n_va, :);
Xte = X_data(n_tr+n_va+1:end, :);  Yte = Y_data(n_tr+n_va+1:end, :);
fprintf('  Train: %dx%d, Val: %dx%d, Test: %dx%d\n', ...
    size(Xtr,1), size(Xtr,2), size(Xva,1), size(Xva,2), size(Xte,1), size(Xte,2));

%% =========================================================================
%  [3/7] Taylor libraries + oscillator PIM (ring topology)
%  =========================================================================
fprintf('\n[3/7] Building Taylor libraries and PIM masks...\n');
% Full (high-order) library for the monolithic baselines
mono = generate_monomial_indices(dim_input, EXPANSION);
n_mono = length(mono);
[A_val_pim, A_unc_pim] = build_oscillator_pim(N_MASSES, M_ACTUATORS, ...
    dim_state, dim_output, n_mono, mono, osc);
pim_sparsity = 1 - mean(A_unc_pim(:));
% Low-order (simpler) library for the Hybrid sub-PhNs
mono_sub = generate_monomial_indices(dim_input, SUB_EXPANSION);
n_mono_sub = length(mono_sub);
[A_val_sub, A_unc_sub] = build_oscillator_pim(N_MASSES, M_ACTUATORS, ...
    dim_state, dim_output, n_mono_sub, mono_sub, osc);
sub_sparsity = 1 - mean(A_unc_sub(:));
fprintf('  Input %dD, monolithic monomials (r=%d): %d, PIM sparsity: %.1f%%\n', ...
    dim_input, EXPANSION, n_mono, pim_sparsity*100);
fprintf('  Sub-PhN library (r=%d): %d monomials, PIM sparsity: %.1f%%\n', ...
    SUB_EXPANSION, n_mono_sub, sub_sparsity*100);

%% =========================================================================
%  [4/7] PCA feature projection + ME-bisecting partition (Algorithm 1)
%  =========================================================================
fprintf('\n[4/7] PCA feature projection and ME-bisecting...\n');
[pca_info, explained_cum] = build_pca(double(Xtr), VAR_THRESH);
n_p = pca_info.n_p;
fprintf('  %d PCs retain %.1f%% variance (n_p=%d, ambient=%d)\n', ...
    n_p, explained_cum(n_p), n_p, dim_input);

F_tr = (double(Xtr) - pca_info.mu) ./ pca_info.rangev * pca_info.Q';
F_va = (double(Xva) - pca_info.mu) ./ pca_info.rangev * pca_info.Q';

n_eps = length(EPS_LIST);
N_by_eps = zeros(1, n_eps);
boxes_by_eps = cell(1, n_eps);
for e = 1:n_eps
    boxes_by_eps{e} = me_bisect(F_tr, EPS_LIST(e), MIN_SAMPLES);
    N_by_eps(e) = numel(boxes_by_eps{e});
    fprintf('  eps=%.2f -> N=%d partitions\n', EPS_LIST(e), N_by_eps(e));
end

% Select the eps-configs whose partition counts are closest to targets
sel = [];
for t = 1:length(TARGETS)
    [~, k] = min(abs(N_by_eps - TARGETS(t)));
    if N_by_eps(k) >= 2 && N_by_eps(k) <= MAX_PARTS && ~any(sel == k)
        sel(end+1) = k; %#ok<AGROW>
    end
end
if isempty(sel)
    [~, k] = min(abs(N_by_eps - 4));
    if N_by_eps(k) >= 2 && N_by_eps(k) <= MAX_PARTS, sel = k; end
end
fprintf('  Training configs: N = %s\n', mat2str(sort(N_by_eps(sel))));

%% =========================================================================
%  [5/7] Train models  (unified protocol: max epochs, patience-25 early stop)
%  =========================================================================
fprintf('\n[5/7] Training models...\n');

% --- 5.1 Monolithic unedited PhNN (weakest baseline, degree EXPANSION) ---
fprintf('\n  [monolithic] Unedited PhNN (r=%d)...\n', EXPANSION);
m_ued = PhNNModel(dim_input, dim_output, mono);
t0 = tic;
[~, va_ued, v_ued] = m_ued.train(Xtr, Ytr, Xva, Yva, LR, EPOCHS, BATCH, PATIENCE);
t_ued = toc(t0);
res_ued = struct('N', 1, 'val_loss', v_ued, 'n_learnable', m_ued.n_learnable, ...
    'sparsity', m_ued.sparsity, 'train_time', t_ued, 'epochs_used', numel(va_ued));
res_ued.model = m_ued;
fprintf('  Unedited: val=%.4e, params=%d, epochs=%d, time=%.1fs\n', ...
    v_ued, m_ued.n_learnable, numel(va_ued), t_ued);

% --- 5.2 Monolithic PIM-edited PhNN (== Hybrid N=1, degree EXPANSION) ---
fprintf('\n  [monolithic] PIM-Edited PhNN (== Hybrid N=1, r=%d)...\n', EXPANSION);
m_pim = PhNNModel(dim_input, dim_output, mono, A_val_pim, A_unc_pim);
t0 = tic;
[~, va_pim, v_pim] = m_pim.train(Xtr, Ytr, Xva, Yva, LR_EDITED, EPOCHS, BATCH, PATIENCE);
t_pim = toc(t0);
res_pim = struct('N', 1, 'val_loss', v_pim, 'n_learnable', m_pim.n_learnable, ...
    'sparsity', m_pim.sparsity, 'train_time', t_pim, 'epochs_used', numel(va_pim), 'n_p', n_p);
res_pim.model = m_pim;
fprintf('  PIM-Edited: val=%.4e, params=%d, epochs=%d, time=%.1fs\n', ...
    v_pim, m_pim.n_learnable, numel(va_pim), t_pim);

% --- 5.3 Hybrid configs (distributed PIM-edited sub-PhNs, degree SUB_EXPANSION) ---
fprintf('\n  [hybrid] distributed PIM-edited sub-PhNs (r=%d)...\n', SUB_EXPANSION);
hybrid = [];
for s = 1:length(sel)
    k = sel(s); Nk = N_by_eps(k); boxes = boxes_by_eps{k};
    fprintf('\n  [hybrid] N=%d (eps=%.2f)...\n', Nk, EPS_LIST(k));
    b_tr = assign_mode(F_tr, boxes);
    b_va = assign_mode(F_va, boxes);
    models = cell(1, Nk);
    means_x = zeros(Nk, dim_input); means_y = zeros(Nk, dim_output);
    val_pred = zeros(size(Yva), 'single'); ok = false(size(Yva, 1), 1);
    n_epochs_h = zeros(1, Nk);
    n_learn_h = 0; n_tot_h = 0;
    t0 = tic;
    for i = 1:Nk
        sxi = (b_tr == i);
        if ~any(sxi)
            warning('Partition %d has no training samples', i);
            models{i} = []; continue;
        end
        mu_x = mean(Xtr(sxi, :), 1); mu_y = mean(Ytr(sxi, :), 1);
        means_x(i, :) = mu_x; means_y(i, :) = mu_y;
        m_i = PhNNModel(dim_input, dim_output, mono_sub, A_val_sub, A_unc_sub);
        svi = find(b_va == i);
        if ~isempty(svi)
            [~, va_i, ~] = m_i.train(Xtr(sxi, :) - mu_x, Ytr(sxi, :) - mu_y, ...
                Xva(svi, :) - mu_x, Yva(svi, :) - mu_y, LR_EDITED, EPOCHS, BATCH, PATIENCE);
        else
            nv = min(50, size(Xva, 1));
            [~, va_i, ~] = m_i.train(Xtr(sxi, :) - mu_x, Ytr(sxi, :) - mu_y, ...
                Xva(1:nv, :) - mu_x, Yva(1:nv, :) - mu_y, LR_EDITED, EPOCHS, BATCH, PATIENCE);
        end
        n_epochs_h(i) = numel(va_i);
        n_learn_h = n_learn_h + m_i.n_learnable; n_tot_h = n_tot_h + m_i.n_total;
        models{i} = m_i;
        fprintf('    sub-PhN %2d/%2d trained (%d samples, %d epochs, %d learnable)\n', ...
            i, Nk, sum(sxi), numel(va_i), m_i.n_learnable);
    end
    t_h = toc(t0);
    for i = 1:Nk
        sxi = (b_va == i);
        if ~isempty(models{i}) && any(sxi)
            val_pred(sxi, :) = models{i}.forward(Xva(sxi, :) - means_x(i, :)) + means_y(i, :);
            ok(sxi) = true;
        end
    end
    if any(ok)
        v_h = mean((val_pred(ok, :) - Yva(ok, :)).^2, 'all');
    else
        v_h = NaN;
    end
    hybrid = [hybrid; struct('N', Nk, 'eps', EPS_LIST(k), 'val_loss', v_h, ...
        'n_learnable', n_learn_h, 'n_total', n_tot_h, ...
        'sparsity', 1 - n_learn_h/max(n_tot_h, 1), ...
        'train_time', t_h, 'epochs_used', max(n_epochs_h), 'n_p', n_p, ...
        'sub_expansion', SUB_EXPANSION, ...
        'means_x', means_x, 'means_y', means_y, 'models', {models}, 'boxes', {boxes})]; %#ok<AGROW>
    fprintf('  Hybrid N=%d: val=%.4e, params=%d, epochs=%d, time=%.1fs\n', ...
        Nk, v_h, n_learn_h, max(n_epochs_h), t_h);
end

% --- 5.4 Ordinary Hybrid baseline: partitioning WITHOUT PIM editing,
%      same low-order sub-PhNs (degree SUB_EXPANSION) ---
ord_hybrid = [];
if ~isempty(sel)
    k0 = sel(1); Nk0 = N_by_eps(k0); boxes0 = boxes_by_eps{k0};
    b_tr0 = assign_mode(F_tr, boxes0); b_va0 = assign_mode(F_va, boxes0);
    models0 = cell(1, Nk0);
    means_x0 = zeros(Nk0, dim_input); means_y0 = zeros(Nk0, dim_output);
    val_pred0 = zeros(size(Yva), 'single'); ok0 = false(size(Yva, 1), 1);
    fprintf('\n  [hybrid-unedited] Ordinary Hybrid PhN (N=%d, r=%d, no PIM editing)...\n', ...
        Nk0, SUB_EXPANSION);
    n_epochs_o = zeros(1, Nk0);
    n_learn_o = 0; n_tot_o = 0;
    t0 = tic;
    for i = 1:Nk0
        sxi = (b_tr0 == i);
        if ~any(sxi), models0{i} = []; continue; end
        mu_x = mean(Xtr(sxi, :), 1); mu_y = mean(Ytr(sxi, :), 1);
        means_x0(i, :) = mu_x; means_y0(i, :) = mu_y;
        m0 = PhNNModel(dim_input, dim_output, mono_sub);  % UNEDITED sub-PhN
        svi = find(b_va0 == i);
        if ~isempty(svi)
            [~, va0, ~] = m0.train(Xtr(sxi, :) - mu_x, Ytr(sxi, :) - mu_y, ...
                Xva(svi, :) - mu_x, Yva(svi, :) - mu_y, LR, EPOCHS, BATCH, PATIENCE);
        else
            nv = min(50, size(Xva, 1));
            [~, va0, ~] = m0.train(Xtr(sxi, :) - mu_x, Ytr(sxi, :) - mu_y, ...
                Xva(1:nv, :) - mu_x, Yva(1:nv, :) - mu_y, LR, EPOCHS, BATCH, PATIENCE);
        end
        n_epochs_o(i) = numel(va0);
        n_learn_o = n_learn_o + m0.n_learnable; n_tot_o = n_tot_o + m0.n_total;
        models0{i} = m0;
        fprintf('    unedited sub-PhN %2d/%2d trained (%d samples, %d epochs)\n', ...
            i, Nk0, sum(sxi), numel(va0));
    end
    t_ord = toc(t0);
    for i = 1:Nk0
        sxi = (b_va0 == i);
        if ~isempty(models0{i}) && any(sxi)
            val_pred0(sxi, :) = models0{i}.forward(Xva(sxi, :) - means_x0(i, :)) + means_y0(i, :);
            ok0(sxi) = true;
        end
    end
    if any(ok0)
        v_ord = mean((val_pred0(ok0, :) - Yva(ok0, :)).^2, 'all');
    else
        v_ord = NaN;
    end
    ord_hybrid = struct('N', Nk0, 'val_loss', v_ord, ...
        'n_learnable', n_learn_o, 'train_time', t_ord, ...
        'epochs_used', max(n_epochs_o), ...
        'models0', {models0}, 'boxes0', {boxes0}, ...
        'means_x', means_x0, 'means_y', means_y0);
    fprintf('  Ordinary Hybrid (unedited) N=%d: val=%.4e, params=%d, epochs=%d, time=%.1fs\n', ...
        Nk0, v_ord, n_learn_o, max(n_epochs_o), t_ord);
end

% --- 5.5 Model-accuracy metrics: single-step test RMSE + rollout RMSE ---
fprintf('\n  Evaluating model accuracy on the test set...\n');
pred_ued = @(X, U) m_ued.forward(single([X U]));
pred_pim = @(X, U) m_pim.forward(single([X U]));

% Generate a contiguous closed-loop test trajectory (true plant, LQR policy)
rng(2026);
n_roll_start = 6; roll_len = 30;
Xstart = randn(n_roll_start, dim_state) * 3.0;
roll_preds = {pred_ued, pred_pim};          % unedited, pim
roll_labels = {'ued', 'pim'};
for s = 1:length(hybrid)
    roll_preds{end+1} = @(X, U) osc_hyb_predict(hybrid(s).models, hybrid(s).boxes, ...
        pca_info, hybrid(s).means_x, hybrid(s).means_y, X, U); %#ok<AGROW>
    roll_labels{end+1} = sprintf('hyb%d', hybrid(s).N); %#ok<AGROW>
end
if ~isempty(ord_hybrid)
    roll_preds{end+1} = @(X, U) osc_hyb_predict(ord_hybrid.models0, ord_hybrid.boxes0, ...
        pca_info, ord_hybrid.means_x, ord_hybrid.means_y, X, U); %#ok<AGROW>
    roll_labels{end+1} = 'ordhyb'; %#ok<AGROW>
end
n_models = numel(roll_preds);
roll = zeros(n_models, roll_len);
for st = 1:n_roll_start
    x = Xstart(st, :)';
    traj = zeros(roll_len + 1, dim_state); traj(1, :) = x';
    Useq = zeros(roll_len, dim_control);
    for t = 1:roll_len
        u = -K_lqr * x;
        u = max(min(u, UB), -UB);
        Useq(t, :) = u';
        x = osc_step(osc, x, u);
        traj(t+1, :) = x';
    end
    for mi = 1:n_models
        xp = Xstart(st, :);
        for t = 1:roll_len
            xp = roll_preds{mi}(single(xp), single(Useq(t, :)));
            roll(mi, t) = roll(mi, t) + sqrt(mean((xp(:) - traj(t+1, :)').^2));
            xp = double(xp);
        end
    end
end
roll = roll / n_roll_start;

% Single-step test RMSE
rmse_test = zeros(1, n_models);
rmse_test(1) = sqrt(mean((m_ued.forward(Xte) - Yte).^2, 'all'));
rmse_test(2) = sqrt(mean((m_pim.forward(Xte) - Yte).^2, 'all'));
for s = 1:length(hybrid)
    rmse_test(2 + s) = osc_hyb_test_rmse(hybrid(s).models, hybrid(s).boxes, pca_info, ...
        hybrid(s).means_x, hybrid(s).means_y, Xte, Yte);
end
if ~isempty(ord_hybrid)
    % rmse_test is preallocated to exactly n_models; ordhyb is the LAST slot
    rmse_test(end) = osc_hyb_test_rmse(ord_hybrid.models0, ord_hybrid.boxes0, pca_info, ...
        ord_hybrid.means_x, ord_hybrid.means_y, Xte, Yte);
end

%% =========================================================================
%  [6/7] Closed-loop control verification
%  =========================================================================
fprintf('\n%s\n', repmat('=', 1, 78));
fprintf('[6/7] Closed-loop regulation control (true plant, %d trials)\n', N_TRIALS);
fprintf('%s\n', repmat('=', 1, 78));

% Trial initial states (shared across controllers for fairness)
rng(777);
x0_list = randn(N_TRIALS, dim_state) * 3.0;

% Dynamic controller set: LQR, monolithic PIM, every hybrid, ordinary hybrid
ctrl = struct();
ctrl.configs = {'lqr', 'pim'};
ctrl.config_labels = {'LQR (optimum)', 'PIM-Edited (N=1)'};
for s = 1:length(hybrid)
    ctrl.configs{end+1} = sprintf('hyb%d', hybrid(s).N); %#ok<AGROW>
    ctrl.config_labels{end+1} = sprintf('Hybrid N=%d', hybrid(s).N); %#ok<AGROW>
end
if ~isempty(ord_hybrid)
    ctrl.configs{end+1} = 'ordhyb'; %#ok<AGROW>
    ctrl.config_labels{end+1} = sprintf('Ordinary Hybrid (unedited N=%d)', ord_hybrid.N); %#ok<AGROW>
end
n_ctrl = numel(ctrl.configs);

% Representative hybrids for the accuracy->control sweep
hybN = [hybrid.N];
i_hmin = find(hybN == min(hybN), 1);
i_hmax = find(hybN == max(hybN), 1);
if i_hmin == i_hmax
    fprintf('  [warning] all hybrids share one partition count (N=%d)\n', hybN(i_hmin));
end

% --- 6.1 Certainty-equivalence LQR (paper experiments.tex method) ---
fprintf('\n  [6.1] Certainty-equivalence LQR (extract linear part, same Riccati)...\n');
W_eff_ued = m_ued.A_value + m_ued.A_uncertain .* m_ued.W_learn;
W_eff_pim = m_pim.A_value + m_pim.A_uncertain .* m_pim.W_learn;
W_effs = struct('pim', W_eff_pim);
for s = 1:length(hybrid)
    k = ctrl.configs{2 + s};
    W_effs.(k) = osc_hyb_linpart(hybrid(s).models, hybrid(s).boxes, pca_info, ...
        hybrid(s).means_x, dim_input);
end
if ~isempty(ord_hybrid)
    W_effs.ordhyb = osc_hyb_linpart(ord_hybrid.models0, ord_hybrid.boxes0, pca_info, ...
        ord_hybrid.means_x, dim_input);
else
    W_effs.ordhyb = W_eff_ued;
end

Ks = struct();
Ks.lqr = K_lqr;
Ks.pim = osc_get_lqr_gain(W_eff_pim, dim_state, dim_control, Q_lqr, R_lqr);
for s = 1:length(hybrid)
    k = ctrl.configs{2 + s};
    Ks.(k) = osc_get_lqr_gain(W_effs.(k), dim_state, dim_control, Q_lqr, R_lqr);
end
Ks.ordhyb = osc_get_lqr_gain(W_effs.ordhyb, dim_state, dim_control, Q_lqr, R_lqr);

lqr_final = struct();
for k = ctrl.configs
    final_all = zeros(N_TRIALS, 1); mean_traj = zeros(N_TRIALS, N_STEPS_LQR + 1);
    for tr = 1:N_TRIALS
        [~, norms] = osc_lqr_run(osc, Ks.(k{1}), x0_list(tr, :)', N_STEPS_LQR, UB, dim_state, dim_control);
        final_all(tr) = norms(end); mean_traj(tr, :) = norms;
    end
    lqr_final.(k{1}) = struct('final', mean(final_all), 'std', std(final_all), ...
        'traj_mean', mean(mean_traj, 1), 'traj_std', std(mean_traj, 0, 1));
end

% --- 6.2 Shooting MPC (report method: random candidates, H-step rollout) ---
fprintf('\n  [6.2] Model-predictive shooting control (%d candidates/step, H=%d, %d trials x %d steps)...\n', ...
    N_CAND, H_SHOOT, N_TRIALS, N_STEPS_SHOOT);
shoot = struct();
shoot_preds = struct();
shoot_preds.pim = pred_pim;
for s = 1:length(hybrid)
    k = ctrl.configs{2 + s};
    shoot_preds.(k) = @(X, U) osc_hyb_predict(hybrid(s).models, hybrid(s).boxes, ...
        pca_info, hybrid(s).means_x, hybrid(s).means_y, X, U);
end
if ~isempty(ord_hybrid)
    shoot_preds.ordhyb = @(X, U) osc_hyb_predict(ord_hybrid.models0, ord_hybrid.boxes0, ...
        pca_info, ord_hybrid.means_x, ord_hybrid.means_y, X, U);
end

for k = ctrl.configs
    final_all = zeros(N_TRIALS, 1); all_traj = zeros(N_TRIALS, N_STEPS_SHOOT + 1);
    for tr = 1:N_TRIALS
        if strcmp(k{1}, 'lqr')
            [~, norms] = osc_lqr_run(osc, K_lqr, x0_list(tr, :)', N_STEPS_SHOOT, UB, dim_state, dim_control);
            final_all(tr) = norms(end); all_traj(tr, :) = norms;
        else
            pred = shoot_preds.(k{1});
            [traj, ~] = osc_shoot(pred, osc, x0_list(tr, :)', N_STEPS_SHOOT, N_CAND, H_SHOOT, ...
                UB, LAM, K_lqr, dim_state, dim_control, 1000 + tr);
            final_all(tr) = norm(traj(end, :)); all_traj(tr, :) = sqrt(sum(traj.^2, 2));
        end
    end
    shoot.(k{1}) = struct('final', mean(final_all), 'std', std(final_all), ...
        'traj_mean', mean(all_traj, 1), 'traj_std', std(all_traj, 0, 1));
end

%% =========================================================================
%  [7/7] Results tables + figures + save
%  =========================================================================
fprintf('\n%s\n', repmat('=', 1, 90));
fprintf('[7/7] RESULTS: Oscillator Network Hybrid + Closed-Loop Control\n');
fprintf('%s\n', repmat('=', 1, 90));

% --- Accuracy table ---
fprintf('\nModel accuracy (test set, shared data):\n');
fprintf('  Monolithic baselines: Taylor r=%d (%d mono); sub-PhNs (incl. ordinary): r=%d (%d mono)\n', ...
    EXPANSION, n_mono, SUB_EXPANSION, n_mono_sub);
fprintf('%-30s %-13s %-12s %-12s %-10s\n', 'Method', 'Test RMSE', 'RMSE@5', 'RMSE@10', 'Params');
fprintf('%s\n', repmat('-', 1, 78));
acc_names = {'Unedited PhNN', 'PIM-Edited (N=1)'};
for s = 1:length(hybrid), acc_names{end+1} = sprintf('Hybrid N=%d', hybrid(s).N); end %#ok<AGROW>
if ~isempty(ord_hybrid), acc_names{end+1} = sprintf('Ordinary Hybrid (unedited N=%d)', ord_hybrid.N); end %#ok<AGROW>
for mi = 1:n_models
    if mi == 1, params = m_ued.n_learnable; elseif mi == 2, params = m_pim.n_learnable;
    elseif mi <= 2 + length(hybrid), params = hybrid(mi - 2).n_learnable;
    else, params = ord_hybrid.n_learnable; end
    fprintf('%-30s %-13.4e %-12.4f %-12.4f %-10d\n', acc_names{mi}, rmse_test(mi), ...
        roll(mi, 5), roll(mi, 10), params);
end

% --- Control tables ---
fprintf('\n[Shooting MPC] closed-loop regulation (%d trials x %d steps, report method):\n', N_TRIALS, N_STEPS_SHOOT);
fprintf('%-30s %-14s %-12s %-12s\n', 'Controller', 'Final ||x||', 'vs OrdHyb', 'vs LQR');
fprintf('%s\n', repmat('-', 1, 70));
for i = 1:n_ctrl
    k = ctrl.configs{i}; r = shoot.(k);
    if strcmp(k, 'ordhyb'), pct_ord = 'baseline'; else, pct_ord = sprintf('%+.1f%%', (r.final - shoot.ordhyb.final)/shoot.ordhyb.final*100); end
    pct_lqr = sprintf('%+.1f%%', (r.final - shoot.lqr.final)/shoot.lqr.final*100);
    fprintf('%-30s %-14.4f %-12s %-12s\n', ctrl.config_labels{i}, r.final, pct_ord, pct_lqr);
end

fprintf('\n[Certainty-equivalence LQR] (%d trials x %d steps, paper method):\n', N_TRIALS, N_STEPS_LQR);
fprintf('%-30s %-14s %-12s %-12s\n', 'Controller', 'Final ||x||', 'vs OrdHyb', 'vs LQR');
fprintf('%s\n', repmat('-', 1, 70));
for i = 1:n_ctrl
    k = ctrl.configs{i}; r = lqr_final.(k);
    if strcmp(k, 'ordhyb'), pct_ord = 'baseline'; else, pct_ord = sprintf('%+.1f%%', (r.final - lqr_final.ordhyb.final)/lqr_final.ordhyb.final*100); end
    pct_lqr = sprintf('%+.1f%%', (r.final - lqr_final.lqr.final)/lqr_final.lqr.final*100);
    fprintf('%-30s %-14.4f %-12s %-12s\n', ctrl.config_labels{i}, r.final, pct_ord, pct_lqr);
end

% --- Figures ---
fprintf('\nGenerating figures...\n');
if ~exist('fig', 'dir'), mkdir('fig'); end
if isempty(OUT_TAG)
    fig_tag = ''; res_file = 'oscillator_hybrid_results.mat';
else
    fig_tag = ['_' OUT_TAG]; res_file = ['oscillator_hybrid_results_' OUT_TAG '.mat'];
end

% Fig 1: model accuracy (single-step test RMSE bar + rollout curves)
figure('Position', [100, 100, 1320, 480]);
subplot(1, 2, 1);
bar(1:n_models, rmse_test, 0.6); set(gca, 'XTickLabel', acc_names, 'YScale', 'log');
ylabel('Single-step test RMSE (log)'); title('(a) Model Accuracy'); grid on;
for mi = 1:n_models
    text(mi, rmse_test(mi)*1.5, sprintf('%.2e', rmse_test(mi)), 'HorizontalAlignment', 'center', 'FontSize', 7);
end
xtickangle(20);
subplot(1, 2, 2); hold on;
for mi = 1:n_models
    plot(1:roll_len, roll(mi, :), 'LineWidth', 1.8, 'Color', hyb_palette(mi), 'DisplayName', acc_names{mi});
end
xlabel('Prediction step (closed-loop, known LQR inputs)'); ylabel('Rollout RMSE');
title('(b) Multi-step Prediction'); legend('FontSize', 6); grid on;
sgtitle(sprintf('Oscillator Hybrid (sub-PhNs: Taylor r=%d): Model Accuracy vs Partitions', SUB_EXPANSION), ...
    'FontWeight', 'bold', 'FontSize', 12);
saveas(gcf, ['fig/OscHyb_ModelAccuracy' fig_tag '.png']); close;

% Fig 2: shooting MPC norm evolution
figure('Position', [100, 100, 1320, 520]); hold on;
for i = 1:n_ctrl
    k = ctrl.configs{i}; r = shoot.(k);
    steps = 0:N_STEPS_SHOOT;
    plot(steps, r.traj_mean, 'Color', hyb_palette(i), 'LineWidth', 2.0, 'DisplayName', ctrl.config_labels{i});
    fill([steps, fliplr(steps)], [max(r.traj_mean - r.traj_std, 1e-12), fliplr(r.traj_mean + r.traj_std)], ...
        hyb_palette(i), 'FaceAlpha', 0.12, 'EdgeColor', 'none');
end
xlabel('Control step'); ylabel('||x||'); title('Shooting MPC: closed-loop regulation (30 trials)');
legend('Location', 'northeast', 'FontSize', 7); grid on;
sgtitle(sprintf('OscHyb Control: Shooting MPC (sub-PhNs: Taylor r=%d)', SUB_EXPANSION), ...
    'FontWeight', 'bold', 'FontSize', 12);
saveas(gcf, ['fig/OscHyb_ControlShooting' fig_tag '.png']); close;

% Fig 3: accuracy -> control link across the partition sweep
figure('Position', [100, 100, 1120, 480]);
Ns_link = [1, hybN];
rmse_link = zeros(1, 1 + length(hybrid)); shoot_link = zeros(1, 1 + length(hybrid));
rmse_link(1) = rmse_test(2); shoot_link(1) = shoot.pim.final;
for s = 1:length(hybrid)
    rmse_link(1 + s) = rmse_test(2 + s);
    shoot_link(1 + s) = shoot.(ctrl.configs{2 + s}).final;
end
subplot(1, 2, 1); hold on;
plot(Ns_link, rmse_link, 'o-', 'LineWidth', 2, 'Color', [0 0.45 0.74], 'MarkerSize', 9, 'MarkerFaceColor', [0 0.45 0.74]);
if ~isempty(ord_hybrid)
    plot(ord_hybrid.N, rmse_test(end), '^', 'MarkerSize', 11, 'MarkerFaceColor', [0.85 0.60 0.20], 'Color', [0.85 0.60 0.20]);
end
xlabel('Partitions N'); ylabel('Single-step test RMSE'); set(gca, 'YScale', 'log');
title('(a) Accuracy vs N'); grid on;
subplot(1, 2, 2); hold on;
plot(Ns_link, shoot_link, 's-', 'LineWidth', 2, 'Color', [0.49 0.18 0.56], 'MarkerSize', 9, 'MarkerFaceColor', [0.49 0.18 0.56]);
if ~isempty(ord_hybrid)
    plot(ord_hybrid.N, shoot.ordhyb.final, '^', 'MarkerSize', 11, 'MarkerFaceColor', [0.85 0.60 0.20], 'Color', [0.85 0.60 0.20]);
end
plot(0, shoot.lqr.final, 'ko', 'MarkerSize', 9, 'MarkerFaceColor', 'k', 'DisplayName', 'LQR optimum');
xlabel('Partitions N'); ylabel('Final ||x|| (shooting MPC)'); title('(b) Control vs N'); grid on;
legend('Location', 'best', 'FontSize', 7);
sgtitle(sprintf('Oscillator Hybrid (degree-%d sub-PhNs): Simpler Model -> Same Control', SUB_EXPANSION), ...
    'FontWeight', 'bold', 'FontSize', 12);
saveas(gcf, ['fig/OscHyb_ControlEffect' fig_tag '.png']); close;

% Fig 4: certainty-equivalence LQR final norm
figure('Position', [100, 100, 1080, 480]);
finals = zeros(1, n_ctrl);
for i = 1:n_ctrl, finals(i) = lqr_final.(ctrl.configs{i}).final; end
b = bar(finals); b.FaceColor = 'flat';
b.CData = zeros(n_ctrl, 3);
for i = 1:n_ctrl, b.CData(i, :) = hyb_palette(i); end
set(gca, 'XTickLabel', ctrl.config_labels, 'YScale', 'log');
ylabel('Final ||x|| (log)'); title('Certainty-equivalence LQR (paper method)');
for i = 1:n_ctrl
    text(i, finals(i)*1.6, sprintf('%.4f', finals(i)), 'HorizontalAlignment', 'center', 'FontSize', 7);
end
xtickangle(20); grid on;
sgtitle(sprintf('OscHyb Control: Certainty-Equivalence LQR (degree-%d sub-PhNs)', SUB_EXPANSION), ...
    'FontWeight', 'bold', 'FontSize', 12);
saveas(gcf, ['fig/OscHyb_ControlLQR' fig_tag '.png']); close;

fprintf('  Figures saved to fig/OscHyb_*%s.png\n', fig_tag);

% --- Assemble & save ---
results = struct();
results.pca_info   = pca_info;
results.res_ued    = res_ued;
results.res_pim    = res_pim;
results.hybrid     = hybrid;
results.ord_hybrid = ord_hybrid;
results.accuracy   = struct('names', {acc_names}, 'rmse_test', rmse_test, ...
    'roll', roll, 'roll_len', roll_len, 'rmse5', roll(:, 5), 'rmse10', roll(:, 10));
results.ctrl_lqr   = lqr_final;
results.ctrl_shoot = shoot;
results.meta       = struct('N_MASSES', N_MASSES, 'M_ACTUATORS', M_ACTUATORS, ...
    'dim_input', dim_input, 'EXPANSION', EXPANSION, 'SUB_EXPANSION', SUB_EXPANSION, ...
    'n_mono', n_mono, 'n_mono_sub', n_mono_sub, ...
    'n_trials', N_TRIALS, 'n_cand', N_CAND, 'H', H_SHOOT, ...
    'n_steps_shoot', N_STEPS_SHOOT, 'n_steps_lqr', N_STEPS_LQR, 'ub', UB);

out_dir = fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))), 'results');  % <repo>/results
if ~exist(out_dir, 'dir'), mkdir(out_dir); end
save(fullfile(out_dir, res_file), 'results');

fprintf('\n%s\n', repmat('=', 1, 78));
fprintf('Oscillator Hybrid + Control Experiment Complete!\n');
fprintf('  Results saved to results/%s\n', res_file);
fprintf('%s\n', repmat('=', 1, 78));

end

%% ========================================================================
%  PCA feature projection (same as hybrid_framework.m)
%  ========================================================================
function [pca_info, explained_cum] = build_pca(X, var_threshold)
    mu = mean(X, 1);
    mn = min(X, [], 1); mx = max(X, [], 1);
    rangev = mx - mn; rangev(rangev == 0) = 1;
    Xn = (X - mu) ./ rangev;
    [n, d] = size(Xn);
    C = (Xn' * Xn) / (n - 1);
    [V, D] = eig(C);
    evals = diag(D);
    [evals, ord] = sort(evals, 'descend');
    V = V(:, ord);
    explained = evals / sum(evals) * 100;
    explained_cum = cumsum(explained);
    n_p = find(explained_cum >= var_threshold * 100, 1);
    if isempty(n_p), n_p = d; end
    Q = V(:, 1:n_p)';
    pca_info = struct('Q', Q, 'mu', mu, 'rangev', rangev, 'n_p', n_p, ...
        'explained_cum', explained_cum);
end

%% ========================================================================
%  Maximum-Entropy bisecting  (Algorithm 1 of the paper)
%  ========================================================================
function leaves = me_bisect(F, eps, min_samples)
    n = size(F, 1);
    lo0 = min(F, [], 1); hi0 = max(F, [], 1);
    active = struct('lo', lo0, 'hi', hi0, 'idx', (1:n)');
    leaves = {};
    while numel(active) > 0
        maxw = -inf; bi = 1;
        for b = 1:numel(active)
            w = max(active(b).hi - active(b).lo);
            if w > maxw, maxw = w; bi = b; end
        end
        box = active(bi);
        [~, d] = max(box.hi - box.lo);
        mid = (box.lo(d) + box.hi(d)) / 2;
        vals = F(box.idx, d);
        m = vals <= mid;
        n1 = sum(m); n2 = numel(box.idx) - n1;
        p = numel(box.idx) / n;
        H_before = -p * log2(p);
        split_ok = false;
        if n1 >= min_samples && n2 >= min_samples
            p1 = n1 / n; p2 = n2 / n;
            H_after = -p1*log2(p1) - p2*log2(p2);
            if (H_after - H_before) >= eps
                split_ok = true;
            end
        end
        if split_ok
            idx1 = box.idx(m); idx2 = box.idx(~m);
            c1.lo = min(F(idx1, :), [], 1); c1.hi = max(F(idx1, :), [], 1); c1.idx = idx1;
            c2.lo = min(F(idx2, :), [], 1); c2.hi = max(F(idx2, :), [], 1); c2.idx = idx2;
            active(bi) = [];
            active(end+1) = c1; %#ok<AGROW>
            active(end+1) = c2; %#ok<AGROW>
        else
            leaves{end+1} = box; %#ok<AGROW>
            active(bi) = [];
        end
    end
end

%% ========================================================================
%  Mode function delta(x): nearest box in Chebyshev sense
%  ========================================================================
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

%% ========================================================================
%  Batched mode-switched prediction for a hybrid model  (n x 40 + n x 5)
%  ========================================================================
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

%% ========================================================================
%  Single-step test RMSE of a hybrid model
%  ========================================================================
function rmse = osc_hyb_test_rmse(models, boxes, pca_info, means_x, means_y, Xte, Yte)
    dim_control = size(means_x, 2) - size(means_y, 2);
    dim_state = size(Xte, 2) - dim_control;
    pred = osc_hyb_predict(models, boxes, pca_info, means_x, means_y, ...
        Xte(:, 1:dim_state), Xte(:, dim_state+1:end));
    rmse = sqrt(mean((pred(:) - Yte(:)).^2));
end

%% ========================================================================
%  Certainty-equivalence LQR: design same LQR from a learned linear part
%  ========================================================================
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

%% ========================================================================
%  Linear part of a hybrid model: active sub-PhN at the equilibrium origin
%  ========================================================================
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

%% ========================================================================
%  Closed loop with a fixed linear gain (LQR or learned), on the true plant
%  ========================================================================
function [traj, norms] = osc_lqr_run(osc, K, x0, n_steps, ub, dim_state, dim_control)
    x = double(x0(:));
    traj = zeros(n_steps + 1, dim_state); traj(1, :) = x';
    norms = zeros(1, n_steps + 1); norms(1) = norm(x);
    for t = 1:n_steps
        u = -K * x;
        u = max(min(u, ub), -ub);
        x = osc_step(osc, x, u);
        traj(t + 1, :) = x';
        norms(t + 1) = norm(x);
    end
end

%% ========================================================================
%  Model-predictive shooting control (report method): random candidate
%  controls, H-step zero-order-hold rollout through the learned model.
%  K_ws = warm-start gain: adds the saturated LQR action u = -K_ws*x as a
%  candidate every step (plain random shooting alone cannot regulate -- see
%  diag_shooting.m).  Pass [] to disable the warm-start.
%  ========================================================================
function [traj, U] = osc_shoot(pred, osc, x0, n_steps, n_cand, H, ub, lam, K_ws, dim_state, dim_control, rngseed)
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

%% ========================================================================
%  Deterministic palette by index (used by the accuracy and control figures)
%  ========================================================================
function c = hyb_palette(i)
    base = [0 0 0; 0 0.45 0.74; 0.49 0.18 0.56; 0.30 0.75 0.30; 0.85 0.60 0.20; ...
            0.25 0.35 0.60; 0.60 0.30 0.30; 0.40 0.60 0.50; 0.85 0.33 0.10; ...
            0.10 0.60 0.60; 0.60 0.10 0.60; 0.60 0.60 0.10; 0.35 0.35 0.35; ...
            0.70 0.40 0.15; 0.15 0.40 0.70; 0.40 0.15 0.70];
    c = base(mod(i - 1, size(base, 1)) + 1, :);
end
