function results = hybrid_framework(N, dt, F, expansion_order, n_train, n_val, n_test, ...
    n_epochs, batch_size, learning_rate, var_threshold, eps_list, min_partition_samples, max_partitions_train, run_scaling)
%% HYBRID_FRAMEWORK  PCA + Maximum-Entropy Bisecting Hybrid Framework (Lorenz-96)
%   Implements the paper's Physics-Regulated Neural Hybrid System (NHS):
%     1. Feature projection  Theta(x) = Q * x' via PCA  (Algorithm step: ali_pca)
%     2. ME bisecting        recursive partition of the feature space, stopping
%                            when the Shannon-entropy variation dH < eps
%                            (Algorithm 1 in the paper)
%     3. Distributed editing train a PIM-edited sub-PhN per partition on its
%                            segmented data; the mode function delta(x) selects
%                            the active sub-network at inference
%   and benchmarks it against the monolithic (single-model) PhNNs.
%
%   results = hybrid_framework()
%   results = hybrid_framework(N, dt, F, expansion_order, n_train, n_val, n_test,
%                              n_epochs, batch_size, learning_rate, var_threshold,
%                              eps_list, min_partition_samples, max_partitions_train)
%
%   Inputs (all optional, with defaults):
%     N                    - Lorenz-96 dimension (default: 40)
%     dt                   - Time step (default: 0.01)
%     F                    - Forcing (default: 8.0)
%     expansion_order      - Taylor order r (default: 2)
%     n_train / n_val / n_test - sample counts (default: 30000 / 3000 / 3000)
%     n_epochs             - epochs per model (default: 200)
%     batch_size           - mini-batch (default: 256)
%     learning_rate        - Adam lr (default: 0.001)
%     var_threshold        - PCA cumulative-variance threshold for n_p (default 0.90)
%     eps_list             - ME-bisecting entropy thresholds to sweep (default)
%     min_partition_samples- min samples per partition (default: 100)
%     max_partitions_train - largest N actually trained (default: 16)
%     run_scaling          - also run the ambient-dimension scaling study (Prop 1)
%
%   Dependencies: generate_train_val_test_data, generate_monomial_indices,
%   build_lorenz96_pim, PhNNModel.

% =========================================================================
% Default parameters
% =========================================================================
if nargin < 1,  N = 40;                end
if nargin < 2,  dt = 0.01;             end
if nargin < 3,  F = 8.0;               end
if nargin < 4,  expansion_order = 2;   end
if nargin < 5,  n_train = 30000;       end
if nargin < 6,  n_val = 3000;          end
if nargin < 7,  n_test = 3000;         end
if nargin < 8,  n_epochs = 200;        end
if nargin < 9,  batch_size = 256;      end
if nargin < 10, learning_rate = 0.001; end
if nargin < 11, var_threshold = 0.90;  end
if nargin < 12, eps_list = [0.85, 0.60, 0.45, 0.35, 0.28, 0.22, 0.17, 0.13, 0.10, 0.08]; end
if nargin < 13, min_partition_samples = 100; end
if nargin < 14, max_partitions_train = 8;    end
if nargin < 15, run_scaling = true;    end
save_figures = true;

fprintf('%s\n', repmat('=', 1, 78));
fprintf('Hybrid Framework: PCA + ME-Bisecting + Distributed PIM Editing (Lorenz-96)\n');
fprintf('  N=%d, dt=%.2f, F=%.1f, r=%d, PCA variance threshold=%.2f\n', ...
    N, dt, F, expansion_order, var_threshold);
fprintf('%s\n', repmat('=', 1, 78));

% =========================================================================
% Step 1: Generate data (identical protocol to lorenz96_experiment, seed 42)
% =========================================================================
fprintf('\n[Step 1] Generating Lorenz-96 data...\n');
[train_traj, val_traj, test_traj] = generate_train_val_test_data(N, dt, F, n_train, n_val, n_test, 42);
X_train = single(train_traj(1:end-1, :));
Y_train = single(train_traj(2:end, :));
X_val   = single(val_traj(1:end-1, :));
Y_val   = single(val_traj(2:end, :));
X_test  = single(test_traj(1:end-1, :));
Y_test  = single(test_traj(2:end, :));
fprintf('  Train: %d, Val: %d, Test: %d\n', size(X_train,1), size(X_val,1), size(X_test,1));

% =========================================================================
% Step 2: Taylor monomials + PIM masks
% =========================================================================
fprintf('\n[Step 2] Building Taylor expansion and PIM masks...\n');
monomials_std = generate_monomial_indices(N, expansion_order);
[A_value_pim, A_uncertain_pim, pim_sparsity] = build_lorenz96_pim(N, dt, monomials_std);
fprintf('  Monomials (r=%d, %dD): %d, PIM sparsity: %.1f%%\n', ...
    expansion_order, N, length(monomials_std), pim_sparsity*100);

% =========================================================================
% Step 3: PCA feature projection  Theta(x) = Q * x'
% =========================================================================
fprintf('\n[Step 3] PCA feature projection...\n');
[pca_info, explained_cum] = build_pca(double(X_train), var_threshold);
n_p = pca_info.n_p;
fprintf('  %d PCs retain %.1f%% variance (n_p=%d, ambient n_x=%d)\n', ...
    n_p, explained_cum(n_p), n_p, N);

F_train = (double(X_train) - pca_info.mu) ./ pca_info.rangev * pca_info.Q';  % n_train x n_p

% =========================================================================
% Step 4: ME-bisecting partition of the feature space (Algorithm 1)
% =========================================================================
fprintf('\n[Step 4] Maximum-Entropy bisecting (sweeping eps)...\n');
n_eps = length(eps_list);
N_by_eps = zeros(1, n_eps);
boxes_by_eps = cell(1, n_eps);
for e = 1:n_eps
    boxes_by_eps{e} = me_bisect(F_train, eps_list(e), min_partition_samples);
    N_by_eps(e) = numel(boxes_by_eps{e});
    fprintf('  eps=%.2f -> N=%d partitions\n', eps_list(e), N_by_eps(e));
end

% Select the eps-configs whose partition counts are closest to targets
targets = [2, 4, 8];
sel = [];
for t = 1:length(targets)
    [~, k] = min(abs(N_by_eps - targets(t)));
    if N_by_eps(k) >= 2 && N_by_eps(k) <= max_partitions_train && ~any(sel == k)
        sel(end+1) = k; %#ok<AGROW>
    end
end
if isempty(sel)
    [~, k] = min(abs(N_by_eps - 4));
    if N_by_eps(k) >= 2 && N_by_eps(k) <= max_partitions_train, sel = k; end
end
fprintf('  Training configs: N = %s\n', mat2str(sort(N_by_eps(sel))));

% =========================================================================
% Step 5: Train monolithic baselines
% Same protocol as every hybrid sub-PhN: max n_epochs with early-stopping
% patience 25, so all methods share an identical epoch budget and the
% reported epochs/time reflect genuine convergence speed.
% =========================================================================
fprintf('\n[Step 5] Training monolithic baselines...\n');
patience = 25;

% Monolithic UNEDITED PhNN
fprintf('\n  [monolithic] Unedited PhNN...\n');
m_ued = PhNNModel(N, N, monomials_std);
t0 = tic;
[~, va_ued, v_ued] = m_ued.train(X_train, Y_train, X_val, Y_val, learning_rate, n_epochs, batch_size, patience);
t_ued = toc(t0);
n_e_ued = numel(va_ued);
[rmse_u, ~] = compute_autoregressive_rmse(m_ued, X_test, test_traj, 200);
res_ued = struct('N', 1, 'val_loss', v_ued, 'rmse50', rmse_u(50), 'rmse_by_step', rmse_u, ...
    'n_learnable', m_ued.n_learnable, 'n_total', m_ued.n_total, ...
    'sparsity', m_ued.sparsity, 'train_time', t_ued, 'epochs_used', n_e_ued);
res_ued.model = m_ued;
fprintf('  Unedited: val=%.4e, params=%d, RMSE@50=%.4g, epochs=%d, time=%.1fs\n', ...
    v_ued, m_ued.n_learnable, rmse_u(50), n_e_ued, t_ued);

% Monolithic PIM-Edited PhNN  == Hybrid with N=1
fprintf('\n  [monolithic] PIM-Edited PhNN (== Hybrid N=1)...\n');
m_pim = PhNNModel(N, N, monomials_std, A_value_pim, A_uncertain_pim);
t0 = tic;
[~, va_pim, v_pim] = m_pim.train(X_train, Y_train, X_val, Y_val, learning_rate, n_epochs, batch_size, patience);
t_pim = toc(t0);
n_e_pim = numel(va_pim);
res_pim = struct('N', 1, 'eps', NaN, 'val_loss', v_pim, ...
    'n_learnable', m_pim.n_learnable, 'n_total', m_pim.n_total, ...
    'sparsity', m_pim.sparsity, 'train_time', t_pim, 'epochs_used', n_e_pim, 'n_p', n_p);
[rmse_p, ~] = compute_autoregressive_rmse(m_pim, X_test, test_traj, 200);
res_pim.rmse50 = rmse_p(50);
res_pim.rmse_by_step = rmse_p;
res_pim.model = m_pim;
fprintf('  PIM-Edited: val=%.4e, params=%d, RMSE@50=%.4f, epochs=%d, time=%.1fs\n', ...
    v_pim, m_pim.n_learnable, rmse_p(50), n_e_pim, t_pim);

% =========================================================================
% Step 6: Train Hybrid configs (distributed PIM-edited sub-PhNs)
% =========================================================================
fprintf('\n[Step 6] Training Hybrid (PCA + ME bisecting + local PIM editing)...\n');
hybrid = [];
all_models = cell(1, length(sel));
all_boxes  = cell(1, length(sel));
for s = 1:length(sel)
    k = sel(s);
    Nk = N_by_eps(k);
    boxes = boxes_by_eps{k};
    fprintf('\n  [hybrid] N=%d (eps=%.2f)...\n', Nk, eps_list(k));

    % Segmented training set: W_i = { (x,t) : Theta(x) in P_i }
    b_train = assign_mode(F_train, boxes);
    b_val   = assign_mode((double(X_val) - pca_info.mu) ./ pca_info.rangev * pca_info.Q', boxes);
    models = cell(1, Nk);
    means_x = zeros(Nk, N);        % per-partition input center
    means_y = zeros(Nk, N);        % per-partition output center
    val_pred = zeros(size(Y_val), 'single');
    ok = false(size(Y_val, 1), 1);
    n_epochs_h = zeros(1, Nk);
    t0 = tic;
    for i = 1:Nk
        sxi = (b_train == i);
        if ~any(sxi)
            warning('Partition %d has no training samples', i);
            models{i} = []; continue;
        end
        % Center both input AND output at the partition's own mean: the local
        % Taylor expansion develops around the partition center, and the PIM
        % fixed coefficients (1-dt on the self term) remain exact because the
        % constant shift f(mu_x)-mu_y is absorbed by the learnable bias.
        % Removing the DC offset keeps the optimizer from being dominated by
        % a large constant term (input-only centering suffered from this).
        mu_x = mean(X_train(sxi, :), 1);
        mu_y = mean(Y_train(sxi, :), 1);
        means_x(i, :) = mu_x; means_y(i, :) = mu_y;
        svi = find(b_val == i);
        m_i = PhNNModel(N, N, monomials_std, A_value_pim, A_uncertain_pim);
        if ~isempty(svi)
            [~, va_i, ~] = m_i.train(X_train(sxi, :) - mu_x, Y_train(sxi, :) - mu_y, ...
                X_val(svi, :) - mu_x, Y_val(svi, :) - mu_y, ...
                learning_rate, n_epochs, batch_size, patience);
        else
            nv = min(50, size(X_val, 1));
            [~, va_i, ~] = m_i.train(X_train(sxi, :) - mu_x, Y_train(sxi, :) - mu_y, ...
                X_val(1:nv, :) - mu_x, Y_val(1:nv, :) - mu_y, ...
                learning_rate, n_epochs, batch_size, patience);
        end
        n_epochs_h(i) = numel(va_i);
        models{i} = m_i;
        fprintf('    sub-PhN %2d/%2d trained (%d samples, %d epochs)\n', i, Nk, sum(sxi), numel(va_i));
    end
    t_h = toc(t0);

    % One-step validation loss via the mode function delta(x)
    for i = 1:Nk
        sxi = (b_val == i);
        if ~isempty(models{i}) && any(sxi)
            val_pred(sxi, :) = models{i}.forward(X_val(sxi, :) - means_x(i, :)) + means_y(i, :);
            ok(sxi) = true;
        end
    end
    if any(ok)
        d2 = (val_pred(ok, :) - Y_val(ok, :)).^2;
        v_h = mean(d2(:));                       % scalar MSE over all samples & dims
    else
        v_h = NaN;
    end

    % Autoregressive RMSE with mode switching
    [rmse_h, ~] = hybrid_rollout_rmse(models, pca_info, boxes, means_x, means_y, test_traj, 200);
    r50 = rmse_h(50);

    hybrid = [hybrid; struct('N', Nk, 'eps', eps_list(k), 'val_loss', v_h, ...
        'rmse50', r50, 'rmse_by_step', rmse_h, ...
        'n_learnable', Nk * m_pim.n_learnable, 'n_total', Nk * m_pim.n_total, ...
        'sparsity', 1 - Nk*m_pim.n_learnable/(Nk*m_pim.n_total), ...
        'train_time', t_h, 'epochs_used', max(n_epochs_h), 'n_p', n_p, ...
        'means_x', means_x, 'means_y', means_y)]; %#ok<AGROW>
    fprintf('  Hybrid N=%d: val=%.4e, params=%d, RMSE@50=%.4f, epochs=%d, time=%.1fs\n', ...
        Nk, v_h, Nk*m_pim.n_learnable, r50, max(n_epochs_h), t_h);
    all_models{s} = models;
    all_boxes{s}  = boxes;
end
% Retain the trained sub-PhN models + partition boxes for the control stage
for s = 1:length(sel)
    hybrid(s).models = all_models{s};
    hybrid(s).boxes  = all_boxes{s};
end

% Ordinary Hybrid baseline: partitioning WITHOUT PIM editing (unedited
% sub-PhNs). Demonstrates that the distributed architecture alone does not
% win -- the matrix-guided editing is what makes the hybrid accurate.
ord_hybrid = [];
if ~isempty(sel)
    k0 = sel(1);                          % smallest-N config (bounded cost)
    Nk0 = N_by_eps(k0);
    boxes0 = boxes_by_eps{k0};
    b_tr0 = assign_mode(F_train, boxes0);
    b_va0 = assign_mode((double(X_val) - pca_info.mu) ./ pca_info.rangev * pca_info.Q', boxes0);
    models0 = cell(1, Nk0);
    means_x0 = zeros(Nk0, N); means_y0 = zeros(Nk0, N);
    val_pred0 = zeros(size(Y_val), 'single'); ok0 = false(size(Y_val, 1), 1);
    fprintf('\n  [hybrid-unedited] Ordinary Hybrid PhN (N=%d, no PIM editing)...\n', Nk0);
    n_epochs_o = zeros(1, Nk0);
    t0 = tic;
    for i = 1:Nk0
        sxi = (b_tr0 == i);
        if ~any(sxi)
            models0{i} = []; continue;
        end
        mu_x = mean(X_train(sxi, :), 1); mu_y = mean(Y_train(sxi, :), 1);
        means_x0(i, :) = mu_x; means_y0(i, :) = mu_y;
        m0 = PhNNModel(N, N, monomials_std);      % UNEDITED sub-PhN
        svi = find(b_va0 == i);
        if ~isempty(svi)
            [~, va0, ~] = m0.train(X_train(sxi, :) - mu_x, Y_train(sxi, :) - mu_y, ...
                X_val(svi, :) - mu_x, Y_val(svi, :) - mu_y, ...
                learning_rate, n_epochs, batch_size, patience);
        else
            nv = min(50, size(X_val, 1));
            [~, va0, ~] = m0.train(X_train(sxi, :) - mu_x, Y_train(sxi, :) - mu_y, ...
                X_val(1:nv, :) - mu_x, Y_val(1:nv, :) - mu_y, ...
                learning_rate, n_epochs, batch_size, patience);
        end
        n_epochs_o(i) = numel(va0);
        models0{i} = m0;
        fprintf('    unedited sub-PhN %2d/%2d trained (%d samples, %d epochs)\n', i, Nk0, sum(sxi), numel(va0));
    end
    t_ord = toc(t0);
    for i = 1:Nk0
        sxi = (b_va0 == i);
        if ~isempty(models0{i}) && any(sxi)
            val_pred0(sxi, :) = models0{i}.forward(X_val(sxi, :) - means_x0(i, :)) + means_y0(i, :);
            ok0(sxi) = true;
        end
    end
    if any(ok0)
        d2 = (val_pred0(ok0, :) - Y_val(ok0, :)).^2;
        v_ord = mean(d2(:));
    else
        v_ord = NaN;
    end
    [rmse_ord, ~] = hybrid_rollout_rmse(models0, pca_info, boxes0, means_x0, means_y0, test_traj, 200);
    ord_hybrid = struct('N', Nk0, 'val_loss', v_ord, 'rmse50', rmse_ord(50), ...
        'rmse_by_step', rmse_ord, 'n_learnable', Nk0*m_ued.n_learnable, ...
        'n_total', Nk0*m_ued.n_total, 'train_time', t_ord, 'epochs_used', max(n_epochs_o));
    ord_hybrid.models0 = models0;
    ord_hybrid.boxes0  = boxes0;
    ord_hybrid.means_x = means_x0;
    ord_hybrid.means_y = means_y0;
    fprintf('  Ordinary Hybrid (unedited) N=%d: val=%.4e, params=%d, RMSE@50=%.4f, epochs=%d, time=%.1fs\n', ...
        Nk0, v_ord, ord_hybrid.n_learnable, rmse_ord(50), ord_hybrid.epochs_used, t_ord);
end

% =========================================================================
% Step 7: Also try to enrich the table with TKM / PIM+TKM from the L96 run
% =========================================================================
res_tkm = []; res_pim_tkm = [];
mat_path = fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))), 'results', 'lorenz96_results.mat');  % <repo>/results (two dirs up from code/hybrid)
if exist(mat_path, 'file')
    ld = load(mat_path);
    if isfield(ld.results, 'tkm')
        res_tkm = ld.results.tkm;
    end
    if isfield(ld.results, 'pim_tkm')
        res_pim_tkm = ld.results.pim_tkm;
    end
    fprintf('\n[note] Loaded TKM / PIM+TKM numbers from lorenz96_results.mat\n');
end

% =========================================================================
% Step 8: Results summary table
% =========================================================================
fprintf('\n%s\n', repmat('=', 1, 78));
fprintf('[Step 8] RESULTS SUMMARY  (Lorenz-96, N=%d, r=%d)\n', N, expansion_order);
fprintf('%s\n', repmat('=', 1, 78));
fprintf('\n%-28s %-13s %-11s %-8s %-11s %-11s\n', 'Method', 'Val Loss', 'Params', 'Epochs', 'TrainTime(s)', 'RMSE@50');
fprintf('%s\n', repmat('-', 1, 84));
r50u = res_ued.rmse50; if isnan(r50u), r50u = Inf; end
fprintf('%-28s %-13.4e %-11d %-8d %-11.1f %-11.4g\n', 'Monolithic Unedited PhNN', ...
    res_ued.val_loss, res_ued.n_learnable, res_ued.epochs_used, res_ued.train_time, r50u);
if ~isempty(res_tkm)
    r50t = res_tkm.rmse_by_step(50); if isnan(r50t), r50t = Inf; end
    fprintf('%-28s %-13.4e %-11d %-8s %-11s %-11.4g\n', 'Monolithic TKM-Edited PhNN', ...
        res_tkm.best_val_loss, res_tkm.n_learnable, '--', '--', r50t);
end
fprintf('%-28s %-13.4e %-11d %-8d %-11.1f %-11.4g\n', 'Monolithic PIM-Edited PhNN (N=1)', ...
    res_pim.val_loss, res_pim.n_learnable, res_pim.epochs_used, res_pim.train_time, res_pim.rmse50);
if ~isempty(res_pim_tkm)
    r50p = res_pim_tkm.rmse_by_step(50); if isnan(r50p), r50p = Inf; end
    fprintf('%-28s %-13.4e %-11d %-8s %-11s %-11.4g\n', 'Monolithic PIM+TKM PhNN', ...
        res_pim_tkm.best_val_loss, res_pim_tkm.n_learnable, '--', '--', r50p);
end
for i = 1:size(hybrid, 1)
    h = hybrid(i);
    fprintf('%-28s %-13.4e %-11d %-8d %-11.1f %-11.4g\n', ...
        sprintf('Hybrid (PCA+ME, N=%d)', h.N), h.val_loss, h.n_learnable, h.epochs_used, h.train_time, h.rmse50);
end
if ~isempty(ord_hybrid)
    fprintf('%-28s %-13.4e %-11d %-8d %-11.1f %-11.4g\n', ...
        sprintf('Ordinary Hybrid (unedited, N=%d)', ord_hybrid.N), ...
        ord_hybrid.val_loss, ord_hybrid.n_learnable, ord_hybrid.epochs_used, ord_hybrid.train_time, ord_hybrid.rmse50);
end
fprintf('%s\n', repmat('-', 1, 84));
fprintf('TrainTime = cumulative single-worker time over ALL sub-models (not parallel wall-clock);\n');
fprintf('Epochs    = max epochs consumed by one sub-model under the shared patience-25 early stop.\n');

% =========================================================================
% Step 9: Visualization
% =========================================================================
fprintf('\n[Step 9] Generating visualizations...\n');
fig_dir = fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))), 'fig');  % <repo>/fig
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end

plot_pca_partitions(F_train, boxes_by_eps, pca_info, explained_cum, n_p, sel, save_figures);
plot_pareto(res_ued, res_pim, hybrid, ord_hybrid, save_figures);
plot_accuracy_vs_partitions(res_pim, hybrid, ord_hybrid, save_figures);
plot_results_bar(res_ued, res_tkm, res_pim, res_pim_tkm, hybrid, ord_hybrid, save_figures);
plot_training_cost(res_ued, res_pim, hybrid, ord_hybrid, save_figures);

% =========================================================================
% Step 10: Scaling study (Proposition 1) - partition count vs ambient dim
% =========================================================================
scaling = [];
if run_scaling
    fprintf('\n[Step 10] Scaling study: partition count vs ambient dimension...\n');
    eps_fixed = 0.20;
    scaling = scaling_demo(eps_fixed, var_threshold, min_partition_samples);
end

% =========================================================================
% Assemble results and save
% =========================================================================
results = struct();
results.pca_info  = pca_info;
results.res_ued   = res_ued;
results.res_pim   = res_pim;
results.hybrid    = hybrid;
results.ord_hybrid = ord_hybrid;
results.scaling   = scaling;
if ~isempty(res_tkm),    results.res_tkm    = res_tkm;    end
if ~isempty(res_pim_tkm),results.res_pim_tkm = res_pim_tkm; end

out_dir = fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))), 'results');  % <repo>/results
if ~exist(out_dir, 'dir'), mkdir(out_dir); end
save(fullfile(out_dir, 'hybrid_framework_results.mat'), 'results');
fprintf('\n%s\n', repmat('=', 1, 78));
fprintf('Hybrid Framework Experiment Complete!\n');
fprintf('  Results saved to results/hybrid_framework_results.mat\n');
fprintf('%s\n', repmat('=', 1, 78));
end

%% ========================================================================
%  PCA feature projection
%  ========================================================================
function [pca_info, explained_cum] = build_pca(X, var_threshold)
% build_pca  PCA of the mean- and range-normalized states, as in the paper.
%   The feature x'_j = (x_j - mu_j) / (max_j - min_j); the factor loading
%   matrix Q (n_p x d) comes from the eigendecomposition of the covariance.
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
% me_bisect  Recursively bisect the feature space, accepting a split only
%   when the Shannon-entropy variation dH = H_after - H_before >= eps.
%   The partition with the largest side (over all boxes and all dimensions)
%   is chosen at each step, matching [k,m] = argmax(width) in the paper.
%   Input : F (n x n_p) feature coordinates; eps >= 0; min_samples.
%   Output: leaves cell array of structs with fields lo, hi, idx.
    n = size(F, 1);
    lo0 = min(F, [], 1); hi0 = max(F, [], 1);
    active = struct('lo', lo0, 'hi', hi0, 'idx', (1:n)');
    leaves = {};

    while numel(active) > 0
        % Pick the active box with the largest side
        maxw = -inf; bi = 1;
        for b = 1:numel(active)
            w = max(active(b).hi - active(b).lo);
            if w > maxw, maxw = w; bi = b; end
        end
        box = active(bi);
        [~, d] = max(box.hi - box.lo);          % widest dimension
        mid = (box.lo(d) + box.hi(d)) / 2;      % bisect the interval
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

function [b_idx] = assign_mode(F, boxes)
% assign_mode  Mode function delta(x): assign each feature vector to the
%   partition whose box contains it (nearest box in Chebyshev sense).
    n = size(F, 1); nb = numel(boxes);
    d = zeros(n, nb);
    for b = 1:nb
        lo = boxes{b}.lo; hi = boxes{b}.hi;
        tmp = max(lo - F, F - hi);      % elementwise overhang per dimension
        tmp = max(tmp, [], 2);          % Chebyshev distance to the box
        d(:, b) = tmp;
    end
    [~, b_idx] = min(d, [], 2);
end

%% ========================================================================
%  Distributed autoregressive rollout with mode switching
%  ========================================================================
function [rmse_by_step, rmse_std] = hybrid_rollout_rmse(models, pca_info, boxes, means_x, means_y, trajectory, horizon)
% hybrid_rollout_rmse  Multi-step autoregressive RMSE of the hybrid NHS.
%   At each step the mode function selects the active sub-PhN from the
%   PCA feature of the current state; the selected sub-PhN predicts x(k+1)
%   after centering the state at the partition's expansion point and adding
%   back the partition's output mean.
    if nargin < 7, horizon = 200; end
    dim_out = size(trajectory, 2);
    K = 1;
    n_traj = size(trajectory, 1);
    max_start = n_traj - K - horizon + 1;
    if max_start < 1
        rmse_by_step = NaN(1, horizon); rmse_std = NaN(1, horizon);
        return;
    end
    n_test_points = min(20, max_start);
    start_indices = randperm(max_start, n_test_points);
    all_errors = zeros(n_test_points, horizon);

    for s = 1:n_test_points
        st = start_indices(s);
        x = trajectory(st, :);
        for h = 1:horizon
            feat = (double(x) - pca_info.mu) ./ pca_info.rangev * pca_info.Q'; % 1 x n_p
            b = assign_mode(feat, boxes);
            if isempty(models{b})
                xp = x;
            else
                xc = double(x) - means_x(b, :);          % local expansion center
                xp = models{b}.forward(single(xc)) + means_y(b, :);
            end
            true_state = trajectory(st + h, :);
            all_errors(s, h) = sqrt(mean((xp(:) - true_state(:)).^2));
            x = xp;
        end
    end
    rmse_by_step = mean(all_errors, 1);
    rmse_std = std(all_errors, 0, 1);
end

%% ========================================================================
%  Scaling study (Proposition 1): partition count vs ambient dimension
%  ========================================================================
function scaling = scaling_demo(eps_fixed, var_threshold, min_partition_samples)
% scaling_demo  For ambient dimensions n_x in {40,60,80}, generate Lorenz-96
%   data, project to the intrinsic feature space via PCA, run ME bisecting at
%   a FIXED eps, and record the number of partitions N. Contrasts with the
%   parameter count of a monolithic unedited PhN, which grows combinatorially.
    nx_list = [40, 60, 80];
    N_part = zeros(1, length(nx_list));
    n_p_list = zeros(1, length(nx_list));
    mono_params = zeros(1, length(nx_list));
    for j = 1:length(nx_list)
        nx = nx_list(j);
        n_train_small = 6000;
        [train_traj, ~, ~] = generate_train_val_test_data(nx, 0.01, 8.0, n_train_small, 400, 400, 100+j);
        X = double(train_traj(1:end-1, :));
        [pca_info, ~] = build_pca(X, var_threshold);
        Ff = (X - pca_info.mu) ./ pca_info.rangev * pca_info.Q';
        boxes = me_bisect(Ff, eps_fixed, min_partition_samples);
        N_part(j) = numel(boxes);
        n_p_list(j) = pca_info.n_p;
        mono_params(j) = get_expanded_dim(nx, 2) * nx;
        fprintf('  n_x=%3d: n_p=%2d (%.1f%% var), N=%3d partitions, monolithic params=%d\n', ...
            nx, pca_info.n_p, pca_info.explained_cum(pca_info.n_p), N_part(j), mono_params(j));
    end
    scaling = struct('nx_list', nx_list, 'N_part', N_part, 'n_p_list', n_p_list, ...
        'mono_params', mono_params, 'eps', eps_fixed);

    % Figure
    figure('Position', [100, 100, 900, 500]);
    yyaxis left;
    bar(nx_list, N_part, 0.5, 'FaceColor', [0 0.45 0.74], 'FaceAlpha', 0.7);
    ylabel('Partition count N (ME bisecting, fixed eps)');
    set(gca, 'YColor', [0 0.45 0.74]);
    hold on;
    plot(nx_list, N_part, 'o', 'Color', [0 0.45 0.74], 'LineWidth', 2, 'MarkerSize', 8);
    yyaxis right;
    plot(nx_list, mono_params, 's-', 'Color', [0.85 0.33 0.10], 'LineWidth', 2, 'MarkerSize', 8);
    ylabel('Monolithic unedited PhN parameters (log)');
    set(gca, 'YScale', 'log', 'YColor', [0.85 0.33 0.10]);
    xlabel('Ambient dimension n_x');
    title(sprintf('Scalability (Prop. 1): N scales with intrinsic dim, not ambient (eps=%.2f)', eps_fixed));
    grid on;
    legend({'Hybrid: partition count N', 'Monolithic unedited: params'}, 'Location', 'northwest');
    if ~exist('fig','dir'), mkdir('fig'); end
    saveas(gcf, 'fig/Hybrid_Scalability.png');
end

%% ========================================================================
%  Visualization helpers
%  ========================================================================
function total = get_expanded_dim(dim, order)
    total = 0;
    for r = 1:order
        total = total + nchoosek(dim + r - 1, r);
    end
end

function plot_pca_partitions(F_train, boxes_by_eps, pca_info, explained_cum, n_p, sel, save_figures)
    figure('Position', [100, 100, 1300, 500]);
    subplot(1, 2, 1);
    plot(1:length(explained_cum), explained_cum, 'b-o', 'LineWidth', 2);
    hold on;
    plot(n_p, explained_cum(n_p), 'ro', 'MarkerSize', 10, 'LineWidth', 2);
    xlabel('Principal component index'); ylabel('Cumulative variance (%)');
    title(sprintf('PCA: %d PCs retain %.1f%% variance', n_p, explained_cum(n_p)));
    grid on;

    subplot(1, 2, 2);
    hold on;
    F1 = F_train(:, 1); F2 = F_train(:, 2);
    boxes = boxes_by_eps{sel(1)};
    Nk = numel(boxes);
    cmap = lines(Nk);
    b_idx = assign_mode(F_train, boxes);
    for b = 1:Nk
        m = (b_idx == b);
        plot(F1(m), F2(m), '.', 'Color', cmap(b, :), 'MarkerSize', 3);
        rectangle('Position', [boxes{b}.lo(1), boxes{b}.lo(2), ...
            boxes{b}.hi(1)-boxes{b}.lo(1), boxes{b}.hi(2)-boxes{b}.lo(2)], ...
            'EdgeColor', 'k', 'LineWidth', 1);
    end
    xlabel('PC 1'); ylabel('PC 2');
    title(sprintf('ME-bisecting partitions in feature space (N=%d)', Nk));
    grid on;

    if save_figures
        if ~exist('fig','dir'), mkdir('fig'); end
        saveas(gcf, 'fig/Hybrid_PCA_Partitions.png');
    end
end

function plot_pareto(res_ued, res_pim, hybrid, ord_hybrid, save_figures)
    figure('Position', [100, 100, 800, 600]);
    hold on;
    pts = cell(0, 3);
    pts(end+1, :) = {'Monolithic Unedited', res_ued.n_learnable, res_ued.val_loss};
    pts(end+1, :) = {'Monolithic PIM (N=1)', res_pim.n_learnable, res_pim.val_loss};
    for i = 1:size(hybrid, 1)
        pts(end+1, :) = {sprintf('Hybrid N=%d', hybrid(i).N), ...
            hybrid(i).n_learnable, hybrid(i).val_loss}; %#ok<AGROW>
    end
    if ~isempty(ord_hybrid)
        pts(end+1, :) = {sprintf('Ordinary Hybrid (unedited, N=%d)', ord_hybrid.N), ...
            ord_hybrid.n_learnable, ord_hybrid.val_loss}; %#ok<AGROW>
    end
    for i = 1:size(pts, 1)
        if strcmp(pts{i,1}, 'Monolithic Unedited')
            c = [0.85 0.33 0.10]; mk = 's';
        elseif strcmp(pts{i,1}, 'Monolithic PIM (N=1)')
            c = [0 0.45 0.74]; mk = 'd';
        elseif ~isempty(ord_hybrid) && strcmp(pts{i,1}, sprintf('Ordinary Hybrid (unedited, N=%d)', ord_hybrid.N))
            c = [0.85 0.60 0.20]; mk = '^';
        else
            c = [0.49 0.18 0.56]; mk = 'o';
        end
        plot(pts{i,2}, pts{i,3}, mk, 'Color', c, 'MarkerSize', 10, ...
            'MarkerFaceColor', c, 'DisplayName', pts{i,1});
    end
    set(gca, 'XScale', 'log', 'YScale', 'log');
    xlabel('Total learnable parameters'); ylabel('One-step validation loss (MSE)');
    title('Accuracy vs Parameter Budget (Lorenz-96, N=40)');
    legend('Location', 'best'); grid on;
    if save_figures
        if ~exist('fig','dir'), mkdir('fig'); end
        saveas(gcf, 'fig/Hybrid_Pareto.png');
    end
end

function plot_accuracy_vs_partitions(res_pim, hybrid, ord_hybrid, save_figures)
    figure('Position', [100, 100, 900, 500]);
    Ns = [1; zeros(size(hybrid,1),1)];
    loss = [res_pim.val_loss; [hybrid.val_loss]'];
    rmse50 = [NaN; [hybrid.rmse50]'];
    for i = 1:size(hybrid,1), Ns(i+1) = hybrid(i).N; end

    subplot(1, 2, 1);
    plot(Ns, loss, 'o-', 'LineWidth', 2, 'Color', [0 0.45 0.74], 'MarkerSize', 8);
    xlabel('Number of partitions N'); ylabel('One-step validation loss (MSE)');
    set(gca, 'YScale', 'log', 'XScale', 'log');
    title('Hybrid accuracy vs partitions'); grid on;

    subplot(1, 2, 2);
    plot(Ns(2:end), rmse50(2:end), 's-', 'LineWidth', 2, 'Color', [0.49 0.18 0.56], 'MarkerSize', 8);
    xlabel('Number of partitions N'); ylabel('Autoregressive RMSE @ 50 steps');
    set(gca, 'YScale', 'log', 'XScale', 'log');
    title('Long-horizon accuracy vs partitions'); grid on;

    if ~isempty(ord_hybrid)
        subplot(1, 2, 1); hold on;
        plot(ord_hybrid.N, ord_hybrid.val_loss, '^', 'MarkerSize', 11, ...
            'MarkerFaceColor', [0.85 0.60 0.20], 'Color', [0.85 0.60 0.20]);
        subplot(1, 2, 2); hold on;
        plot(ord_hybrid.N, ord_hybrid.rmse50, '^', 'MarkerSize', 11, ...
            'MarkerFaceColor', [0.85 0.60 0.20], 'Color', [0.85 0.60 0.20]);
    end

    if save_figures
        if ~exist('fig','dir'), mkdir('fig'); end
        saveas(gcf, 'fig/Hybrid_AccuracyVsPartitions.png');
    end
end

function plot_results_bar(res_ued, res_tkm, res_pim, res_pim_tkm, hybrid, ord_hybrid, save_figures)
    names = {'Unedited PhNN'};
    vals = res_ued.val_loss;
    cs = [0.85 0.33 0.10];
    if ~isempty(res_tkm), names{end+1} = 'TKM'; vals(end+1) = res_tkm.best_val_loss; cs(end+1,:) = [0.93 0.69 0.13]; end %#ok<AGROW>
    if ~isempty(res_pim), names{end+1} = 'PIM (N=1)'; vals(end+1) = res_pim.val_loss; cs(end+1,:) = [0 0.45 0.74]; end %#ok<AGROW>
    if ~isempty(res_pim_tkm), names{end+1} = 'PIM+TKM'; vals(end+1) = res_pim_tkm.best_val_loss; cs(end+1,:) = [0.49 0.18 0.56]; end %#ok<AGROW>
    for i = 1:size(hybrid, 1)
        names{end+1} = sprintf('Hybrid N=%d', hybrid(i).N); %#ok<AGROW>
        vals(end+1) = hybrid(i).val_loss; %#ok<AGROW>
        cs(end+1,:) = [0.30 0.75 0.30]; %#ok<AGROW>
    end
    if ~isempty(ord_hybrid)
        names{end+1} = sprintf('Ord Hybrid N=%d', ord_hybrid.N); %#ok<AGROW>
        vals(end+1) = ord_hybrid.val_loss; %#ok<AGROW>
        cs(end+1,:) = [0.85 0.60 0.20]; %#ok<AGROW>
    end

    figure('Position', [100, 100, 1200, 500]);
    b = bar(1:numel(vals), vals, 'FaceAlpha', 0.7);
    b.FaceColor = 'flat';
    for i = 1:numel(vals), b.CData(i, :) = cs(i, :); end
    set(gca, 'YScale', 'log', 'XTickLabel', names, 'XTickLabelRotation', 25);
    xlabel('Method'); ylabel('One-step validation loss (MSE, log)');
    title('Lorenz-96: Hybrid Framework vs Monolithic Methods');
    for i = 1:numel(vals)
        text(i, vals(i)*1.6, sprintf('%.2e', vals(i)), 'HorizontalAlignment', 'center', 'FontSize', 8);
    end
    grid on;
    if save_figures
        if ~exist('fig','dir'), mkdir('fig'); end
        saveas(gcf, 'fig/Hybrid_Results.png');
    end
end

function plot_training_cost(res_ued, res_pim, hybrid, ord_hybrid, save_figures)
% PLOT_TRAINING_COST  Wall-clock time and epochs used per method.
% All methods share the identical epoch budget (max n_epochs, patience 25),
% so the bars show genuine convergence-speed / computational-cost differences.
    names = {'Unedited','PIM (N=1)'};
    times = [res_ued.train_time, res_pim.train_time];
    eps_  = [res_ued.epochs_used, res_pim.epochs_used];
    for i = 1:size(hybrid, 1)
        names{end+1} = sprintf('Hybrid N=%d', hybrid(i).N); %#ok<AGROW>
        times(end+1) = hybrid(i).train_time; %#ok<AGROW>
        eps_(end+1)  = hybrid(i).epochs_used; %#ok<AGROW>
    end
    if ~isempty(ord_hybrid)
        names{end+1} = sprintf('OrdHyb N=%d', ord_hybrid.N); %#ok<AGROW>
        times(end+1) = ord_hybrid.train_time; %#ok<AGROW>
        eps_(end+1)  = ord_hybrid.epochs_used; %#ok<AGROW>
    end

    figure('Position', [100, 100, 720, 560]);
    subplot(2,1,1);
    bar(1:numel(times), times, 0.6, 'FaceColor', [0.30 0.45 0.75]);
    grid on;
    set(gca, 'XTick', 1:numel(names), 'XTickLabel', names, 'XTickLabelRotation', 30);
    ylabel('Cumulative training time, single worker (s)');
    title('Cumulative training cost under identical epoch budget (max 200, early-stop patience 25)');
    for i = 1:numel(times)
        text(i, times(i), sprintf('%.1f s', times(i)), 'HorizontalAlignment','center', ...
            'VerticalAlignment','bottom', 'FontSize', 8);
    end
    subplot(2,1,2);
    bar(1:numel(eps_), eps_, 0.6, 'FaceColor', [0.85 0.33 0.30]);
    grid on;
    set(gca, 'XTick', 1:numel(names), 'XTickLabel', names, 'XTickLabelRotation', 30);
    ylabel('Epochs used (early stop)');
    ylim([0, max(eps_) * 1.25 + 10]);
    for i = 1:numel(eps_)
        text(i, eps_(i), sprintf('%d', eps_(i)), 'HorizontalAlignment','center', ...
            'VerticalAlignment','bottom', 'FontSize', 8);
    end
    if save_figures
        if ~exist('fig','dir'), mkdir('fig'); end
        saveas(gcf, 'fig/Hybrid_TrainingCost.png');
    end
end
