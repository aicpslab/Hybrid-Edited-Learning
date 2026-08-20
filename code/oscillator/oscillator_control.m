function results = oscillator_control()
%% OSCILLATOR_CONTROL  Coupled Oscillator Network Control -- PIM/TKM Editing
%   Benchmark contrasting with Lorenz-96:
%     Lorenz-96:  Sparse ring coupling, dynamics-only, 40D
%     Oscillator: Sparse ring coupling, dynamics+CONTROL, 40D
%
%   N coupled spring-mass-damper oscillators on a ring. M actuated.
%   State: [x_1,...,x_N, v_1,...,v_N] in R^{2N}
%   Control: [u_1,...,u_M] in R^{M}
%
%   results = oscillator_control()

%% Parameters
N_MASSES = 20;      % 20 masses -> 40D state (same as Lorenz-96!)
M_ACTUATORS = 5;    % 5 control inputs
EXPANSION = 2;
EPOCHS = 150; BATCH = 256; LR = 0.001;

dim_state = 2 * N_MASSES;   % 40D
dim_control = M_ACTUATORS;
dim_input = dim_state + dim_control;  % 45D: state + control
dim_output = dim_state;              % predict next state

fprintf('%s\n', repmat('=', 1, 70));
fprintf('Coupled Oscillator Network: N=%d (40D state), M=%d controls\n', N_MASSES, M_ACTUATORS);
fprintf('  PIM: Sparse ring topology (same as Lorenz-96)\n');
fprintf('  TKM: Second-order ODE (Markov in state-space)\n');
fprintf('  Task: Learn dynamics AND approximate LQR controller\n');
fprintf('%s\n', repmat('=', 1, 70));

% --- Setup ---
fprintf('\n[1/5] Setting up oscillator network & LQR controller...\n');
rng(42);
osc = setup_oscillator(N_MASSES, M_ACTUATORS);

% Design LQR
Q_lqr = eye(dim_state) * 0.1; R_lqr = eye(dim_control) * 0.01;
K_lqr = design_lqr(osc.A_mat, osc.B_mat, Q_lqr, R_lqr);
fprintf('  LQR gain K: %dx%d, ||K|| = %.2f\n', size(K_lqr,1), size(K_lqr,2), norm(K_lqr));

% --- Generate Data ---
fprintf('\n[2/5] Generating training data (random initial states + LQR control)...\n');
n_samples = 8000;
X_data = zeros(n_samples, dim_input, 'single');
Y_data = zeros(n_samples, dim_output, 'single');

for i = 1:n_samples
    x = randn(dim_state, 1) * 2.0;
    u = -K_lqr * x + randn(dim_control, 1) * 0.05;
    u = max(min(u, 2.0), -2.0);
    x_next = osc_step(osc, x, u);
    X_data(i, :) = [x; u]';
    Y_data(i, :) = x_next';
end

% Split
idx = randperm(n_samples);
X_data = X_data(idx, :); Y_data = Y_data(idx, :);
n_tr = floor(n_samples * 0.7); n_va = floor(n_samples * 0.15);
Xtr = X_data(1:n_tr, :); Ytr = Y_data(1:n_tr, :);
Xva = X_data(n_tr+1:n_tr+n_va, :); Yva = Y_data(n_tr+1:n_tr+n_va, :);
Xte = X_data(n_tr+n_va+1:end, :); Yte = Y_data(n_tr+n_va+1:end, :);
fprintf('  Train: %dx%d, Val: %dx%d, Test: %dx%d\n', ...
    size(Xtr,1), size(Xtr,2), size(Xva,1), size(Xva,2), size(Xte,1), size(Xte,2));

% --- Taylor Library ---
fprintf('\n[3/5] Building Taylor library and editing masks...\n');
mono = generate_monomial_indices(dim_input, EXPANSION);
n_mono = length(mono);
fprintf('  Input: %dD, Monomials (r=%d): %d\n', dim_input, EXPANSION, n_mono);

% --- Build PIM for Oscillator Network ---
[A_val_pim, A_unc_pim] = build_oscillator_pim(N_MASSES, M_ACTUATORS, dim_state, dim_output, n_mono, mono, osc);
pim_sparsity = 1.0 - mean(A_unc_pim(:));
fprintf('  PIM sparsity: %.1f%% (ring topology, same structure as Lorenz-96)\n', pim_sparsity*100);

% --- Build TKM mask ---
A_unc_tkm = ones(dim_output, n_mono, 'single');
for h = 1:n_mono
    midx = mono{h};
    has_state = any(midx <= dim_state);
    has_control = any(midx > dim_state);
    if has_state && has_control && length(midx) > 1
        A_unc_tkm(:, h) = 0;
    end
end
tkm_sparsity = 1.0 - mean(A_unc_tkm(:));
fprintf('  TKM sparsity (state*control cross-terms): %.1f%%\n', tkm_sparsity*100);

% Combined
A_unc_pt = A_unc_pim .* A_unc_tkm;
fprintf('  PIM+TKM combined sparsity: %.1f%%\n', (1-mean(A_unc_pt(:)))*100);

% Random mask (matching PIM sparsity)
A_unc_rand = ones(dim_output, n_mono, 'single');
n_prune = floor(pim_sparsity * numel(A_unc_rand));
ridx = randperm(numel(A_unc_rand), n_prune);
A_unc_rand(ridx) = 0;

% --- Train All Models ---
fprintf('\n[4/5] Training models (%d epochs, batch=%d, lr=%.3f)...\n', EPOCHS, BATCH, LR);
results = struct();

configs = {
    'unedited', 'Unedited PhNN', [], [];
    'pim',      'PIM-Edited PhNN', A_val_pim, A_unc_pim;
    'tkm',      'TKM-Edited PhNN', [], A_unc_tkm;
    'pim_tkm',  'PIM+TKM PhNN', A_val_pim, A_unc_pt;
    'random',   'Random-Pruned PhNN', [], A_unc_rand;
};

for cfg = 1:size(configs, 1)
    key = configs{cfg, 1}; label = configs{cfg, 2};
    av = configs{cfg, 3}; au = configs{cfg, 4};

    if isempty(av), av = zeros(dim_output, n_mono, 'single'); end
    if isempty(au), au = ones(dim_output, n_mono, 'single'); end

    fprintf('\n  [%s] %s...\n', key, label);
    m = PhNNModel(dim_input, dim_output, mono, av, au);
    t0 = tic;
    [tl, vl, bv] = m.train(Xtr, Ytr, Xva, Yva, LR, EPOCHS, BATCH, EPOCHS+1);
    dtime = toc(t0);
    test_rmse = sqrt(mean((m.forward(Xte) - Yte).^2, 'all'));
    results.(key) = struct('model', m, 'tl', tl, 'vl', vl, 'bv', bv, ...
        'rmse', test_rmse, 'params', m.n_learnable, 'sparsity', m.sparsity, ...
        'time', dtime, 'label', label);
    fprintf('  -> Val loss=%.4e, RMSE=%.4e, Params=%d, Sparsity=%.1f%%\n', ...
        bv, test_rmse, m.n_learnable, m.sparsity*100);
end

% MLP baseline
fprintf('\n  [mlp] MLP Baseline...\n');
mlp = MLPModel(dim_input, dim_output, [128, 64]);
t0 = tic;
[tl, vl, bv] = mlp.train(Xtr, Ytr, Xva, Yva, LR, EPOCHS, BATCH);
dtime = toc(t0);
test_rmse = sqrt(mean((mlp.forward(Xte) - Yte).^2, 'all'));
results.mlp = struct('model', mlp, 'tl', tl, 'vl', vl, 'bv', bv, ...
    'rmse', test_rmse, 'params', mlp.n_total, 'sparsity', 0.0, ...
    'time', dtime, 'label', 'MLP Baseline');
fprintf('  -> Val loss=%.4e, RMSE=%.4e, Params=%d\n', bv, test_rmse, mlp.n_total);

% --- Results ---
fprintf('\n%s\n', repmat('=', 1, 90));
fprintf('[5/5] RESULTS: Coupled Oscillator Network Control\n');
fprintf('  PIM structure: Sparse ring (same as Lorenz-96)\n');
fprintf('%s\n', repmat('=', 1, 90));

keys_plot = {'unedited', 'pim', 'tkm', 'pim_tkm', 'random', 'mlp'};
fprintf('\n%-28s %-14s %-14s %-12s %-10s\n', ...
    'Method', 'Val Loss', 'Test RMSE', 'Params', 'Sparsity');
fprintf('%s\n', repmat('-', 1, 90));
for i = 1:length(keys_plot)
    r = results.(keys_plot{i});
    fprintf('%-28s %-14.6e %-14.6e %-12d %-9.1f%%\n', ...
        r.label, r.bv, r.rmse, r.params, r.sparsity*100);
end

% Key comparisons
ru = results.unedited; rp = results.pim;
fprintf('\n  === Cross-Experiment: Oscillator vs Lorenz-96 ===\n');
fprintf('  Oscillator PIM/Unedited val loss ratio:  %.4f\n', rp.bv / max(ru.bv, 1e-15));
fprintf('  Oscillator PIM sparsity:  %.1f%%\n', pim_sparsity*100);

% TKM vs Random
rtkm = results.tkm; rrand = results.random;
fprintf('\n  === TKM vs Random Pruning ===\n');
fprintf('  TKM val loss:        %.6e\n', rtkm.bv);
fprintf('  Random (same sparsity): %.6e\n', rrand.bv);
if rtkm.bv < rrand.bv
    fprintf('  TKM improvement: %.1f%%\n', (rrand.bv - rtkm.bv)/rrand.bv*100);
else
    fprintf('  Random actually BETTER by %.1f%%\n', (rtkm.bv - rrand.bv)/rtkm.bv*100);
end

% ============================================================
% FIGURES
% ============================================================
fprintf('\nGenerating figures...\n');

colors_plot = [0.85 0.33 0.10; 0 0.45 0.74; 0.93 0.69 0.13; ...
               0.49 0.18 0.56; 0.64 0.08 0.18; 0.47 0.67 0.19];

% Fig O1: Training curves
figure('Position', [100, 100, 1200, 480]);
subplot(1,2,1); hold on; subplot(1,2,2); hold on;
for i = 1:length(keys_plot)
    r = results.(keys_plot{i}); c = colors_plot(i,:);
    subplot(1,2,1);
    smooth = conv(r.tl, ones(1,8)/8, 'valid');
    semilogy(smooth, 'Color', c, 'LineWidth', 1.2, 'DisplayName', r.label);
    subplot(1,2,2);
    semilogy(r.vl, 'Color', c, 'LineWidth', 1.5, 'DisplayName', sprintf('%s (%.2e)', r.label, r.bv));
end
subplot(1,2,1); xlabel('Epoch'); ylabel('MSE'); title('Training Loss'); xlim([0, EPOCHS]); legend('FontSize', 6.5); grid on;
subplot(1,2,2); xlabel('Epoch'); ylabel('MSE'); title('Validation Loss'); xlim([0, EPOCHS]); legend('FontSize', 6.5); grid on;
sgtitle('Fig O1: Oscillator Network -- Training Curves (40D state + 5D control)', 'FontWeight', 'bold', 'FontSize', 13);
saveas(gcf, 'fig/FigO1_TrainingCurves.png');
close;
fprintf('  FigO1 saved.\n');

% Fig O2: RMSE
figure('Position', [100, 100, 1000, 480]);
rms = zeros(1, length(keys_plot));
for i = 1:length(keys_plot), rms(i) = results.(keys_plot{i}).rmse; end
b = bar(rms); b.FaceColor = 'flat';
for i = 1:length(keys_plot), b.CData(i,:) = colors_plot(i,:); end
set(gca, 'XTickLabel', cellfun(@(x) results.(x).label, keys_plot, 'UniformOutput', false), 'YScale', 'log');
for i = 1:length(rms), text(i, rms(i)*1.1, sprintf('%.4f', rms(i)), 'HorizontalAlignment', 'center', 'FontSize', 8); end
ylabel('Test RMSE (log)'); title('Oscillator Network: Prediction Accuracy');
sgtitle('Fig O2: RMSE Comparison', 'FontWeight', 'bold', 'FontSize', 13);
saveas(gcf, 'fig/FigO2_RMSE.png');
close;
fprintf('  FigO2 saved.\n');

% Fig O3: Parameter-Accuracy trade-off
figure('Position', [100, 100, 800, 550]); hold on;
for i = 1:length(keys_plot)
    r = results.(keys_plot{i});
    scatter(r.params, r.rmse, 180, colors_plot(i,:), 'filled', ...
        'MarkerEdgeColor', 'k', 'DisplayName', r.label);
end
set(gca, 'XScale', 'log', 'YScale', 'log');
xlabel('Learnable Parameters'); ylabel('Test RMSE');
title('Parameter-Accuracy Trade-off'); legend('FontSize', 8); grid on;
sgtitle('Fig O3: Efficiency vs Accuracy', 'FontWeight', 'bold', 'FontSize', 13);
saveas(gcf, 'fig/FigO3_ParamTradeoff.png');
close;
fprintf('  FigO3 saved.\n');

% Fig O4: Cross-experiment comparison
figure('Position', [100, 100, 1200, 480]);
subplot(1,2,1);
l96_losses = [5.27e-1, 4.66e-5, 7.91e-1, 2.76e-5];  % Unedited, PIM, TKM, PIM+TKM (Table tab:l96)
l96_colors = colors_plot(1:4, :);
b = bar(l96_losses); b.FaceColor = 'flat';
for i = 1:4, b.CData(i,:) = l96_colors(i,:); end
set(gca, 'XTickLabel', {'Unedited', 'PIM', 'TKM', 'PIM+TKM'}, 'YScale', 'log');
ylabel('Val Loss (log)'); title('(a) Lorenz-96 (40D dynamics only)');
for i = 1:4
    v = l96_losses(i);
    if v < 0.1
        text(i, v*1.15, sprintf('%.2e', v), 'HorizontalAlignment', 'center', 'FontSize', 8);
    else
        text(i, v*1.15, sprintf('%.2f', v), 'HorizontalAlignment', 'center', 'FontSize', 8);
    end
end

subplot(1,2,2);
osc_losses = zeros(1,4);
for i = 1:4, osc_losses(i) = results.(keys_plot{i}).bv; end
b = bar(osc_losses); b.FaceColor = 'flat';
for i = 1:4, b.CData(i,:) = colors_plot(i,:); end
set(gca, 'XTickLabel', {'Unedited', 'PIM', 'TKM', 'PIM+TKM'}, 'YScale', 'log');
ylabel('Val Loss (log)'); title('(b) Oscillator Network (40D + 5D control)');
for i = 1:4, text(i, osc_losses(i)*1.15, sprintf('%.2e', osc_losses(i)), 'HorizontalAlignment', 'center', 'FontSize', 8); end
sgtitle('Fig O4: Cross-Experiment -- PIM Effect (Same Ring Topology)', 'FontWeight', 'bold', 'FontSize', 13);
saveas(gcf, 'fig/FigO4_Lorenz96vsOscillator.png');
close;
fprintf('  FigO4 saved.\n');

% Fig O5: Weight matrices
figure('Position', [100, 100, 1600, 420]);
for i = 1:4
    subplot(1,4,i);
    m = results.(keys_plot{i}).model;
    W = abs(m.A_value + m.A_uncertain .* m.W_learn);
    n_show = min(150, size(W,2));
    imagesc(W(:, 1:n_show)); colormap('hot'); colorbar;
    xlabel('Monomial'); title(['(' char('a'+i-1) ') ' results.(keys_plot{i}).label]);
    if i==1, ylabel('Output Dim'); end
end
sgtitle('Fig O5: Weight Matrix |W| -- Oscillator Network (40D+5D, r=2)', 'FontWeight', 'bold', 'FontSize', 13);
saveas(gcf, 'fig/FigO5_Weights.png');
close;
fprintf('  FigO5 saved.\n');

fprintf('All figures saved.\n');
fprintf('%s\n', repmat('=', 1, 70));

% Save experiment data for offline plotting
out_dir = fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))), 'results');  % <repo>/results
if ~exist(out_dir, 'dir'), mkdir(out_dir); end
save(fullfile(out_dir, 'oscillator_results.mat'), 'results');
fprintf('  Results saved to results/oscillator_results.mat\n');

end

%% ========================================================================
%  Oscillator Network Setup
%  ========================================================================

function osc = setup_oscillator(N, M)
    osc.N = N; osc.M = M; osc.dt = 0.05;
    osc.m = 0.5 + rand(N,1) * 0.5;    % masses [0.5, 1.0]
    osc.k = 2.0 + rand(N,1) * 1.0;    % spring constants [2, 3]
    osc.c = 0.3 + rand(N,1) * 0.2;    % damping [0.3, 0.5]
    osc.d = 0.1 + rand(N,1) * 0.1;    % friction [0.1, 0.2]

    % Actuator placement: evenly spaced
    osc.actuated = false(N, 1);
    osc.actuated(round(linspace(1, N, M))) = true;
    osc.B_mat = zeros(2*N, M);
    act_idx = find(osc.actuated);
    for j = 1:M
        osc.B_mat(N + act_idx(j), j) = osc.dt / osc.m(act_idx(j));
    end

    % Build linear dynamics matrix
    osc.A_mat = build_A_mat(osc);
end

function A = build_A_mat(osc)
    N = osc.N; dt = osc.dt;
    A = zeros(2*N, 2*N);
    % Position update: x(k+1) = x(k) + dt*v(k)
    A(1:N, 1:N) = eye(N);
    A(1:N, N+1:end) = dt * eye(N);
    % Velocity update
    for i = 1:N
        ip = mod(i, N) + 1;       % i+1 (cyclic)
        im = mod(i-2, N) + 1;     % i-1 (cyclic)

        A(N+i, i)    = A(N+i, i)    - dt * (osc.k(i) + osc.k(im)) / osc.m(i);
        A(N+i, im)   = A(N+i, im)   + dt * osc.k(im) / osc.m(i);
        A(N+i, ip)   = A(N+i, ip)   + dt * osc.k(i) / osc.m(i);
        A(N+i, N+i)  = A(N+i, N+i)  + 1.0 - dt * (osc.c(i) + osc.c(im) + osc.d(i)) / osc.m(i);
        A(N+i, N+im) = A(N+i, N+im) + dt * osc.c(im) / osc.m(i);
        A(N+i, N+ip) = A(N+i, N+ip) + dt * osc.c(i) / osc.m(i);
    end
end

function x_next = osc_step(osc, x, u)
    N = osc.N; dt = osc.dt;
    pos = x(1:N); vel = x(N+1:end);
    u_full = zeros(N, 1);
    act_idx = find(osc.actuated);
    for j = 1:length(act_idx)
        u_full(act_idx(j)) = u(j);
    end

    acc = zeros(N, 1);
    for i = 1:N
        ip = mod(i, N) + 1; im = mod(i-2, N) + 1;
        F_spring = osc.k(i)*(pos(ip)-pos(i)) + osc.k(im)*(pos(im)-pos(i));
        F_damper = osc.c(i)*(vel(ip)-vel(i)) + osc.c(im)*(vel(im)-vel(i));
        F_friction = -osc.d(i) * vel(i);
        F_control = u_full(i);
        acc(i) = (F_spring + F_damper + F_friction + F_control) / osc.m(i);
    end

    pos_next = pos + dt * vel;
    vel_next = vel + dt * acc;
    x_next = [pos_next; vel_next];
end

%% ========================================================================
%  LQR Design
%  ========================================================================

function K = design_lqr(A, B, Q, R)
    % Discrete-time algebraic Riccati equation via iteration
    P = Q;
    for iter = 1:200
        P = Q + A' * P * A - A' * P * B * ((R + B' * P * B) \ (B' * P * A));
    end
    K = (R + B' * P * B) \ (B' * P * A);
end

%% ========================================================================
%  PIM for Oscillator Network
%  ========================================================================

function [A_val_pim, A_unc_pim] = build_oscillator_pim(N_MASSES, M_ACTUATORS, dim_state, dim_output, n_mono, mono, osc)
    A_val_pim = zeros(dim_output, n_mono, 'single');
    A_unc_pim = zeros(dim_output, n_mono, 'single');

    for out_i = 1:dim_output
        i_mass = mod(out_i - 1, N_MASSES) + 1;
        ip = mod(i_mass, N_MASSES) + 1;
        im = mod(i_mass - 2, N_MASSES) + 1;

        % Determine relevant input variables
        if out_i <= N_MASSES  % Position output
            relevant_vars = [i_mass, N_MASSES + i_mass];  % x_i, v_i
        else  % Velocity output
            relevant_vars = [im, i_mass, ip, ...
                N_MASSES+im, N_MASSES+i_mass, N_MASSES+ip];
        end

        % Add control inputs if this mass is actuated
        if osc.actuated(i_mass)
            act_idx = find(osc.actuated);
            ctrl_idx = dim_state + find(act_idx == i_mass);
            relevant_vars = [relevant_vars, ctrl_idx];
        end

        for h = 1:n_mono
            midx = mono{h};
            if all(ismember(midx, relevant_vars))
                A_unc_pim(out_i, h) = 1;
            end
            % Known linear terms
            if length(midx) == 1 && midx(1) == i_mass && out_i <= N_MASSES
                A_val_pim(out_i, h) = 1.0;  % x_i -> x_i
            elseif length(midx) == 1 && midx(1) == N_MASSES + i_mass && out_i <= N_MASSES
                A_val_pim(out_i, h) = osc.dt;  % v_i -> x_i coeff = dt
            end
        end
    end
end

% MLPModel is defined in MLPModel.m (standalone class file)
