function [results_ctrl, models] = control_evaluation()
%% CONTROL_EVALUATION  Closed-loop regulation using learned dynamics models
%   Compares each PhNN variant + MLP + LQR on the oscillator network.
%
%   [results_ctrl, models] = control_evaluation()

N_MASSES = 20; M_ACTUATORS = 5;
EXPANSION = 2; EPOCHS = 150; BATCH = 256; LR = 0.001;

dim_state = 2 * N_MASSES; dim_control = M_ACTUATORS;
dim_input = dim_state + dim_control; dim_output = dim_state;

fprintf('%s\n', repmat('=', 1, 70));
fprintf('CONTROL PERFORMANCE EVALUATION: Closed-loop Regulation\n');
fprintf('%s\n', repmat('=', 1, 70));

% --- Setup ---
rng(42);
osc = setup_oscillator(N_MASSES, M_ACTUATORS);
Q_lqr = eye(dim_state) * 0.1; R_lqr = eye(dim_control) * 0.01;
K_lqr = design_lqr(osc.A_mat, osc.B_mat, Q_lqr, R_lqr);

% --- Train dynamics models ---
fprintf('\n[1] Training dynamics models...\n');
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

idx = randperm(n_samples);
X_data = X_data(idx, :); Y_data = Y_data(idx, :);
n_tr = floor(n_samples * 0.7); n_va = floor(n_samples * 0.15);
Xtr = X_data(1:n_tr, :); Ytr = Y_data(1:n_tr, :);
Xva = X_data(n_tr+1:n_tr+n_va, :); Yva = Y_data(n_tr+1:n_tr+n_va, :);

mono = generate_monomial_indices(dim_input, EXPANSION);
n_mono = length(mono);

% Build masks (same as oscillator_control)
[A_val_pim, A_unc_pim] = build_oscillator_pim(N_MASSES, M_ACTUATORS, ...
    dim_state, dim_output, n_mono, mono, osc);

A_unc_tkm = ones(dim_output, n_mono, 'single');
for h = 1:n_mono
    midx = mono{h};
    has_s = any(midx <= dim_state); has_c = any(midx > dim_state);
    if has_s && has_c && length(midx) > 1, A_unc_tkm(:, h) = 0; end
end
A_unc_pt = A_unc_pim .* A_unc_tkm;

A_unc_rand = ones(dim_output, n_mono, 'single');
n_prune = floor((1-mean(A_unc_pim(:))) * numel(A_unc_rand));
ridx = randperm(numel(A_unc_rand), n_prune);
A_unc_rand(ridx) = 0;

% Train models
models = struct();
configs = {
    'unedited', [], [];
    'pim', A_val_pim, A_unc_pim;
    'tkm', [], A_unc_tkm;
    'pim_tkm', A_val_pim, A_unc_pt;
    'random', [], A_unc_rand;
};

for cfg = 1:size(configs, 1)
    name = configs{cfg, 1}; av = configs{cfg, 2}; au = configs{cfg, 3};
    if isempty(av), av = zeros(dim_output, n_mono, 'single'); end
    if isempty(au), au = ones(dim_output, n_mono, 'single'); end
    fprintf('  Training %s...\n', name);
    m = PhNNModel(dim_input, dim_output, mono, av, au);
    m.train(Xtr, Ytr, Xva, Yva, LR, EPOCHS, BATCH, EPOCHS+1);
    models.(name) = m;
end

fprintf('  Training MLP...\n');
mlp = MLPModel(dim_input, dim_output, [128, 64]);
mlp.train(Xtr, Ytr, Xva, Yva, LR, EPOCHS, BATCH);
models.mlp = mlp;

% ============================================================
% CLOSED-LOOP CONTROL EVALUATION
% ============================================================
fprintf('\n[2] Running closed-loop control evaluation (certainty-equivalence LQR)...\n');
rng(123);
n_trials = 30; n_steps = 500;

all_norms = struct();
for k = {'unedited', 'pim', 'tkm', 'pim_tkm', 'random', 'mlp', 'lqr'}
    all_norms.(k{1}) = [];
end

% Certainty-equivalence LQR: extract learned (A,B) from each model, then use
% the SAME LQR design as the ground-truth controller. Only the model differs.
K_hat = struct();
for k = {'unedited', 'pim', 'tkm', 'pim_tkm', 'random', 'mlp'}
    K_hat.(k{1}) = get_lqr_gain(models.(k{1}), dim_state, dim_control, Q_lqr, R_lqr);
end

for trial = 1:n_trials
    x0 = randn(dim_state, 1) * 3.0;
    if mod(trial, 5) == 0, fprintf('  Trial %d/%d...\n', trial, n_trials); end

    % LQR (ground-truth optimal)
    [~, ~, norms_lqr] = run_lqr_control(osc, K_lqr, x0, n_steps);
    all_norms.lqr(end+1, :) = norms_lqr'; %#ok<AGROW>

    % Learned controllers (same LQR algorithm, learned A,B)
    for k = {'unedited', 'pim', 'tkm', 'pim_tkm', 'random', 'mlp'}
        [~, ~, norms] = run_lqr_control(osc, K_hat.(k{1}), x0, n_steps);
        all_norms.(k{1})(end+1, :) = norms'; %#ok<AGROW>
    end
end

% Average results
results_ctrl = struct();
for k = {'lqr', 'unedited', 'pim', 'tkm', 'pim_tkm', 'random', 'mlp'}
    norms_arr = all_norms.(k{1});  % (n_trials, n_steps+1)
    r = struct();
    r.mean = mean(norms_arr, 1);
    r.std = std(norms_arr, 0, 1);
    r.final_val = mean(norms_arr(:, end));
    r.settling = find(r.mean < 1.0, 1, 'first');
    if isempty(r.settling), r.settling = NaN; end
    r.success = mean(norms_arr(:, end) < 1.0) * 100;
    results_ctrl.(k{1}) = r;
end

% ============================================================
% RESULTS
% ============================================================
fprintf('\n%s\n', repmat('=', 1, 85));
fprintf('CONTROL PERFORMANCE: Closed-Loop Regulation (%d trials, %d steps)\n', n_trials, n_steps);
fprintf('%s\n', repmat('=', 1, 85));

labels_ctrl = struct('lqr', 'LQR (Ground Truth)', 'unedited', 'Unedited PhNN', ...
    'pim', 'PIM-Edited PhNN', 'tkm', 'TKM-Edited PhNN', ...
    'pim_tkm', 'PIM+TKM PhNN', 'random', 'Random-Pruned PhNN', 'mlp', 'MLP Baseline');

fprintf('\n%-22s %-16s %-12s %-14s\n', 'Controller', 'Final ||x||', 'Settling', 'Success Rate');
fprintf('%s\n', repmat('-', 1, 65));
for k = {'lqr', 'unedited', 'pim', 'tkm', 'pim_tkm', 'random', 'mlp'}
    r = results_ctrl.(k{1});
    if isnan(r.settling)
        settle_str = 'N/A';
    else
        settle_str = sprintf('%d steps', r.settling);
    end
    fprintf('%-22s %-16.4f %-12s %.0f%%\n', labels_ctrl.(k{1}), r.final_val, settle_str, r.success);
end

% Key comparisons
fprintf('\n  LQR (optimal):           ||x||_final = %.4f\n', results_ctrl.lqr.final_val);
fprintf('  Unedited PhNN controller:  ||x||_final = %.4f\n', results_ctrl.unedited.final_val);
fprintf('  PIM-Edited controller:     ||x||_final = %.4f\n', results_ctrl.pim.final_val);
r_pim = results_ctrl.pim; r_un = results_ctrl.unedited;
fprintf('  PIM improvement over Unedited: %.1f%%\n', (r_un.final_val - r_pim.final_val)/r_un.final_val*100);
fprintf('  PIM success rate: %.0f%% vs Unedited: %.0f%%\n', r_pim.success, r_un.success);

% ============================================================
% FIGURES
% ============================================================
fprintf('\n[3] Generating control figures...\n');

COLORS = struct('lqr', [0 0 0], 'unedited', [0.85 0.33 0.10], 'pim', [0 0.45 0.74], ...
    'tkm', [0.93 0.69 0.13], 'pim_tkm', [0.49 0.18 0.56], ...
    'random', [0.64 0.08 0.18], 'mlp', [0.47 0.67 0.19]);

% Fig C1: State norm evolution
figure('Position', [100, 100, 1300, 550]);
subplot(1,2,1); hold on;
for k = {'lqr', 'pim', 'pim_tkm', 'unedited'}
    r = results_ctrl.(k{1}); c = COLORS.(k{1});
    steps = 0:length(r.mean)-1;
    semilogy(steps, r.mean, 'Color', c, 'LineWidth', 2.0, 'DisplayName', labels_ctrl.(k{1}));
    fill([steps, fliplr(steps)], [max(r.mean-r.std, 1e-10), fliplr(r.mean+r.std)], ...
        c, 'FaceAlpha', 0.12, 'EdgeColor', 'none');
end
xlabel('Time Step'); ylabel('||x|| (log)'); title('(a) Edited vs Unedited vs LQR');
legend('FontSize', 7); grid on; yline(1.0, ':', 'Color', [0.5 0.5 0.5]);

subplot(1,2,2); hold on;
for k = {'lqr', 'pim', 'mlp', 'random'}
    r = results_ctrl.(k{1}); c = COLORS.(k{1});
    steps = 0:length(r.mean)-1;
    semilogy(steps, r.mean, 'Color', c, 'LineWidth', 2.0, 'DisplayName', labels_ctrl.(k{1}));
    fill([steps, fliplr(steps)], [max(r.mean-r.std, 1e-10), fliplr(r.mean+r.std)], ...
        c, 'FaceAlpha', 0.12, 'EdgeColor', 'none');
end
xlabel('Time Step'); ylabel('||x|| (log)'); title('(b) PIM vs Baselines');
legend('FontSize', 7); grid on; yline(1.0, ':', 'Color', [0.5 0.5 0.5]);
sgtitle('Fig C1: Closed-Loop Regulation', 'FontWeight', 'bold', 'FontSize', 13);
saveas(gcf, 'fig/FigC1_Regulation.png');
close;
fprintf('  FigC1 saved.\n');

% Fig C2: Final state norm bar chart
figure('Position', [100, 100, 1000, 480]);
keys_bar = {'lqr', 'pim', 'pim_tkm', 'tkm', 'unedited', 'mlp', 'random'};
dns_bar = {'LQR', 'PIM', 'PIM+TKM', 'TKM', 'Unedited', 'MLP', 'Random'};
finals = zeros(1,7); successes = zeros(1,7);
cols_bar = zeros(7,3);
for i = 1:7
    r = results_ctrl.(keys_bar{i});
    finals(i) = r.final_val; successes(i) = r.success;
    cols_bar(i,:) = COLORS.(keys_bar{i});
end
b = bar(finals); b.FaceColor = 'flat';
for i = 1:7, b.CData(i,:) = cols_bar(i,:); end
set(gca, 'XTickLabel', dns_bar, 'YScale', 'log');
for i = 1:7
    text(i, finals(i)*1.5, sprintf('%.3f\n(%.0f%%)', finals(i), successes(i)), ...
        'HorizontalAlignment', 'center', 'FontSize', 7);
end
ylabel('Final ||x|| (log)'); title('Regulation Performance');
sgtitle('Fig C2: Final State Norm -- Lower is Better', 'FontWeight', 'bold', 'FontSize', 13);
saveas(gcf, 'fig/FigC2_FinalNorm.png');
close;
fprintf('  FigC2 saved.\n');

% Fig C3: Phase portrait
figure('Position', [100, 100, 1300, 800]);
x0_demo = randn(dim_state, 1) * 3.0;
n_plot = 500;
demo_keys = {'lqr', 'unedited', 'pim', 'pim_tkm', 'random', 'mlp'};
for ax_idx = 1:6
    subplot(2,3,ax_idx); hold on;
    key = demo_keys{ax_idx};
    if strcmp(key, 'lqr')
        [traj, ~, ~] = run_lqr_control(osc, K_lqr, x0_demo, n_plot);
    else
        [traj, ~, ~] = run_lqr_control(osc, K_hat.(key), x0_demo, n_plot);
    end
    plot(traj(:,1), traj(:, N_MASSES+1), 'Color', COLORS.(key), 'LineWidth', 1.5);
    scatter(traj(1,1), traj(1, N_MASSES+1), 40, 'k', 'filled', 'Marker', 'o');
    scatter(traj(end,1), traj(end, N_MASSES+1), 80, COLORS.(key), 'filled', 'Marker', '*');
    xline(0, 'Color', [0.5 0.5 0.5]); yline(0, 'Color', [0.5 0.5 0.5]);
    xlabel('$x_1$'); ylabel('$v_1$');
    title(labels_ctrl.(key));
end
sgtitle('Fig C3: Phase Portrait (Mass 1) -- Black=start, Star=end', 'FontWeight', 'bold', 'FontSize', 13);
saveas(gcf, 'fig/FigC3_PhasePortrait.png');
close;
fprintf('  FigC3 saved.\n');

% Fig C4: Ablation
figure('Position', [100, 100, 850, 480]);
abl_keys = {'unedited', 'tkm', 'pim_tkm', 'pim'};
abl_dns = {'Unedited', '+ TKM', '+ PIM+TKM', 'PIM (Best)'};
abl_vals = zeros(1,4); abl_cols = zeros(4,3);
for i = 1:4
    abl_vals(i) = results_ctrl.(abl_keys{i}).final_val;
    abl_cols(i,:) = COLORS.(abl_keys{i});
end
b = bar(abl_vals); b.FaceColor = 'flat';
for i = 1:4, b.CData(i,:) = abl_cols(i,:); end
set(gca, 'XTickLabel', abl_dns);
for i = 1:4
    text(i, abl_vals(i)*1.1, sprintf('%.4f', abl_vals(i)), 'HorizontalAlignment', 'center', 'FontSize', 10);
end
ylabel('Final ||x||'); title('Component Contributions');
for i = 2:4
    imp = (abl_vals(1)-abl_vals(i))/abl_vals(1)*100;
    text(i, abl_vals(i)*1.5, sprintf('%.1f%%', imp), 'HorizontalAlignment', 'center', 'FontSize', 8, 'Color', [0 0.5 0]);
end
sgtitle('Fig C4: Ablation Study', 'FontWeight', 'bold', 'FontSize', 13);
saveas(gcf, 'fig/FigC4_Ablation.png');
close;
fprintf('  FigC4 saved.\n');

fprintf('\nAll control figures saved.\n');
fprintf('%s\n', repmat('=', 1, 70));

% Save experiment data for offline plotting
out_dir = fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))), 'results');  % <repo>/results
if ~exist(out_dir, 'dir'), mkdir(out_dir); end
save(fullfile(out_dir, 'control_results.mat'), 'results_ctrl', 'models');
fprintf('  Results saved to results/control_results.mat\n');

end

%% ========================================================================
%  Certainty-equivalence LQR: extract learned (A,B), apply same LQR design
%  ========================================================================

function K_hat = get_lqr_gain(model, dim_state, dim_control, Q, R)
    % Extract the linear (A,B) from a learned model and design the same
    % LQR feedback as the ground-truth controller (certainty equivalence).
    if isa(model, 'PhNNModel')
        W_eff = model.A_value + model.A_uncertain .* model.W_learn;
        A_hat = double(W_eff(:, 1:dim_state));
        B_hat = double(W_eff(:, dim_state+1:dim_state+dim_control));
    elseif isa(model, 'MLPModel')
        [A_hat, B_hat] = linearize_mlp(model, dim_state, dim_control);
    else
        error('get_lqr_gain: unknown model class');
    end

    K_hat = zeros(dim_control, dim_state);
    % Guard: a learned (A,B) that is open-loop unstable or uncontrollable
    % means the model cannot support a stabilizing LQR -> zero gain.
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

function [A_hat, B_hat] = linearize_mlp(mlp, dim_state, dim_control)
    % Linearize the MLP around the origin: dy/dx = W3' * diag(g2) * W2' *
    % diag(g1) * W1', where g1/g2 are the ReLU activation patterns at x=0.
    W1 = mlp.weights{1}; b1 = mlp.biases{1};
    W2 = mlp.weights{2}; b2 = mlp.biases{2};
    W3 = mlp.weights{3};
    a1 = b1;                       % pre-activation of hidden layer 1 at x=0
    g1 = single(a1 > 0);           % ReLU'(a1)
    h1 = max(0, a1);
    a2 = h1 * W2 + b2;             % pre-activation of hidden layer 2
    g2 = single(a2 > 0);           % ReLU'(a2)
    J = W3' * diag(g2) * W2' * diag(g1) * W1';   % dim_out x dim_in
    A_hat = double(J(:, 1:dim_state));
    B_hat = double(J(:, dim_state+1:dim_state+dim_control));
end

function [traj, controls, state_norms] = run_lqr_control(osc, K_lqr, x0, n_steps)
    x = x0(:);
    traj = zeros(n_steps+1, length(x)); traj(1,:) = x';
    controls = zeros(n_steps, size(K_lqr,1));
    state_norms = zeros(1, n_steps+1); state_norms(1) = norm(x);

    for t = 1:n_steps
        u = -K_lqr * x;
        u = max(min(u, 2), -2);
        controls(t,:) = u';
        x = osc_step(osc, x, u);
        traj(t+1,:) = x';
        state_norms(t+1) = norm(x);
    end
end
