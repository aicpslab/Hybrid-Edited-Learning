function results = mpc_improved()
%% MPC_IMPROVED  MPC Ablation Study & Fair Baselines
%   Ablations:
%     A1: Unedited PhNN (no editing)
%     A2: PIM editing only
%     A3: TKM editing only
%     A4: PIM+TKM
%
%   Baselines:
%     B1: Standard MLP (3-layer)
%     B2: Random-pruned PhNN (same sparsity as TKM)
%
%   All methods use IDENTICAL data, training epochs, and evaluation.
%
%   results = mpc_improved()

EPOCHS = 150; BATCH = 256; LR = 0.001;

fprintf('%s\n', repmat('=', 1, 70));
fprintf('MPC Ablation Study & Baseline Comparison\n');
fprintf('%s\n', repmat('=', 1, 70));

% --- Setup ---
vehicle = setup_vehicle();
N_MPC = 2;  % prediction horizon
mpc = setup_mpc(vehicle, N_MPC);

dim_in = 4 * (N_MPC + 1);  % 16D
dim_out = 2;
expansion_order = 2;

% --- Generate Data ---
fprintf('\n[1/7] Generating MPC training data...\n');
n_total = 6000;
[Xtr, Utr, Xva, Uva, Xte, Ute] = generate_mpc_dataset(vehicle, mpc, n_total, 3, 42);
fprintf('  Train: %dx%d, Val: %dx%d, Test: %dx%d\n', ...
    size(Xtr,1), size(Xtr,2), size(Xva,1), size(Xva,2), size(Xte,1), size(Xte,2));

% --- Taylor Library ---
fprintf('\n[2/7] Building Taylor library...\n');
mono = generate_monomial_indices(dim_in, expansion_order);
n_mono = length(mono);
fprintf('  Input dim: %d, Monomials (r=%d): %d\n', dim_in, expansion_order, n_mono);

% --- Build Editing Masks ---
% PIM: all learnable (realistic scenario of limited physics knowledge)
A_unc_pim = ones(dim_out, n_mono, 'single');

% TKM: prune cross-horizon monomials
A_unc_tkm = ones(dim_out, n_mono, 'single');
for h = 1:n_mono
    midx = mono{h};
    steps_involved = unique(floor((midx - 1) / 4));  % 4 vars per step
    if length(steps_involved) > 1
        A_unc_tkm(:, h) = 0;
    end
end
tkm_sparsity = 1.0 - mean(A_unc_tkm(:));
fprintf('  TKM sparsity (cross-horizon pruning): %.1f%%\n', tkm_sparsity*100);

% Random pruning mask (matching TKM sparsity)
A_unc_random = ones(dim_out, n_mono, 'single');
n_prune = floor(tkm_sparsity * numel(A_unc_random));
ridx = randperm(numel(A_unc_random), n_prune);
A_unc_random(ridx) = 0;

% --- Train All Models ---
fprintf('\n[3/7] Training models...\n');
results = struct();

% A1: Unedited PhNN
fprintf('\n  [A1] Unedited PhNN\n');
m = PhNNModel(dim_in, dim_out, mono);
t0 = tic;
[tl, vl, bv] = m.train(Xtr, Utr, Xva, Uva, LR, EPOCHS, BATCH, EPOCHS+1);
dtime = toc(t0);
test_rmse = sqrt(mean((m.forward(Xte) - Ute).^2, 'all'));
results.unedited = struct('model', m, 'tl', tl, 'vl', vl, 'bv', bv, ...
    'rmse', test_rmse, 'params', m.n_learnable, 'sparsity', m.sparsity, 'time', dtime);
fprintf('  -> Val loss=%.4e, Test RMSE=%.4e, Params=%d, Time=%.0fs\n', bv, test_rmse, m.n_learnable, dtime);

% A2: PIM only
fprintf('\n  [A2] PIM-only\n');
m = PhNNModel(dim_in, dim_out, mono, [], A_unc_pim);
t0 = tic;
[tl, vl, bv] = m.train(Xtr, Utr, Xva, Uva, LR, EPOCHS, BATCH, EPOCHS+1);
dtime = toc(t0);
test_rmse = sqrt(mean((m.forward(Xte) - Ute).^2, 'all'));
results.pim_only = struct('model', m, 'tl', tl, 'vl', vl, 'bv', bv, ...
    'rmse', test_rmse, 'params', m.n_learnable, 'sparsity', m.sparsity, 'time', dtime);
fprintf('  -> Val loss=%.4e, Test RMSE=%.4e, Params=%d, Time=%.0fs\n', bv, test_rmse, m.n_learnable, dtime);

% A3: TKM only
fprintf('\n  [A3] TKM-only\n');
m = PhNNModel(dim_in, dim_out, mono, [], A_unc_tkm);
t0 = tic;
[tl, vl, bv] = m.train(Xtr, Utr, Xva, Uva, LR, EPOCHS, BATCH, EPOCHS+1);
dtime = toc(t0);
test_rmse = sqrt(mean((m.forward(Xte) - Ute).^2, 'all'));
results.tkm_only = struct('model', m, 'tl', tl, 'vl', vl, 'bv', bv, ...
    'rmse', test_rmse, 'params', m.n_learnable, 'sparsity', m.sparsity, 'time', dtime);
fprintf('  -> Val loss=%.4e, Test RMSE=%.4e, Params=%d, Time=%.0fs\n', bv, test_rmse, m.n_learnable, dtime);

% A4: PIM+TKM
fprintf('\n  [A4] PIM+TKM\n');
A_unc_pt = A_unc_pim .* A_unc_tkm;
m = PhNNModel(dim_in, dim_out, mono, [], A_unc_pt);
t0 = tic;
[tl, vl, bv] = m.train(Xtr, Utr, Xva, Uva, LR, EPOCHS, BATCH, EPOCHS+1);
dtime = toc(t0);
test_rmse = sqrt(mean((m.forward(Xte) - Ute).^2, 'all'));
results.pim_tkm = struct('model', m, 'tl', tl, 'vl', vl, 'bv', bv, ...
    'rmse', test_rmse, 'params', m.n_learnable, 'sparsity', m.sparsity, 'time', dtime);
fprintf('  -> Val loss=%.4e, Test RMSE=%.4e, Params=%d, Time=%.0fs\n', bv, test_rmse, m.n_learnable, dtime);

% B1: Standard MLP
fprintf('\n  [B1] Standard MLP\n');
mlp = MLPModel(dim_in, dim_out, [64, 32]);
t0 = tic;
[tl, vl, bv] = mlp.train(Xtr, Utr, Xva, Uva, LR, EPOCHS, BATCH);
dtime = toc(t0);
test_rmse = sqrt(mean((mlp.forward(Xte) - Ute).^2, 'all'));
results.mlp = struct('model', mlp, 'tl', tl, 'vl', vl, 'bv', bv, ...
    'rmse', test_rmse, 'params', mlp.n_total, 'sparsity', 0.0, 'time', dtime);
fprintf('  -> Val loss=%.4e, Test RMSE=%.4e, Params=%d, Time=%.0fs\n', bv, test_rmse, mlp.n_total, dtime);

% B2: Random-pruned PhNN
fprintf('\n  [B2] Random-pruned PhNN\n');
m = PhNNModel(dim_in, dim_out, mono, [], A_unc_random);
t0 = tic;
[tl, vl, bv] = m.train(Xtr, Utr, Xva, Uva, LR, EPOCHS, BATCH, EPOCHS+1);
dtime = toc(t0);
test_rmse = sqrt(mean((m.forward(Xte) - Ute).^2, 'all'));
results.random_prune = struct('model', m, 'tl', tl, 'vl', vl, 'bv', bv, ...
    'rmse', test_rmse, 'params', m.n_learnable, 'sparsity', m.sparsity, 'time', dtime);
fprintf('  -> Val loss=%.4e, Test RMSE=%.4e, Params=%d, Time=%.0fs\n', bv, test_rmse, m.n_learnable, dtime);

% --- Results Summary ---
fprintf('\n%s\n', repmat('=', 1, 90));
fprintf('RESULTS: MPC Ablation & Baseline Comparison\n');
fprintf('%s\n', repmat('=', 1, 90));

keys_order = {'unedited', 'pim_only', 'tkm_only', 'pim_tkm', 'mlp', 'random_prune'};
dns = {'A1: Unedited PhNN', 'A2: PIM only', 'A3: TKM only', 'A4: PIM+TKM', 'B1: MLP', 'B2: Random Prune'};

fprintf('\n%-28s %-14s %-14s %-12s %-10s %-8s\n', ...
    'Method', 'Test RMSE', 'Val Loss', 'Params', 'Sparsity', 'Time');
fprintf('%s\n', repmat('-', 1, 90));
for i = 1:length(keys_order)
    r = results.(keys_order{i});
    fprintf('%-28s %-14.6e %-14.6e %-12d %-9.1f%% %-7.0fs\n', ...
        dns{i}, r.rmse, r.bv, r.params, r.sparsity*100, r.time);
end

% Key comparisons
ru = results.unedited;
fprintf('\n  Ablation Analysis:\n');
for k = {'pim_only', 'tkm_only', 'pim_tkm'}
    r = results.(k{1});
    ratio = r.bv / max(ru.bv, 1e-15);
    fprintf('    %s: loss ratio vs Unedited = %.4f\n', k{1}, ratio);
end

rp = results.random_prune; rt = results.tkm_only;
fprintf('\n  TKM vs Random (both at %.1f%% sparsity):\n', tkm_sparsity*100);
fprintf('    TKM loss:      %.6e\n', rt.bv);
fprintf('    Random loss:   %.6e\n', rp.bv);
if rt.bv < rp.bv
    fprintf('    TKM improves over random by %.1f%%\n', (rp.bv-rt.bv)/rp.bv*100);
else
    fprintf('    Random actually outperforms TKM by %.1f%%\n', (rt.bv-rp.bv)/rt.bv*100);
end

% --- Figures ---
fprintf('\n[5/7] Generating figures...\n');

cols = [0 0 0; 0 0.45 0.74; 0.93 0.69 0.13; 0.49 0.18 0.56; 0.85 0.33 0.10; 0.47 0.67 0.19];

% Fig M1: Training curves
figure('Position', [100, 100, 1200, 480]);
subplot(1,2,1); hold on; subplot(1,2,2); hold on;
for i = 1:length(keys_order)
    r = results.(keys_order{i}); c = cols(i,:);
    subplot(1,2,1);
    smooth = conv(r.tl, ones(1,8)/8, 'valid');
    semilogy(smooth, 'Color', c, 'LineWidth', 1.2, 'DisplayName', dns{i});
    subplot(1,2,2);
    semilogy(r.vl, 'Color', c, 'LineWidth', 1.5, 'DisplayName', sprintf('%s (%.2e)', dns{i}, r.bv));
end
subplot(1,2,1); xlabel('Epoch'); ylabel('MSE'); title('Training Loss'); xlim([0, EPOCHS]); legend('FontSize', 7); grid on;
subplot(1,2,2); xlabel('Epoch'); ylabel('MSE'); title('Validation Loss'); xlim([0, EPOCHS]); legend('FontSize', 7); grid on;
sgtitle('Fig M1: MPC Training Curves', 'FontWeight', 'bold', 'FontSize', 13);
saveas(gcf, 'fig/FigM1_TrainingCurves.png');
close;
fprintf('  FigM1 saved.\n');

% Fig M2: Test RMSE bar chart
figure('Position', [100, 100, 1000, 480]);
rms = zeros(1,6);
for i = 1:6, rms(i) = results.(keys_order{i}).rmse; end
b = bar(rms); b.FaceColor = 'flat';
for i = 1:6, b.CData(i,:) = cols(i,:); end
set(gca, 'XTickLabel', dns, 'YScale', 'log');
for i = 1:6, text(i, rms(i)*1.1, sprintf('%.4f', rms(i)), 'HorizontalAlignment', 'center', 'FontSize', 8); end
ylabel('Test RMSE (log)'); title('MPC Approximation Accuracy');
sgtitle('Fig M2: Test RMSE', 'FontWeight', 'bold', 'FontSize', 13);
saveas(gcf, 'fig/FigM2_RMSE.png');
close;
fprintf('  FigM2 saved.\n');

% Fig M3: Parameter vs Accuracy
figure('Position', [100, 100, 800, 550]); hold on;
for i = 1:length(keys_order)
    r = results.(keys_order{i});
    scatter(r.params, r.rmse, 180, cols(i,:), 'filled', ...
        'MarkerEdgeColor', 'k', 'DisplayName', dns{i});
end
set(gca, 'XScale', 'log', 'YScale', 'log');
xlabel('Learnable Parameters'); ylabel('Test RMSE');
title('Parameter-Accuracy Trade-off'); legend('FontSize', 8); grid on;
sgtitle('Fig M3: Efficiency vs Accuracy', 'FontWeight', 'bold', 'FontSize', 13);
saveas(gcf, 'fig/FigM3_ParamTradeoff.png');
close;
fprintf('  FigM3 saved.\n');

% Fig M4: TKM vs Random
figure('Position', [100, 100, 1200, 480]);
subplot(1,2,1);
vals_loss = [ru.bv, rt.bv, rp.bv];
b = bar(vals_loss); b.FaceColor = 'flat';
b.CData(1,:) = cols(1,:); b.CData(2,:) = cols(3,:); b.CData(3,:) = cols(6,:);
set(gca, 'XTickLabel', {'Unedited', 'TKM', 'Random'}, 'YScale', 'log');
ylabel('Validation MSE'); title('Validation Loss');
for i = 1:3, text(i, vals_loss(i)*1.15, sprintf('%.4f', vals_loss(i)), 'HorizontalAlignment', 'center', 'FontSize', 9); end

subplot(1,2,2);
vals_rmse_vals = [ru.rmse, rt.rmse, rp.rmse];
b = bar(vals_rmse_vals); b.FaceColor = 'flat';
b.CData(1,:) = cols(1,:); b.CData(2,:) = cols(3,:); b.CData(3,:) = cols(6,:);
set(gca, 'XTickLabel', {'Unedited', 'TKM', 'Random'}, 'YScale', 'log');
ylabel('Test RMSE'); title('Test RMSE');
for i = 1:3, text(i, vals_rmse_vals(i)*1.15, sprintf('%.4f', vals_rmse_vals(i)), 'HorizontalAlignment', 'center', 'FontSize', 9); end
sgtitle('Fig M4: TKM (Structured) vs Random Pruning', 'FontWeight', 'bold', 'FontSize', 13);
saveas(gcf, 'fig/FigM4_TKMvsRandom.png');
close;
fprintf('  FigM4 saved.\n');

fprintf('\n%s\n', repmat('=', 1, 70));
fprintf('MPC EXPERIMENT COMPLETE\n');
fprintf('  All methods: %d epochs, batch=%d, lr=%.3f\n', EPOCHS, BATCH, LR);
fprintf('%s\n', repmat('=', 1, 70));

end

%% ========================================================================
%  Vehicle Model & MPC
%  ========================================================================

function vehicle = setup_vehicle()
    vehicle.L = 2.7;
    vehicle.dt = 0.2;
end

function x_next = vehicle_step(vehicle, x, u)
    px = x(1); py = x(2); v = x(3); psi = x(4);
    throttle = u(1); steer = u(2);
    px_next = px + v * cos(psi) * vehicle.dt;
    py_next = py + v * sin(psi) * vehicle.dt;
    v_next = v + throttle * vehicle.dt;
    psi_next = psi + v * tan(steer) / vehicle.L * vehicle.dt;
    x_next = [px_next; py_next; v_next; psi_next];
end

function mpc = setup_mpc(vehicle, N)
    mpc.vehicle = vehicle; mpc.N = N;
    mpc.Q = diag([10, 10, 1, 1]);
    mpc.R = diag([0.1, 0.1]);
    mpc.u_bounds = [-1.5, 4.0; -0.05, 0.05];  % [min; max] per control
end

function ref_traj = generate_ref_traj(n_steps, dt, scenario)
    t = (0:n_steps-1)' * dt;
    switch scenario
        case 'obstacle'
            x_ref = 2.0 * t;
            y_ref = 3.0 * sin(t * 0.5) .* (1 - exp(-t/5));
        case 'circle'
            x_ref = 15 * cos(t * 0.3);
            y_ref = 15 * sin(t * 0.3);
        case 'lane_change'
            x_ref = 3.0 * t;
            y_ref = 3.0 ./ (1 + exp(-(t-10)/2));
        otherwise
            x_ref = 2.0 * t;
            y_ref = zeros(size(t));
    end
    ref_traj = [x_ref, y_ref, ones(size(t))*2.0, zeros(size(t))];
end

function u_opt = mpc_solve(mpc, x_current, ref_window)
    n_x = 4; n_u = 2; N = mpc.N;
    n_vars = N * n_u;
    bounds = mpc.u_bounds;

    % Cost function (nested)
    function c = cost_fn(U_flat)
        U = reshape(U_flat, [n_u, N])';
        x_pred = x_current(:);
        total_cost = 0.0;
        for i = 1:N
            x_pred = vehicle_step(mpc.vehicle, x_pred, U(i,:)');
            if i <= size(ref_window, 1)
                dx = x_pred(1:2) - ref_window(i, 1:2)';
                total_cost = total_cost + dx' * mpc.Q(1:2,1:2) * dx + U(i,:) * mpc.R * U(i,:)';
            end
        end
        c = total_cost;
    end

    % Random shooting
    best_U = []; best_cost = inf;
    for s = 1:200
        U_cand = bounds(1,:)' + (bounds(2,:)' - bounds(1,:)') .* rand(n_vars, 1);
        c = cost_fn(U_cand);
        if c < best_cost
            best_cost = c; best_U = U_cand;
        end
    end

    % Local refinement
    U = best_U;
    sigma = 0.1;
    for s = 1:50
        U_cand = U + randn(n_vars, 1) * sigma;
        U_cand = max(min(U_cand, bounds(2,:)'), bounds(1,:)');
        c_cand = cost_fn(U_cand);
        if c_cand < best_cost
            best_cost = c_cand; U = U_cand;
            sigma = sigma * 1.1;
        else
            sigma = sigma * 0.95;
        end
        if sigma < 1e-6, break; end
    end

    u_opt = U(1:n_u)';
end

function [train_set, val_set, test_set] = generate_mpc_dataset(vehicle, mpc, n_samples, n_scenarios, seed)
    rng(seed);
    dt = vehicle.dt;
    n_per_scenario = floor(n_samples / n_scenarios);
    scenarios = {'obstacle', 'circle', 'lane_change'};

    all_X = []; all_U = [];

    for s_idx = 1:length(scenarios)
        scenario = scenarios{s_idx};
        ref = generate_ref_traj(300, dt, scenario);
        count = 0;
        while count < n_per_scenario
            t0 = randi([1, 200]);
            px0 = ref(t0, 1) + randn() * 2;
            py0 = ref(t0, 2) + randn() * 2;
            v0 = max(0.5, 2.0 + randn() * 1.0);
            psi0 = (rand() - 0.5) * pi/2;
            x = [px0; py0; v0; psi0];

            ref_window = ref(t0:min(t0+mpc.N, end), :);
            if size(ref_window, 1) < mpc.N + 1, continue; end

            try
                u_opt = mpc_solve(mpc, x, ref_window);
            catch
                continue;
            end

            % Build input vector
            input_vec = [];
            for i = 1:mpc.N+1
                dx = x - ref_window(i, :)';
                input_vec = [input_vec; dx];
            end

            all_X(end+1, :) = input_vec'; %#ok<AGROW>
            all_U(end+1, :) = u_opt;      %#ok<AGROW>
            count = count + 1;

            x = vehicle_step(vehicle, x, u_opt');
        end
    end

    X = single(all_X); U = single(all_U);
    idx = randperm(size(X, 1));
    X = X(idx, :); U = U(idx, :);
    n_tr = floor(size(X,1) * 0.7); n_va = floor(size(X,1) * 0.15);

    train_set = {X(1:n_tr, :), U(1:n_tr, :)};
    val_set   = {X(n_tr+1:n_tr+n_va, :), U(n_tr+1:n_tr+n_va, :)};
    test_set  = {X(n_tr+n_va+1:end, :), U(n_tr+n_va+1:end, :)};
end
