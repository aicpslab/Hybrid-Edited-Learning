function diag_shooting()
%% DIAG_SHOOTING  Audit of the long-horizon shooting-MPC regulation result.
%   Is the 300-step plateau (||x|| -> ~2.4) a MODEL/EXPERIMENT bug, or an
%   inherent property of the greedy random-shooting MPC controller?
%
%   Decisive test: run shooting MPC with the TRUE plant as the rollout
%   predictor (a PERFECT model).  If perfect-model shooting also plateaus,
%   the controller design is the cause.  If it regulates, the learned
%   models or their wrappers are the cause.
%
%   Also checks: plant open-loop/closed-loop stability, the linear identity
%   A*x+B*u == osc_step, and certainty-equivalence LQR from the same model.

clear; close all;
addpath(fileparts(mfilename('fullpath')));

S = load(fullfile('results', 'oscillator_hybrid_results.mat'));
r = S.results;
dim_state = 2 * r.meta.N_MASSES; dim_control = r.meta.M_ACTUATORS;
Q = eye(dim_state) * 0.1; R = eye(dim_control) * 0.01;

osc = setup_oscillator(r.meta.N_MASSES, r.meta.M_ACTUATORS);
A = osc.A_mat; B = osc.B_mat;
K = design_lqr(A, B, Q, R);

fprintf('=== Plant / controller sanity ===\n');
fprintf('  open-loop max|eig(A)|         = %.6f\n', max(abs(eig(A))));
fprintf('  closed-loop max|eig(A-BK)|    = %.6f\n', max(abs(eig(A - B * K))));
xt = randn(dim_state, 1); ut = randn(dim_control, 1) * 0.5;
err = norm(A * xt + B * ut - osc_step(osc, xt, ut));
fprintf('  ||A*x+B*u - osc_step||        = %.3e  (plant is linear)\n', err);

% batched perfect predictor (exact linear dynamics)
pred_true = @(X, U) X * A' + U * B';

rng(777); x0 = randn(1, dim_state) * 3.0;
fprintf('  ||x_0|| = %.3f\n\n', norm(x0));

% --- 1) perfect-model shooting, SAME settings as experiment (N_CAND=400,H=5) ---
[traj_p1, ~] = osc_shoot(pred_true, osc, x0', 300, 400, 5, 2.0, 0.01, dim_state, dim_control, 1);
n1 = sqrt(sum(traj_p1.^2, 2));
fprintf('1. Perfect model  (N=400, H=5,  300 steps):  final=%.4f   norms@60/120/300=%.3f/%.3f/%.3f\n', ...
    n1(end), n1(61), n1(121), n1(301));

% --- 2) perfect model, more candidates + longer horizon ---
[traj_p2, ~] = osc_shoot(pred_true, osc, x0', 300, 1500, 8, 2.0, 0.01, dim_state, dim_control, 2);
n2 = sqrt(sum(traj_p2.^2, 2));
fprintf('2. Perfect model  (N=1500, H=8, 300 steps):  final=%.4f   norms@60/120/300=%.3f/%.3f/%.3f\n', ...
    n2(end), n2(61), n2(121), n2(301));

% --- 3) perfect model, longer horizon only ---
[traj_p3, ~] = osc_shoot(pred_true, osc, x0', 300, 400, 20, 2.0, 0.01, dim_state, dim_control, 3);
n3 = sqrt(sum(traj_p3.^2, 2));
fprintf('3. Perfect model  (N=400, H=20, 300 steps):  final=%.4f   norms@60/120/300=%.3f/%.3f/%.3f\n', ...
    n3(end), n3(61), n3(121), n3(301));

% --- 4) perfect model with LQR warm-start candidate ---
[traj_p4, ~] = osc_shoot_lqrws(pred_true, osc, x0', 300, 400, 5, 2.0, 0.01, K, dim_state, dim_control, 4);
n4 = sqrt(sum(traj_p4.^2, 2));
fprintf('4. Perfect model  (N=400, H=5, +LQR warm-start, 300):  final=%.4f   norms@60/120/300=%.3f/%.3f/%.3f\n', ...
    n4(end), n4(61), n4(121), n4(301));

% --- 5) perfect model, 1000 steps (does it EVER decay?) ---
[traj_p5, ~] = osc_shoot(pred_true, osc, x0', 1000, 400, 5, 2.0, 0.01, dim_state, dim_control, 5);
n5 = sqrt(sum(traj_p5.^2, 2));
fprintf('5. Perfect model  (N=400, H=5, 1000 steps):  final=%.4f\n', n5(end));

% --- 6) learned single PIM (deg-2) shooting, reproduce the experiment ---
m_pim = r.res_pim.model;
pred_pim = @(X, U) m_pim.forward(single([X U]));
[traj_m, ~] = osc_shoot(pred_pim, osc, x0', 300, 400, 5, 2.0, 0.01, dim_state, dim_control, 6);
nm = sqrt(sum(traj_m.^2, 2));
fprintf('6. Learned single PIM (deg-2, N=400,H=5, 300):  final=%.4f   norms@60/120/300=%.3f/%.3f/%.3f\n', ...
    nm(end), nm(61), nm(121), nm(301));

% --- 7) certainty-equivalence LQR from the SAME model ---
W_eff = double(m_pim.A_value + m_pim.A_uncertain .* m_pim.W_learn);
A_hat = W_eff(:, 1:dim_state); B_hat = W_eff(:, dim_state + 1:dim_state + dim_control);
K_pim = design_lqr(A_hat, B_hat, Q, R);
[~, nrm_lqr] = osc_lqr_run(osc, K_pim, x0', 300, 2.0, dim_state, dim_control);
fprintf('7. Certainty-equiv LQR (single PIM, 300 steps):  final=%.6f   norms@60/120/300=%.3f/%.3f/%.3f\n', ...
    nrm_lqr(end), nrm_lqr(61), nrm_lqr(121), nrm_lqr(301));

fprintf('\nDone.\n');
end

%% =========================================================================
%  Greedy random-shooting MPC (identical to oscillator_hybrid_control.m)
%  =========================================================================
function [traj, U] = osc_shoot(pred, osc, x0, n_steps, n_cand, H, ub, lam, dim_state, dim_control, rngseed)
    rng(rngseed);
    x = double(x0(:));
    traj = zeros(n_steps + 1, dim_state); traj(1, :) = x';
    U = zeros(n_steps, dim_control);
    Uprev = single(zeros(1, dim_control));
    for t = 1:n_steps
        U_cand = single(-ub + 2*ub*rand(n_cand, dim_control));
        U_cand(1, :) = single(zeros(1, dim_control));
        U_cand(2, :) = Uprev;
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

function [traj, U] = osc_shoot_lqrws(pred, osc, x0, n_steps, n_cand, H, ub, lam, K, dim_state, dim_control, rngseed)
    rng(rngseed);
    x = double(x0(:));
    traj = zeros(n_steps + 1, dim_state); traj(1, :) = x';
    U = zeros(n_steps, dim_control);
    Uprev = single(zeros(1, dim_control));
    for t = 1:n_steps
        U_cand = single(-ub + 2*ub*rand(n_cand, dim_control));
        U_cand(1, :) = single(zeros(1, dim_control));
        U_cand(2, :) = Uprev;
        U_cand(3, :) = single(-K * x);
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

function [traj, norms] = osc_lqr_run(osc, K, x0, n_steps, ub, dim_state, dim_control)
    x = double(x0(:));
    traj = zeros(n_steps + 1, dim_state); traj(1, :) = x';
    norms = zeros(1, n_steps + 1); norms(1) = norm(x);
    for t = 1:n_steps
        u = -K * x; u = max(min(u, ub), -ub);
        x = osc_step(osc, x, u);
        traj(t + 1, :) = x'; norms(t + 1) = norm(x);
    end
end
