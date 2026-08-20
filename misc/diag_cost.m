function diag_cost
%% DIAG_COST  Pin down WHY PIM CE-LQR achieves lower true cost than nominal LQR.

N = 20; dim_state = 2*N; dim_control = 5;
S = load('results/control_results.mat');
models = S.models;
osc = setup_oscillator(N, dim_control);
Q = eye(dim_state)*0.1; R = eye(dim_control)*0.01;

K_lqr = design_lqr(osc.A_mat, osc.B_mat, Q, R);

% ---- extract learned K for pim / pim_tkm ----
Ks = struct();
Ks.lqr = K_lqr;
for k = {'pim','pim_tkm'}
    m = models.(k{1});
    W_eff = m.A_value + m.A_uncertain .* m.W_learn;
    A_h = double(W_eff(:,1:dim_state)); B_h = double(W_eff(:,dim_state+1:dim_state+dim_control));
    Ks.(k{1}) = design_lqr(A_h, B_h, Q, R);
end

% ---- 1) gain magnitude ----
fprintf('\n=== gain & closed-loop spectral radius ===\n');
fprintf('%-10s %-14s %-14s\n','ctrl','||K||_fro','rho(A-BK)');
for k = {'lqr','pim','pim_tkm'}
    Kk = Ks.(k{1});
    rho = max(abs(eig(osc.A_mat - osc.B_mat*Kk)));
    fprintf('%-10s %-14.4f %-14.4f\n', k{1}, norm(Kk,'fro'), rho);
end

% ---- 2) Riccati convergence: 200 vs 2000 ----
fprintf('\n=== Riccati convergence (200 vs 2000 iters) ===\n');
P = Q; K200 = design_lqr(osc.A_mat,osc.B_mat,Q,R);
for iter=1:2000
    P = Q + osc.A_mat'*P*osc.A_mat - osc.A_mat'*P*osc.B_mat*((R+osc.B_mat'*P*osc.B_mat)\(osc.B_mat'*P*osc.A_mat));
end
K2000 = (R + osc.B_mat'*P*osc.B_mat)\(osc.B_mat'*P*osc.A_mat);
fprintf('||K(200)-K(2000)||_fro = %.6f   (0 => converged)\n', norm(K200-K2000,'fro'));

% ---- 3) saturation count on one nominal-LQR trial ----
fprintf('\n=== saturation frequency (|u| hitting +-2) ===\n');
rng(123); x0 = randn(dim_state,1)*3;
for k = {'lqr','pim'}
    Kk = Ks.(k{1}); x = x0; sat = 0; umax = 0;
    for t = 1:500
        u = max(min(-Kk*x,2),-2);
        umax = max(umax, max(abs(u)));
        if any(abs(abs(u)-2) < 1e-9), sat = sat + 1; end
        x = osc_step(osc,x,u);
    end
    fprintf('%-10s saturated steps: %3d / 500,  max|u| = %.3f\n', k{1}, sat, umax);
end

% ---- 4) cost WITH vs WITHOUT saturation ----
fprintf('\n=== true cost J: saturated (umax=2) vs unsaturated (umax=1e6) ===\n');
rng(123); n_trials = 30; n_steps = 500;
for k = {'lqr','pim'}
    Kk = Ks.(k{1});
    for um = [2, 1e6]
        Js = zeros(1,n_trials);
        for t = 1:n_trials
            x = randn(dim_state,1)*3; J = 0;
            for s = 1:n_steps
                u = max(min(-Kk*x,um),-um);
                J = J + x'*Q*x + u'*R*u;
                x = osc_step(osc,x,u);
            end
            Js(t) = J;
        end
        fprintf('%-10s umax=%-6.0f  mean J = %.3f\n', k{1}, um, mean(Js));
    end
end
end
