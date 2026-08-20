function confirm_bug
%% CONFIRM_BUG  Verify that osc.B_mat is missing the dt factor, and that this
%  is what makes the nominal LQR gain too weak (hence PIM "beats" it).

N = 20; dim_state = 2*N; dim_control = 5;
osc = setup_oscillator(N, dim_control);
dt = osc.dt;

% 1) Direct check: true B (from osc_step: vel += dt*u/m) vs stored B_mat
%    Empirically: perturb one control channel, compare osc_step to A_mat*x + B_mat*u.
rng(7);
x = randn(dim_state,1);
u = zeros(dim_control,1); u(1) = 1.0;
x1_step  = osc_step(osc, x, u);
x1_lin   = osc.A_mat*x + osc.B_mat*u;
fprintf('one-step residual ||osc_step - (A*x + B_mat*u)|| = %.4e\n', norm(x1_step - x1_lin));

B_true = osc.B_mat * dt;   % candidate correct B = dt/m
x1_lin_true = osc.A_mat*x + B_true*u;
fprintf('one-step residual ||osc_step - (A*x + dt*B_mat*u)|| = %.4e  (should be ~0)\n', norm(x1_step - x1_lin_true));

% 2) Compare LQR gains: buggy B vs correct B(=dt*B_mat)
Q = eye(dim_state)*0.1; R = eye(dim_control)*0.01;
K_buggy   = design_lqr(osc.A_mat, osc.B_mat,   Q, R);
K_correct = design_lqr(osc.A_mat, dt*osc.B_mat, Q, R);
fprintf('\n||K (buggy B=1/m)|| = %.4f\n', norm(K_buggy,'fro'));
fprintf('||K (correct B=dt/m)|| = %.4f\n', norm(K_correct,'fro'));

% 3) True closed-loop spectral radius (dynamics use TRUE B = dt*B_mat)
rho_buggy   = max(abs(eig(osc.A_mat - dt*osc.B_mat*K_buggy)));
rho_correct = max(abs(eig(osc.A_mat - dt*osc.B_mat*K_correct)));
fprintf('\nrho(A_true - B_true*K_buggy)   = %.4f\n', rho_buggy);
fprintf('rho(A_true - B_true*K_correct) = %.4f\n', rho_correct);
end
