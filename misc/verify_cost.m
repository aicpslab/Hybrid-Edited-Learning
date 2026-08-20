function verify_cost
%% VERIFY_COST  Compare controllers under the TRUE LQR cost J = sum(x'Qx + u'Ru)
%  Uses the already-trained models saved by control_evaluation.

N = 20; dim_state = 2*N; dim_control = 5;
S = load('results/control_results.mat');
models = S.models;
osc = setup_oscillator(N, dim_control);
Q = eye(dim_state)*0.1; R = eye(dim_control)*0.01;
K_lqr = design_lqr(osc.A_mat, osc.B_mat, Q, R);

names = {'unedited','pim','tkm','pim_tkm','random','mlp'};
K_hat = struct();
for i = 1:numel(names)
    m = models.(names{i});
    if isa(m, 'PhNNModel')
        W_eff = m.A_value + m.A_uncertain .* m.W_learn;
        A_h = double(W_eff(:, 1:dim_state));
        B_h = double(W_eff(:, dim_state+1:dim_state+dim_control));
    else
        W1 = m.weights{1}; b1 = m.biases{1};
        W2 = m.weights{2}; b2 = m.biases{2};
        W3 = m.weights{3};
        a1 = b1; g1 = single(a1>0); h1 = max(0,a1);
        a2 = h1*W2 + b2; g2 = single(a2>0);
        J = W3'*diag(g2)*W2'*diag(g1)*W1';
        A_h = double(J(:,1:dim_state)); B_h = double(J(:,dim_state+1:end));
    end
    if max(abs(eig(A_h))) > 1.0+1e-6 || norm(B_h,'fro') < 1e-6
        K_hat.(names{i}) = zeros(dim_control, dim_state);
    else
        K_hat.(names{i}) = design_lqr(A_h, B_h, Q, R);
    end
end

rng(123); n_trials = 30; n_steps = 500;
allkeys = [{'lqr'}, names];
costs = struct(); finals = struct();
for i = 1:numel(allkeys), costs.(allkeys{i}) = []; finals.(allkeys{i}) = []; end
for t = 1:n_trials
    x0 = randn(dim_state,1)*3;
    [Jc, fn] = run_cost(osc, K_lqr, x0, n_steps, Q, R);
    costs.lqr(end+1) = Jc; finals.lqr(end+1) = fn;
    for i = 1:numel(names)
        [Jc, fn] = run_cost(osc, K_hat.(names{i}), x0, n_steps, Q, R);
        costs.(names{i})(end+1) = Jc; finals.(names{i})(end+1) = fn;
    end
end

fprintf('\n%-14s %-18s %-16s %-14s\n', 'Controller', 'mean TRUE cost J', 'mean final ||x||', 'J ratio vs LQR');
fprintf('%s\n', repmat('-',1,66));
jl = mean(costs.lqr);
for i = 1:numel(allkeys)
    k = allkeys{i};
    fprintf('%-14s %-18.3f %-16.4f %-14.3f\n', k, mean(costs.(k)), mean(finals.(k)), mean(costs.(k))/jl);
end
end

function [Jc, fn] = run_cost(osc, K, x0, n_steps, Q, R)
x = x0(:); Jc = 0;
for k = 1:n_steps
    u = max(min(-K*x, 2), -2);
    Jc = Jc + x'*Q*x + u'*R*u;
    x = osc_step(osc, x, u);
end
fn = norm(x);
end
