function [X_train, Y_train, X_val, Y_val, X_test, Y_test] = ...
    generate_lorenz96_data(N, dt, F, n_train, n_val, n_test, seed)
% GENERATE_LORENZ96_DATA Generate train/val/test data for Lorenz-96 dynamics learning.
%
%   Task: Learn discrete-time mapping  x(k+1) = Phi(x(k))
%   using the Lorenz-96 system as the ground truth.
%
%   Data is generated from multiple random initial conditions to ensure
%   coverage of the chaotic attractor. Each split uses different seeds.
%
%   Inputs:
%       N       - System dimension (recommended: 40)
%       dt      - Time step for discrete mapping (recommended: 0.01)
%       F       - Forcing parameter (recommended: 8.0)
%       n_train - Number of training samples
%       n_val   - Number of validation samples
%       n_test  - Number of test samples
%       seed    - Random seed for reproducibility
%
%   Outputs:
%       X_train, Y_train - Training input/output pairs: X(k) -> X(k+1)
%       X_val, Y_val     - Validation pairs
%       X_test, Y_test   - Test pairs

    rng(seed);

    % Number of trajectories (split into multiple ICs for diversity)
    n_traj_train = 4;

    % ---- Training data ----
    fprintf('  Generating training data (%d trajectories x %d samples)...\n', ...
            n_traj_train, n_train / n_traj_train);

    all_X = []; all_Y = [];

    for t = 1:n_traj_train
        % Random initial condition near the attractor (F ~ 8)
        x0 = randn(N, 1) * 3.0 + F;
        [~, traj] = lorenz96_rk4(x0, dt, n_train/n_traj_train + 1, F, 2000);

        % Build input-output pairs: X(k) -> X(k+1)
        X_t = traj(1:end-1, :);
        Y_t = traj(2:end, :);

        all_X = [all_X; X_t];  %#ok<AGROW>
        all_Y = [all_Y; Y_t];  %#ok<AGROW>
    end

    X_train = single(all_X);
    Y_train = single(all_Y);

    % ---- Validation data ----
    fprintf('  Generating validation data (%d samples)...\n', n_val);
    x0_val = randn(N, 1) * 3.0 + F;
    [~, traj_val] = lorenz96_rk4(x0_val, dt, n_val + 1, F, 2000 + seed);
    X_val = single(traj_val(1:end-1, :));
    Y_val = single(traj_val(2:end, :));

    % ---- Test data ----
    fprintf('  Generating test data (%d samples)...\n', n_test);
    x0_test = randn(N, 1) * 3.0 + F;
    [~, traj_test] = lorenz96_rk4(x0_test, dt, n_test + 1, F, 2000 + seed*2);
    X_test = single(traj_test(1:end-1, :));
    Y_test = single(traj_test(2:end, :));

    fprintf('  Data generation complete.\n');
    fprintf('  Train: (%d, %d), Val: (%d, %d), Test: (%d, %d)\n', ...
            size(X_train), size(Y_train), size(X_val));
end
