function [train_traj, val_traj, test_traj] = generate_train_val_test_data(N, dt, F, n_train, n_val, n_test, seed)
%% GENERATE_TRAIN_VAL_TEST_DATA  Generate Lorenz-96 train/val/test datasets
%   [train_traj, val_traj, test_traj] = generate_train_val_test_data(N, dt, F, n_train, n_val, n_test, seed)
    rng(seed);

    % Training: 4 trajectories from different initial conditions
    n_train_traj = 4;
    train_data = cell(n_train_traj, 1);
    for i = 1:n_train_traj
        x0 = randn(N, 1) * 3.0 + F;
        train_data{i} = generate_l96_trajectory(x0, dt, n_train / n_train_traj, F, 2000);
    end
    train_traj = cat(1, train_data{:});

    % Validation
    x0_val = randn(N, 1) * 3.0 + F;
    val_traj = generate_l96_trajectory(x0_val, dt, n_val, F, 2000);

    % Test
    x0_test = randn(N, 1) * 3.0 + F;
    test_traj = generate_l96_trajectory(x0_test, dt, n_test, F, 2000);
end

function trajectory = generate_l96_trajectory(x0, dt, n_steps, F, spinup)
    if nargin < 5, spinup = 5000; end
    N = length(x0);
    x = x0(:);
    for s = 1:spinup
        x = rk4_step_l96(x, dt, F);
    end
    trajectory = zeros(n_steps, N);
    for t = 1:n_steps
        x = rk4_step_l96(x, dt, F);
        trajectory(t, :) = x';
    end
end

function x_next = rk4_step_l96(x, dt, F)
    k1 = l96_derivative(x, F);
    k2 = l96_derivative(x + 0.5 * dt * k1, F);
    k3 = l96_derivative(x + 0.5 * dt * k2, F);
    k4 = l96_derivative(x + dt * k3, F);
    x_next = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4);
end

function dx = l96_derivative(x, F)
    N = length(x);
    dx = zeros(N, 1);
    for i = 1:N
        ip1 = mod(i, N) + 1;
        im2 = mod(i-3, N) + 1;
        im1 = mod(i-2, N) + 1;
        dx(i) = (x(ip1) - x(im2)) * x(im1) - x(i) + F;
    end
end
