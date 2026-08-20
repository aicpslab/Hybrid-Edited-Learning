function [rmse_by_step, rmse_std] = compute_autoregressive_rmse(model, X_test, trajectory, horizon)
%% COMPUTE_AUTOREGRESSIVE_RMSE  Multi-step autoregressive RMSE (corrected)
%   [rmse_by_step, rmse_std] = compute_autoregressive_rmse(model, X_test, trajectory, horizon)
%
%   Performs true windowed multi-step rollout for BOTH standard (K=1) and
%   temporal (K>1) models. X_test is retained for signature compatibility;
%   the rollout is driven entirely by `trajectory`.
%
%   Alignment: with a window of K consecutive states [traj(s),...,traj(s+K-1)]
%   (oldest first, matching build_temporal_data), the model predicts traj(s+K).
%   Horizon h therefore targets traj(s+K+h-1) -- fixing the previous off-by-one
%   that compared the h-th prediction against traj(s+h-1).
    if nargin < 4, horizon = 200; end

    dim_out = model.dim_out;
    K = model.dim_in / dim_out;   % window length: 1 for standard, 2 for temporal

    n_traj = size(trajectory, 1);
    max_start = n_traj - K - horizon + 1;
    if max_start < 1
        rmse_by_step = NaN(1, horizon);
        rmse_std = NaN(1, horizon);
        return;
    end

    n_test_points = min(20, max_start);
    start_indices = randperm(max_start, n_test_points);

    all_errors = zeros(n_test_points, horizon);

    for s = 1:n_test_points
        start_idx = start_indices(s);
        window = trajectory(start_idx : start_idx + K - 1, :);   % K x dim_out

        for h = 1:horizon
            x_input = reshape(window', 1, []);      % 1 x dim_in, oldest first
            x_pred  = model.forward(x_input);       % 1 x dim_out

            true_state = trajectory(start_idx + K + h - 1, :);
            all_errors(s, h) = sqrt(mean((x_pred - true_state).^2));

            window = [window(2:end, :); x_pred];    % slide: drop oldest, append prediction
        end
    end

    rmse_by_step = mean(all_errors, 1);
    rmse_std = std(all_errors, 0, 1);
end
