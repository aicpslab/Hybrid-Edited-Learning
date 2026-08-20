function [X_temp, Y_temp] = build_temporal_data(traj, dim, K)
%% BUILD_TEMPORAL_DATA  Build temporal input [x(k-K+1), ..., x(k-1), x(k)] -> x(k+1)
%   Stacks the K most recent states OLDEST FIRST, LATEST LAST.
%   [X_temp, Y_temp] = build_temporal_data(traj, dim, K)
%   traj: (n_steps, dim) trajectory
%   dim: state dimension per time step
%   K: number of temporal steps
    n_samples = size(traj, 1) - K;
    X_temp = zeros(n_samples, dim * K, 'single');
    for k = 1:K
        col_start = (k-1) * dim + 1;
        col_end = k * dim;
        X_temp(:, col_start:col_end) = single(traj(k:n_samples+k-1, :));
    end
    Y_temp = single(traj(K+1:end, :));
end
