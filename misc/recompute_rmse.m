% recompute_rmse.m -- Recompute multi-step RMSE with corrected alignment and
% windowed rollout, reusing the already-trained models saved in
% results/lorenz96_results.mat (no re-simulation / re-training required).
cd(fileparts(fileparts(mfilename('fullpath'))));  % cd to repo root so relative results/ resolves
rng(0);   % reproducible start points

S = load('results/lorenz96_results.mat');
results = S.results; models = S.models; test_traj = S.test_traj;

names = {'unedited', 'pim', 'tkm', 'pim_tkm'};
for i = 1:numel(names)
    name = names{i};
    m = models.(name);
    [rms, rstd] = compute_autoregressive_rmse(m, [], test_traj, 200);
    results.(name).rmse_by_step = rms;
    results.(name).rmse_std = rstd;
    fprintf('%-10s step1=%.3e step10=%.3e step50=%.3e step200=%.3e\n', ...
        name, rms(1), rms(min(10,end)), rms(min(50,end)), rms(end));
end

save('results/lorenz96_results.mat', 'results', 'models', 'test_traj');
fprintf('Updated results/lorenz96_results.mat\n');
