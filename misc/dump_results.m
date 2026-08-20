function dump_results()
%% DUMP_RESULTS  Development utility: run the experiments and write key
%   metrics to <repo-root>/results/exp_results.txt.
%
%   NOTE: Debugging/development utility only -- not part of the main pipeline.
%   Writes a tab-separated summary of the L96 and controlled experiments.

    outfile = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'results', 'exp_results.txt');
    fid = fopen(outfile, 'w');
    if fid < 0
        error('cannot open output file');
    end

    fprintf(fid, '=== LORENZ96 EXPERIMENT ===\n');
    results = lorenz96_experiment();
    keys = {'unedited', 'pim', 'tkm', 'pim_tkm'};
    for k = 1:numel(keys)
        r = results.(keys{k});
        r50 = r.rmse_by_step(min(50, numel(r.rmse_by_step)));
        r200 = r.rmse_by_step(min(200, numel(r.rmse_by_step)));
        fprintf(fid, '%s\tval=%.6e\trmse50=%.4f\trmse200=%.4f\tlearn=%d\tn_total=%d\tsp=%.4f\ttime=%.1f\n', ...
            keys{k}, r.best_val_loss, r50, r200, r.n_learnable, r.n_total, r.sparsity, r.train_time);
    end

    fprintf(fid, '\n=== CONTROLLED EXPERIMENT ===\n');
    [rc, ~] = experiment_controlled();
    for k = 1:numel(keys)
        r = rc.(keys{k});
        fprintf(fid, '%s\tval=%.6e\trmse=%.6e\tlearn=%d\tn_total=%d\tsp=%.4f\ttime=%.1f\n', ...
            keys{k}, r.best_val_loss, r.test_rmse, r.n_learnable, r.n_total, r.sparsity, r.train_time);
    end

    fclose(fid);
    fprintf('RESULTS WRITTEN TO %s\n', outfile);
end
