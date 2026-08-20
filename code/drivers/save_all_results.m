function save_all_results()
%% SAVE_ALL_RESULTS  Run all four experiments once and save .mat data.
%   Each experiment script already saves its own results/*.mat (see the
%   save block at the end of each file). This driver runs them sequentially
%   with error isolation so one failure does not stop the others.
%
%   After this completes, run plot_all_figures.m to regenerate the 16
%   report figures from the saved data (no re-simulation needed).

    base = fileparts(fileparts(fileparts(mfilename('fullpath'))));  % repo root (two folders up from code/drivers)
    addpath(base);
    cd(base);

    experiments = {'lorenz96_experiment', 'experiment_sindy', ...
                   'oscillator_control', 'control_evaluation'};
    for i = 1:numel(experiments)
        fn = experiments{i};
        fprintf('\n########## RUNNING %s ##########\n', fn);
        try
            feval(fn);
        catch err
            fprintf('!!! %s FAILED: %s\n', fn, err.message);
            if ~isempty(err.stack)
                fprintf('    at %s (line %d)\n', err.stack(1).name, err.stack(1).line);
            end
        end
    end
    fprintf('\nAll experiments complete. Results in %s\n', fullfile(base, 'results'));
end
