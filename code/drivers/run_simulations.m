function run_simulations()
%% RUN_SIMULATIONS  Run the two most recent .m simulations:
%     1. lorenz96_experiment.m  (Lorenz-96 dynamics, PIM/TKM editing)
%     2. oscillator_control.m   (coupled oscillator network + LQR control)
%   Results are saved to simulation_run_results.mat; a log is written to
%   simulation_run_log.txt. Each simulation runs independently (try/catch).

cd(fileparts(fileparts(fileparts(mfilename('fullpath')))));  % cd to repo root so results/ resolves

fid = fopen(fullfile('results', 'simulation_run_log.txt'), 'w');
fprintf(1, '=== Starting lorenz96_experiment ===\n');
fprintf(fid, '=== lorenz96_experiment ===\n');
try
    t0 = tic;
    r1 = lorenz96_experiment();
    fprintf(1, 'lorenz96_experiment DONE in %.1f s\n', toc(t0));
    fprintf(fid, 'lorenz96_experiment: OK, %.1f s\n', toc(t0));
catch e
    fprintf(1, 'lorenz96_experiment FAILED: %s\n', e.message);
    fprintf(fid, 'lorenz96_experiment: FAIL - %s\n', e.message);
end

fprintf(1, '\n=== Starting oscillator_control ===\n');
fprintf(fid, '=== oscillator_control ===\n');
try
    t0 = tic;
    r2 = oscillator_control();
    fprintf(1, 'oscillator_control DONE in %.1f s\n', toc(t0));
    fprintf(fid, 'oscillator_control: OK, %.1f s\n', toc(t0));
catch e
    fprintf(1, 'oscillator_control FAILED: %s\n', e.message);
    fprintf(fid, 'oscillator_control: FAIL - %s\n', e.message);
end

fclose(fid);
save(fullfile('results', 'simulation_run_results.mat'), 'r1', 'r2');
fprintf(1, '\n=== ALL DONE ===\n');
end
