% SMOKE_OSCILLATOR_HYBRID  Quick end-to-end validation of the oscillator
% hybrid + closed-loop control pipeline with drastically reduced sizes.
% Exercises the low-order (SUB_EXPANSION=1) sub-PhN variant across a
% partition sweep (TARGETS=[2,4]).
opts = struct();
opts.N_SAMPLES = 900;
opts.EPOCHS = 12;
opts.BATCH = 256;
opts.PATIENCE = 4;
opts.MIN_SAMPLES = 20;
opts.N_TRIALS = 3;
opts.N_CAND = 60;
opts.H_SHOOT = 2;
opts.N_STEPS_SHOOT = 8;
opts.N_STEPS_LQR = 20;
opts.TARGETS = [2, 4];
opts.MAX_PARTS = 8;
tic;
res = oscillator_hybrid_control(opts);
fprintf('\n=== SMOKE OK: total %.1f s ===\n', toc);
fprintf('hybrid N values trained: %s\n', mat2str([res.hybrid.N]));
cfg = fieldnames(res.ctrl_shoot);
fprintf('shoot finals: ');
for i = 1:numel(cfg)
    fprintf('%s=%.4f ', cfg{i}, res.ctrl_shoot.(cfg{i}).final);
end
fprintf('\n');
fprintf('lqr   finals: ');
for i = 1:numel(cfg)
    fprintf('%s=%.4f ', cfg{i}, res.ctrl_lqr.(cfg{i}).final);
end
fprintf('\n');
