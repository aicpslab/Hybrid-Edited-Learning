function dump_report_data()
%% DUMP_REPORT_DATA  Print all numbers needed for the Hybrid experiment report.
clear; close all; addpath(fileparts(mfilename('fullpath')));

fprintf('################ oscillator_hybrid_results.mat ################\n');
S = load(fullfile('results', 'oscillator_hybrid_results.mat'));
r = S.results;

fprintf('\n-- meta --\n');
fn = fieldnames(r.meta);
for i = 1:numel(fn)
    v = r.meta.(fn{i});
    if isnumeric(v)
        fprintf('  meta.%s = %s\n', fn{i}, mat2str(v));
    else
        fprintf('  meta.%s = %s\n', fn{i}, char(v));
    end
end

fprintf('\n-- Monolithic baselines --\n');
for k = {'res_ued', 'res_pim'}
    f = r.(k{1});
    fprintf('  %s: val=%.4e n_learnable=%d sparsity=%.4f train_time=%.2fs epochs=%d\n', ...
        k{1}, f.val_loss, f.n_learnable, f.sparsity, f.train_time, f.epochs_used);
end

fprintf('\n-- Hybrid sub-PhNs (degree-1) --\n');
for s = 1:numel(r.hybrid)
    h = r.hybrid(s);
    fprintf('  Hybrid N=%d: eps=%.2f val=%.4e n_learnable=%d n_total=%d sparsity=%.4f train_time=%.2fs epochs=%d\n', ...
        h.N, h.eps, h.val_loss, h.n_learnable, h.n_total, h.sparsity, h.train_time, h.epochs_used);
end

fprintf('\n-- Ordinary Hybrid (unedited) --\n');
oh = r.ord_hybrid;
fn = fieldnames(oh);
ln = '';
for j = 1:numel(fn)
    v = oh.(fn{j});
    if isnumeric(v) && numel(v) == 1
        ln = [ln sprintf('  %s=%.4g', fn{j}, v)]; %#ok<AGROW>
    end
end
fprintf('  %s\n', ln);

fprintf('\n-- accuracy --\n');
a = r.accuracy;
fn = fieldnames(a);
for i = 1:numel(fn)
    v = a.(fn{i});
    if isstruct(v)
        fn2 = fieldnames(v);
        s2 = '';
        for j = 1:numel(fn2)
            w = v.(fn2{j});
            if isnumeric(w) && numel(w) <= 12
                s2 = [s2 sprintf('%s=%s ', fn2{j}, mat2str(w))]; %#ok<AGROW>
            elseif ischar(w)
                s2 = [s2 sprintf('%s=''%s'' ', fn2{j}, w)]; %#ok<AGROW>
            end
        end
        fprintf('  accuracy.%s: %s\n', fn{i}, s2);
    elseif isnumeric(v) && numel(v) <= 12
        fprintf('  accuracy.%s = %s\n', fn{i}, mat2str(v));
    elseif ischar(v)
        fprintf('  accuracy.%s = ''%s''\n', fn{i}, v);
    end
end

fprintf('\n-- ctrl_shoot (60-step shooting, H=?) --\n');
cs = r.ctrl_shoot;
fn = fieldnames(cs);
for i = 1:numel(fn)
    fprintf('  %-10s final=%.4f\n', fn{i}, cs.(fn{i}).final);
end

fprintf('\n-- ctrl_lqr (500-step certainty-equiv LQR) --\n');
cl = r.ctrl_lqr;
fn = fieldnames(cl);
for i = 1:numel(fn)
    fprintf('  %-10s final=%.4f\n', fn{i}, cl.(fn{i}).final);
end

fprintf('\n-- pca_info --\n');
p = r.pca_info;
fn = fieldnames(p);
for i = 1:numel(fn)
    v = p.(fn{i});
    fprintf('  pca_info.%s = %s\n', fn{i}, mat2str(v));
end

fprintf('\n################ compare_single_vs_hybrid16.mat ################\n');
if exist(fullfile('results', 'compare_single_vs_hybrid16.mat'), 'file')
    C16 = load(fullfile('results', 'compare_single_vs_hybrid16.mat'));
    print_struct(C16, 0);
else
    fprintf('  (not found)\n');
end

fprintf('\n################ compare_single_vs_hybrid_48.mat ################\n');
if exist(fullfile('results', 'compare_single_vs_hybrid_48.mat'), 'file')
    C48 = load(fullfile('results', 'compare_single_vs_hybrid_48.mat'));
    print_struct(C48, 0);
else
    fprintf('  (not found)\n');
end

fprintf('\n################ oscillator_shoot_effective.mat ################\n');
E = load(fullfile('results', 'oscillator_shoot_effective.mat'));
ee = E.res;
fprintf('  method: n_steps=%d n_trials=%d H=%d n_cand=%d warm_start=%s lam=%.3f ub=%.1f\n', ...
    ee.method.n_steps, ee.method.n_trials, ee.method.H, ee.method.n_cand, ...
    ee.method.warm_start, ee.method.lam, ee.method.ub);
cfg = ee.configs{1};
fprintf('  initial ||x0|| = %.3f\n', ee.(cfg{1}).x0norm);
for k = 1:numel(cfg)
    R = ee.(cfg{k});
    fprintf('  %-10s final=%.4f reduction=%.4f n_reach1=%d/30 mean_step2one=%.1f norms@60/120/300/500=%.3f/%.3f/%.3f/%.3f\n', ...
        cfg{k}, R.final, R.reduction, R.n_reach1, R.mean_step2one, ...
        R.traj_mean(61), R.traj_mean(121), R.traj_mean(301), R.traj_mean(501));
end

fprintf('\nDump complete.\n');
end

%% =========================================================================
function print_struct(st, depth)
    fn = fieldnames(st);
    for i = 1:numel(fn)
        v = st.(fn{i});
        if isstruct(v)
            fprintf('  %s%s:\n', repmat('  ', 1, depth), fn{i});
            print_struct(v, depth + 1);
        elseif isnumeric(v) || islogical(v)
            if numel(v) <= 12
                fprintf('  %s%s = %s\n', repmat('  ', 1, depth), fn{i}, mat2str(v));
            else
                fprintf('  %s%s = [%dx%d ...]\n', repmat('  ', 1, depth), fn{i}, size(v,1), size(v,2));
            end
        elseif iscell(v)
            fprintf('  %s%s = cell{%s}\n', repmat('  ', 1, depth), fn{i}, mat2str(size(v)));
        elseif ischar(v)
            fprintf('  %s%s = ''%s''\n', repmat('  ', 1, depth), fn{i}, v);
        else
            fprintf('  %s%s = <%s>\n', repmat('  ', 1, depth), fn{i}, class(v));
        end
    end
end
