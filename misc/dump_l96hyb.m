function dump_l96hyb()
%% DUMP_L96HYB  Development utility: inspect the field structure of saved results
%   Recursively prints the field names and values of the key result .mat files
%   (lorenz96_results.mat, hybrid_framework_results.mat, oscillator_results.mat)
%   to the command window. Useful for debugging and report extraction.
%
%   NOTE: Debugging/development utility only -- not part of the main pipeline.
%   Requires the experiment scripts to have been run first (results under
%   <repo-root>/results/).

clear; close all; addpath(fileparts(mfilename('fullpath')));
for f = {'results/lorenz96_results.mat', 'results/hybrid_framework_results.mat', 'results/oscillator_results.mat'}
    if ~exist(f{1}, 'file'), fprintf('MISSING %s\n\n', f{1}); continue; end
    S = load(f{1});
    fprintf('################ %s ################\n', f{1});
    fprintf('top fields: %s\n', strjoin(fieldnames(S), ', '));
    dumpst(S, 1);
    fprintf('\n');
end
end

function dumpst(st, d)
    if ~isstruct(st), return; end
    fn = fieldnames(st);
    for i = 1:numel(fn)
        v = st.(fn{i});
        if isstruct(v)
            fprintf('%s%s: <struct fields: %s>\n', repmat('  ', 1, d), fn{i}, strjoin(fieldnames(v), ','));
            dumpst(v, d + 1);
        elseif isnumeric(v)
            if numel(v) <= 8
                fprintf('%s%s = %s\n', repmat('  ', 1, d), fn{i}, mat2str(v, 5));
            else
                fprintf('%s%s = [%dx%d %s]\n', repmat('  ', 1, d), fn{i}, size(v, 1), size(v, 2), class(v));
            end
        elseif ischar(v)
            fprintf('%s%s = ''%s''\n', repmat('  ', 1, d), fn{i}, v);
        elseif iscell(v)
            fprintf('%s%s = cell(%s)\n', repmat('  ', 1, d), fn{i}, mat2str(size(v)));
        elseif islogical(v)
            fprintf('%s%s = [logical %dx%d]\n', repmat('  ', 1, d), fn{i}, size(v, 1), size(v, 2));
        end
    end
end
