%% STARTUP  Set up the MATLAB environment for the Hybrid-Edited-Learning repo.
%
%   Run this once at the start of each session -- MATLAB also runs it
%   automatically when it is launched with this folder as the current folder.
%
%       >> startup
%
%   What it does:
%     1. Changes to the repository root, so every relative output path used
%        by the scripts (results/, fig/) resolves to the shared folders.
%     2. Adds the repository root, all of code/ (recursively) and the
%        debug-utility folder misc/ to the MATLAB search path.
%
%        NOTE: the legacy/ folder (earlier procedural implementation) is
%        intentionally NOT added, to avoid shadowing the current model code
%        (e.g. legacy/taylor_expand.m vs code/models/taylor_expand.m).
%
%   No simulation is run here. See docs/QUICKSTART.md for the run order.

here = fileparts(mfilename('fullpath'));        % repository root
cd(here);
addpath(here);
addpath(genpath(fullfile(here, 'code')));       % models, utils, experiments, drivers, tools
addpath(fullfile(here, 'misc'));                % development/debug utilities
fprintf('Hybrid-Edited-Learning: working directory = %s\n', here);
fprintf('Added code/ (all subfolders) and misc/ to the MATLAB path.\n');
