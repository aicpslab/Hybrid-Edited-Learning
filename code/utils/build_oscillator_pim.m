function [A_val_pim, A_unc_pim] = build_oscillator_pim(N_MASSES, M_ACTUATORS, dim_state, dim_output, n_mono, mono, osc)
%% BUILD_OSCILLATOR_PIM  PIM for coupled oscillator ring network
%   [A_val_pim, A_unc_pim] = build_oscillator_pim(N_MASSES, M_ACTUATORS, dim_state, dim_output, n_mono, mono, osc)
%
%   Same ring-topology principle as Lorenz-96 PIM:
%   Each mass i only interacts with neighbors {i-1, i, i+1} (cyclic).
%   Position outputs depend on {x_i, v_i}; velocity outputs on {x_{i-1}, x_i, x_{i+1}, v_{i-1}, v_i, v_{i+1}}.
    A_val_pim = zeros(dim_output, n_mono, 'single');
    A_unc_pim = zeros(dim_output, n_mono, 'single');

    for out_i = 1:dim_output
        i_mass = mod(out_i - 1, N_MASSES) + 1;
        ip = mod(i_mass, N_MASSES) + 1;
        im = mod(i_mass - 2, N_MASSES) + 1;

        % Determine relevant input variables
        if out_i <= N_MASSES  % Position output
            relevant_vars = [i_mass, N_MASSES + i_mass];  % x_i, v_i
        else  % Velocity output
            relevant_vars = [im, i_mass, ip, ...
                N_MASSES+im, N_MASSES+i_mass, N_MASSES+ip];
        end

        % Add control inputs if this mass is actuated
        if osc.actuated(i_mass)
            act_idx = find(osc.actuated);
            ctrl_idx = dim_state + find(act_idx == i_mass);
            relevant_vars = [relevant_vars, ctrl_idx];
        end

        for h = 1:n_mono
            midx = mono{h};
            if all(ismember(midx, relevant_vars))
                A_unc_pim(out_i, h) = 1;
            end
            % Known linear terms
            if length(midx) == 1 && midx(1) == i_mass && out_i <= N_MASSES
                A_val_pim(out_i, h) = 1.0;  % x_i -> x_i
            elseif length(midx) == 1 && midx(1) == N_MASSES + i_mass && out_i <= N_MASSES
                A_val_pim(out_i, h) = osc.dt;  % v_i -> x_i coeff = dt
            end
        end
    end
end
