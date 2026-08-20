function [A_uncertain_pim_tkm, A_value_pim_temporal, pim_tkm_sparsity] = ...
    build_lorenz96_pim_tkm(dim, dt, monomials_temp, temporal_steps, A_uncertain_tkm)
%% BUILD_LORENZ96_PIM_TKM  Combined PIM+TKM mask for the temporal input
%   [A_uncertain_pim_tkm, A_value_pim_temporal, pim_tkm_sparsity] = ...
%       build_lorenz96_pim_tkm(dim, dt, monomials_temp, temporal_steps, A_uncertain_tkm)
%
%   Combines the Physics Information Matrix (PIM) locality prior with the
%   Temporal Knowledge Matrix (TKM) cross-time pruning for the temporal
%   (lagged) input representation of Lorenz-96:
%       - PIM:  within each temporal block, only the cyclic neighbor set
%               {i-2, i-1, i, i+1} of output i is learnable
%       - TKM:  monomials spanning more than one time step are pruned
%       - Known self-coefficient (1 - dt) is fixed for the last block
%
%   Inputs:
%       dim             - state dimension (e.g. 40)
%       dt              - time step for the known self-coefficient
%       monomials_temp  - cell array of monomials over the temporal input
%       temporal_steps  - number of lagged blocks (K)
%       A_uncertain_tkm - (dim, n_mono_temp) TKM mask from BUILD_LORENZ96_TKM
%
%   Outputs:
%       A_uncertain_pim_tkm  - combined binary mask (1 = learn, 0 = prune)
%       A_value_pim_temporal - fixed known values (self-coefficient)
%       pim_tkm_sparsity     - fraction of pruned connections
%
%   See also: BUILD_LORENZ96_TKM, BUILD_LORENZ96_PIM
    n_mono_temp = length(monomials_temp);
    A_uncertain_pim_temporal = zeros(dim, n_mono_temp, 'single');
    A_value_pim_temporal = zeros(dim, n_mono_temp, 'single');

    for h = 1:n_mono_temp
        midx = monomials_temp{h};

        for k = 0:temporal_steps-1
            block_start = k * dim + 1;
            block_end = (k + 1) * dim;
            in_block_mask = (midx >= block_start) & (midx <= block_end);
            if all(in_block_mask)
                indices_in_block = midx - block_start + 1;  % 1-based within block

                for i = 1:dim
                    i_m2 = mod(i-3, dim) + 1; i_m1 = mod(i-2, dim) + 1;
                    i_p1 = mod(i, dim) + 1;
                    relevant = [i_m2, i_m1, i, i_p1];

                    if all(ismember(indices_in_block, relevant))
                        A_uncertain_pim_temporal(i, h) = 1;
                    end
                    if k == temporal_steps - 1 && length(indices_in_block) == 1 && indices_in_block(1) == i
                        A_value_pim_temporal(i, h) = 1.0 - dt;
                    end
                end
            end
        end
    end

    A_uncertain_pim_tkm = A_uncertain_pim_temporal .* A_uncertain_tkm;
    pim_tkm_sparsity = 1.0 - mean(A_uncertain_pim_tkm(:));
end
