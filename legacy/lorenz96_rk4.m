function [t, X] = lorenz96_rk4(x0, dt, n_steps, F, spinup)
% LORENZ96_RK4 Simulate Lorenz-96 system using RK4 integration.
%
%   Lorenz-96 dynamics (N variables, cyclic boundary):
%       dx_i/dt = (x_{i+1} - x_{i-2}) * x_{i-1} - x_i + F
%
%   This is a standard benchmark for high-dimensional chaotic systems.
%   For F=8.0, the system exhibits spatiotemporal chaos with Lyapunov
%   exponent ~1.67, making it a challenging test for dynamics learning.
%
%   The coupling structure is SPARSE: each x_i depends only on
%       {x_{i-2}, x_{i-1}, x_i, x_{i+1}}  (4 out of N neighbors)
%
%   This sparse ring topology is exactly what the PIM editing exploits.
%
%   Inputs:
%       x0     - (N, 1) initial state vector
%       dt     - Integration time step (recommended: 0.01)
%       n_steps - Number of steps to output (after spinup)
%       F      - Forcing parameter (standard: 8.0 for chaos)
%       spinup - Number of initial steps to discard (default: 5000)
%
%   Outputs:
%       t - (n_steps, 1) time vector
%       X - (n_steps, N) state trajectory
%
%   Reference:
%       Lorenz, E. N. (1996). Predictability: A problem partly solved.
%       Proc. Seminar on Predictability, Vol. 1.
%
%   See also: GENERATE_LORENZ96_DATA, BUILD_PIM_LORENZ96

    if nargin < 5, spinup = 5000; end

    N = length(x0);
    x = x0(:);  % Ensure column vector

    % Spinup phase (discard transient)
    for step = 1:spinup
        x = rk4_step_l96(x, dt, N, F);
    end

    % Collection phase
    X = zeros(n_steps, N);
    t = (0:n_steps-1)' * dt;

    for step = 1:n_steps
        x = rk4_step_l96(x, dt, N, F);
        X(step, :) = x';
    end
end


function x_next = rk4_step_l96(x, dt, N, F)
% Single RK4 integration step for Lorenz-96.

    k1 = lorenz96_rhs(x, N, F);
    k2 = lorenz96_rhs(x + 0.5*dt*k1, N, F);
    k3 = lorenz96_rhs(x + 0.5*dt*k2, N, F);
    k4 = lorenz96_rhs(x + dt*k3, N, F);

    x_next = x + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4);
end


function dx = lorenz96_rhs(x, N, F)
% Right-hand side of Lorenz-96 ODE.
% dx_i/dt = (x_{i+1} - x_{i-2}) * x_{i-1} - x_i + F
% Cyclic boundary conditions: x_{N+1} = x_1, x_0 = x_N, x_{-1} = x_{N-1}

    dx = zeros(N, 1);

    for i = 1:N
        % Indices with cyclic boundary (1-based MATLAB indexing)
        ip1 = mod(i, N) + 1;           % i+1
        im1 = mod(i-2, N) + 1;         % i-1
        im2 = mod(i-3, N) + 1;         % i-2

        dx(i) = (x(ip1) - x(im2)) * x(im1) - x(i) + F;
    end
end
