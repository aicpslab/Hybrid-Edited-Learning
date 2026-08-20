function osc = setup_oscillator(N, M)
%% SETUP_OSCILLATOR  Create coupled oscillator network with ring topology
%   osc = setup_oscillator(N, M)
%   N: number of masses (state dim = 2N)
%   M: number of actuators (control dim = M)
    if nargin < 1, N = 20; end
    if nargin < 2, M = 5; end

    osc.N = N; osc.M = M; osc.dt = 0.05;
    rng(42);
    osc.m = 0.5 + rand(N,1) * 0.5;    % masses [0.5, 1.0]
    osc.k = 2.0 + rand(N,1) * 1.0;    % spring constants [2, 3]
    osc.c = 0.3 + rand(N,1) * 0.2;    % damping [0.3, 0.5]
    osc.d = 0.1 + rand(N,1) * 0.1;    % friction [0.1, 0.2]

    % Actuator placement: evenly spaced
    osc.actuated = false(N, 1);
    osc.actuated(round(linspace(1, N, M))) = true;
    osc.B_mat = zeros(2*N, M);
    act_idx = find(osc.actuated);
    for j = 1:M
        osc.B_mat(N + act_idx(j), j) = osc.dt / osc.m(act_idx(j));
    end

    % Build linear dynamics matrix
    osc.A_mat = build_A_mat(osc);
end

function A = build_A_mat(osc)
    N = osc.N; dt = osc.dt;
    A = zeros(2*N, 2*N);
    % Position update: x(k+1) = x(k) + dt*v(k)
    A(1:N, 1:N) = eye(N);
    A(1:N, N+1:end) = dt * eye(N);
    % Velocity update
    for i = 1:N
        ip = mod(i, N) + 1;       % i+1 (cyclic)
        im = mod(i-2, N) + 1;     % i-1 (cyclic)
        A(N+i, i)    = A(N+i, i)    - dt * (osc.k(i) + osc.k(im)) / osc.m(i);
        A(N+i, im)   = A(N+i, im)   + dt * osc.k(im) / osc.m(i);
        A(N+i, ip)   = A(N+i, ip)   + dt * osc.k(i) / osc.m(i);
        A(N+i, N+i)  = A(N+i, N+i)  + 1.0 - dt * (osc.c(i) + osc.c(im) + osc.d(i)) / osc.m(i);
        A(N+i, N+im) = A(N+i, N+im) + dt * osc.c(im) / osc.m(i);
        A(N+i, N+ip) = A(N+i, N+ip) + dt * osc.c(i) / osc.m(i);
    end
end
