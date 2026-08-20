function x_next = osc_step(osc, x, u)
%% OSC_STEP  Single time step of coupled oscillator dynamics (Euler)
%   x_next = osc_step(osc, x, u)
%   osc: oscillator struct from setup_oscillator
%   x: current state [pos_1,...,pos_N, vel_1,...,vel_N]
%   u: control input [u_1,...,u_M]
    N = osc.N; dt = osc.dt;
    pos = x(1:N); vel = x(N+1:end);
    u_full = zeros(N, 1);
    act_idx = find(osc.actuated);
    for j = 1:length(act_idx)
        u_full(act_idx(j)) = u(j);
    end

    acc = zeros(N, 1);
    for i = 1:N
        ip = mod(i, N) + 1; im = mod(i-2, N) + 1;
        F_spring = osc.k(i)*(pos(ip)-pos(i)) + osc.k(im)*(pos(im)-pos(i));
        F_damper = osc.c(i)*(vel(ip)-vel(i)) + osc.c(im)*(vel(im)-vel(i));
        F_friction = -osc.d(i) * vel(i);
        F_control = u_full(i);
        acc(i) = (F_spring + F_damper + F_friction + F_control) / osc.m(i);
    end

    pos_next = pos + dt * vel;
    vel_next = vel + dt * acc;
    x_next = [pos_next; vel_next];
end
