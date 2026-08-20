function K = design_lqr(A, B, Q, R)
%% DESIGN_LQR  Discrete-time LQR gain via Riccati iteration
%   K = design_lqr(A, B, Q, R)
%
%   Solves the discrete-time algebraic Riccati equation by fixed-point
%   iteration on P, then returns the state-feedback gain K that minimizes
%       J = sum_k ( x' Q x + u' R u )
%   under the discrete dynamics x_{k+1} = A x_k + B u_k.
%
%   Inputs:
%       A - discrete state matrix (n x n)
%       B - discrete input matrix (n x m)
%       Q - state weighting matrix (n x n, PSD)
%       R - input weighting matrix (m x m, PD)
%
%   Output:
%       K - state-feedback gain (m x n) such that u = -K x
%
%   Supports the coupled-oscillator certainty-equivalence LQR control
%   experiments (Sections 4.2 / 5 of the paper).
    P = Q;
    for iter = 1:200
        P = Q + A' * P * A - A' * P * B * ((R + B' * P * B) \ (B' * P * A));
    end
    K = (R + B' * P * B) \ (B' * P * A);
end
