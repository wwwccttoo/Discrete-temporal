function c = getEKF(c)

% import casadi
import casadi.*

% get linearized system matrices
xdot = c.f(c.x, c.u, c.d);
ymeas = c.h(c.x,c.d);
ddot = c.d;
A1 = Function('A1', {c.x, c.u, c.d}, {jacobian(vertcat(xdot,ddot),vertcat(c.x,c.d))});
H1 = Function('H1', {c.x, c.d}, {jacobian(ymeas,vertcat(c.x,c.d))});

% store information
c.obs.A1 = A1;
c.obs.H1 = H1;

end
