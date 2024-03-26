function c = getTargetTracker(c, varargin)

% set default options if only 1 input
if nargin == 1
    solver_opts.print_level = 0;
    solver_opts.max_iter = 1000;
    solver_opts.tol = 1e-8;
end
if nargin > 1
    solver_opts.print_level = varargin{1};
end
if nargin > 2
    solver_opts.max_iter = varargin{2};
end
if nargin > 3
    solver_opts.tol = varargin{3};
end

% import casadi
import casadi.*

% extract bounds
x_min = c.bounds.x_min;
x_max = c.bounds.x_max;
u_min = c.bounds.u_min;
u_max = c.bounds.u_max;
y_min = c.bounds.y_min;
y_max = c.bounds.y_max;
yc_min = c.bounds.yc_min;
yc_max = c.bounds.yc_max;
x_init = c.bounds.x_init;
u_init = c.bounds.u_init;
y_init = c.bounds.y_init;
yc_init = c.bounds.yc_init;

% create solver
opti = casadi.Opti();

% define parameters
D = opti.parameter(c.nd); opti.set_value(D, zeros(c.nd,1));
Ytar= opti.parameter(c.nyc); opti.set_value(Ytar, zeros(c.nyc,1));

% define variables
X = opti.variable(c.nx); opti.set_initial(X, x_init)
U = opti.variable(c.nu); opti.set_initial(U, u_init)
Y = opti.variable(c.ny); opti.set_initial(Y, y_init)
Yc = opti.variable(c.nyc); opti.set_initial(Yc, yc_init)

% add constraints on variables
opti.subject_to( x_min <= X <= x_max )
opti.subject_to( u_min <= U <= u_max )
opti.subject_to( y_min <= Y <= y_max )
opti.subject_to( yc_min <= Yc <= yc_max )

% add steady-state constraints [assume discrete-time]
opti.subject_to( X == c.f(X,U,D) )
opti.subject_to( Y == c.h(X,D) )
opti.subject_to( Yc == c.r(Y) )

% objective function
% J = (Yc(1)-Ytar(1))^2;
J = (Yc(1)-Ytar(1))^2 + (Yc(2)-Ytar(2))^2;
opti.minimize( J )

% specify solver options
p_opts = struct('expand',true,'print_time',0);
s_opts = struct('max_iter',solver_opts.max_iter,'tol',solver_opts.tol);
s_opts.print_level = solver_opts.print_level;
opti.solver('ipopt',p_opts,s_opts);

% store information
c.tt.opti = opti;
c.tt.X = X;
c.tt.U = U;
c.tt.D = D;
c.tt.Y = Y;
c.tt.Yc = Yc;
c.tt.Ytar = Ytar;
c.tt.J = J;
c.tt.opts = solver_opts;

end
