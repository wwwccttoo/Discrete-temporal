function c = getMPC(c, mpc_type, varargin)

% set default options if only 2 inputs
if nargin == 2
    solver_opts.print_level = 0;
    solver_opts.max_iter = 1000;
    solver_opts.tol = 1e-6;
end
if nargin > 2
    solver_opts.print_level = varargin{1};
end
if nargin > 3
    solver_opts.max_iter = varargin{2};
end
if nargin > 4
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

% extract mpc parameters
N = c.mpc.N;
soft = c.mpc.soft;
soft_penalty = c.mpc.soft_penalty;
warm_start = c.mpc.warm_start;
lstage = c.mpc.lstage;
lterm = c.mpc.lterm;
term_equality_cons = c.mpc.term_equality_cons;
backoff = c.mpc.backoff;

% Create solver
opti = casadi.Opti();

% Initialize cell for all states and inputs over horizon
X = cell(N+1,1);
U = cell(N,1);
E = cell(N+1,1);

% Define parameters
D = opti.parameter(c.nd); opti.set_value(D, zeros(c.nd,1));
X{1} = opti.parameter(c.nx); opti.set_value(X{1}, zeros(c.nx,1));
if mpc_type == 0
    Xss = opti.parameter(c.nx); opti.set_value(Xss, zeros(c.nx,1));
    Uss = opti.parameter(c.nu); opti.set_value(Uss, zeros(c.nu,1));
elseif mpc_type == 1
    Ytar = opti.parameter(c.nyc); opti.set_value(Ytar, zeros(c.nyc,1));
    Xss = opti.variable(c.nx); opti.set_initial(Xss, x_init)
    Uss = opti.variable(c.nu); opti.set_initial(Uss, u_init)
    Yss = opti.variable(c.ny); opti.set_initial(Yss, y_init)
    Ycss = opti.variable(c.nyc); opti.set_initial(Ycss, yc_init)
    opti.subject_to( x_min <= Xss <= x_max )
    opti.subject_to( u_min <= Uss <= u_max )
    opti.subject_to( y_min <= Yss <= y_max )
    opti.subject_to( yc_min <= Ycss <= yc_max )
    opti.subject_to( Xss == c.f(Xss,Uss,D) )
    opti.subject_to( Yss == c.h(Xss,D) )
    opti.subject_to( Ycss == c.r(Yss) )
else
    error('mpc type is not supported at this point')
end

% Define mpc problem
J = 0;
for k = 1:N
    U{k} = opti.variable(c.nu);
    opti.subject_to( u_min <= U{k} <= u_max )
    opti.set_initial(U{k}, u_init)

    X{k+1} = opti.variable(c.nx);
    if soft == 1
        E{k+1} = opti.variable(c.ny);
        opti.subject_to( E{k+1} >= zeros(c.ny,1) )
        opti.set_initial(E{k+1}, zeros(c.ny,1))
        opti.subject_to( c.h(X{k+1},D) <= y_max - backoff + E{k+1} )
        opti.subject_to( y_min + backoff - E{k+1} <= c.h(X{k+1},D) )
    else
        opti.subject_to( y_min + backoff <= c.h(X{k+1},D) <= y_max - backoff )
    end
    opti.set_initial(X{k+1}, x_init)

    opti.subject_to( X{k+1} == c.f(X{k},U{k},D) )

    J = J + lstage(X{k},U{k},Xss,Uss);
    if soft == 1
        J = J + soft_penalty * (E{k+1}'*E{k+1});
    end
end
J = J + lterm(X{k},Xss);
if mpc_type == 1
    target_penalty = 1e3;
    J = J + target_penalty * (Ytar-Ycss)'*(Ytar-Ycss);
elseif mpc_type ~= 0
    error('mpc type is not supported at this point')
end
opti.minimize( J )
if term_equality_cons == 1
    opti.subject_to( X{N+1} == Xss )
end

% specify solver options
p_opts = struct('expand',true,'print_time',0,'error_on_fail', false);
s_opts = struct('max_iter',solver_opts.max_iter,'tol',solver_opts.tol);
s_opts.print_level = solver_opts.print_level;
opti.solver('ipopt',p_opts,s_opts);

% store information
c.mpc.opti = opti;
if mpc_type == 1
    c.mpc.Ytar = Ytar;
    c.mpc.Yss = Yss;
    c.mpc.Ycss = Ycss;
elseif mpc_type ~= 0
    error('mpc type is not supported at this point')
end
c.mpc.Xss = Xss;
c.mpc.Uss = Uss;
c.mpc.X = X;
c.mpc.U = U;
c.mpc.D = D;
c.mpc.E = E;
c.mpc.J = J;
c.mpc.opts = solver_opts;

end
