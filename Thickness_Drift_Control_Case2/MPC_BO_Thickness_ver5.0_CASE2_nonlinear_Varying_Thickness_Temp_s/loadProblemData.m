function c = loadProblemData(c, flags, Tref, Iref, PTref, Aref, dA)


%% Plant and controller model info

% import casadi
import casadi.*

% sampling time
Ts = 1;

% load system matrices from Data model ID
modelp = load('APPJmodel_TEOS_UCB_LAM_modord3.mat'); %
model = load('APPJmodel_TEOS_UCB_LAM_modord3.mat'); %

% declare variables
nx = 3;
nu = 2; % inputs
nd = 3; % offset-free disturbances
nw = 3; % process noise
ny = 3; % measured outputs
nv = ny; % measurement noise
x = SX.sym('x', nx);
u = SX.sym('u', nu);
d = SX.sym('d', nd);
w = SX.sym('w', nw);
v = SX.sym('v', nv);

%% discrete-time approximation for plant
% system matrix
Ap = modelp.A * 0.9;
Bp = modelp.B * 0.9;
Cp = modelp.C;

xsspc = modelp.yss; % [Ts; I(706); I(777)]
usspc = modelp.uss; % [P; q]

dB = 1.00;
Ap = [Ap(1,1)  Ap(1,2)  Ap(1,3);...
      Ap(2,1)  Ap(2,2)  Ap(2,3);...
      Ap(3,1)  Ap(3,2)  Ap(3,3)];
Bp = [Bp(1,1)*dB  Bp(1,2);...
      Bp(2,1)*dB  Bp(2,2);...
      Bp(3,1)*dB  Bp(3,2)];
Cp = [Cp(1,1)  Cp(1,2)  Cp(1,3);...
      Cp(2,1)  Cp(2,2)  Cp(2,3);...
      Cp(3,1)  Cp(3,2)  Cp(3,3)];

fp1 = Ap(1,1)*x(1) + Ap(1,2)*x(2) + Ap(1,3)*x(3) + Bp(1,1)*u(1) + Bp(1,2)*u(2) + w(1);
fp2 = Ap(2,1)*x(1) + Ap(2,2)*x(2) + Ap(2,3)*x(3) + Bp(2,1)*u(1) + Bp(2,2)*u(2) + w(2);
fp3 = Ap(3,1)*x(1) + Ap(3,2)*x(2) + Ap(3,3)*x(3) + Bp(3,1)*u(1) + Bp(3,2)*u(2) + w(3);
fp = Function('fp', {x,u,w}, {[fp1 ; fp2; fp3]});

% measurement equation for plant
y1 = Cp(1,1)*x(1) + Cp(1,2)*x(2) + Cp(1,3)*x(3) + v(1);
y2 = Cp(2,1)*x(1) + Cp(2,2)*x(2) + Cp(2,3)*x(3) + v(2);
y3 = Cp(3,1)*x(1) + Cp(3,2)*x(2) + Cp(3,3)*x(3) + v(3);
hp = Function('hp', {x,v}, {[y1; y2; y3]});

%% discrete-time approximation for control model
% system matrix
A = model.A;
B = model.B;
C = model.C;

xssc = model.yss'; % [Ts; I(706); I(777)]
ussc = model.uss'; % [P; q]

A = [Aref(1,1) A(1,2) A(1,3);...
     Aref(2,1) A(2,2) A(2,3);...
     Aref(3,1) A(3,2) A(3,3)];

f1 = A(1,1)*x(1) + A(1,2)*x(2) + A(1,3)*x(3) + B(1,1)*u(1) + B(1,2)*u(2);
f2 = A(2,1)*x(1) + A(2,2)*x(2) + A(2,3)*x(3) + B(2,1)*u(1) + B(2,2)*u(2);
f3 = A(3,1)*x(1) + A(3,2)*x(2) + A(3,3)*x(3) + B(3,1)*u(1) + B(3,2)*u(2);

if flags.offset == 1
    f1 = f1 + d(1);
    f2 = f2 + d(2);
    f3 = f3 + d(3);
end
f = Function('f', {x,u,d}, {[f1 ; f2; f3]});

% measurement equation for control model
y1 = C(1,1)*x(1) + C(1,2)*x(2) + C(1,3)*x(3);
y2 = C(2,1)*x(1) + C(2,2)*x(2) + C(2,3)*x(3);
y3 = C(3,1)*x(1) + C(3,2)*x(2) + C(3,3)*x(3);
h = Function('h', {x,d}, {[y1; y2; y3]});


% controlled output equation
ymeas = SX.sym('ymeas', ny);
yc = ymeas(1:2);
r = Function('r', {ymeas}, {yc});

%% Thickness model
w_thick = 1.0;
alpha = 1.0;
dYprs = dA*w_thick*alpha*((x(2)+xsspc(2))/(x(3)+xsspc(3)))*Ts/60;
% dh = alpha*(x[1]+xssp[1]+w[1])*ts/60
deltaYprs = Function('deltaH', {x}, {dYprs});

%% Bound
% specify bounds
x_min = [-20.0 ; -2000.0; -3000.0] - xssc*0 - 1000;
x_max = [ 20.0 ;  2000.0;  3000.0] - xssc*0 + 1000;
u_min = [-2.0; -2.0] - ussc*0 - 20;
u_max = [ 2.0;  2.0] - ussc*0 + 20;
y_min = x_min;
y_max = x_max;
v_min = 0*-0.01*ones(nv,1);
v_max = 0*0.01*ones(nv,1);
w_min = 0*0.25*ones(nw,1);
w_max = 0*0.3*ones(nw,1);
yc_min = y_min(1:2);
yc_max = y_max(1:2);

bounds.x_min = x_min;
bounds.x_max = x_max;
bounds.u_min = u_min;
bounds.u_max = u_max;
bounds.w_min = w_min;
bounds.w_max = w_max;
bounds.y_min = y_min;
bounds.y_max = y_max;
bounds.yc_min = yc_min;
bounds.yc_max = yc_max;
bounds.v_min = v_min;
bounds.v_max = v_max;

% initial variable guesses
x0 = zeros(3,1) - xssc*0;
% x_init = (bounds.x_min + bounds.x_max)/2;
% u_init = (bounds.u_min + bounds.u_max)/2;
% y_init = (bounds.y_min + bounds.y_max)/2;
% yc_init = (bounds.yc_min + bounds.yc_max)/2;
x_init = x0;
u_init = zeros(nu,1);
y_init = x0;
yc_init = y_init(1:2);

bounds.x_init = x_init;
bounds.u_init = u_init;
bounds.y_init = y_init;
bounds.yc_init = yc_init;

% Store information
c.fp = fp;
c.hp = hp;
c.f = f;
c.h = h;
c.r = r;
c.nx = nx;
c.nu = nu;
c.nd = nd;
c.nw = nw;
c.ny = ny;
c.nyc = length(yc);
c.nv = nv;
c.x = x;
c.u = u;
c.d = d;
c.w = w;
c.y = SX.sym('y', ny);
c.v = v;
c.Ts = Ts;
c.bounds = bounds;
c.xssp = xsspc;
c.ussp = usspc;
c.xss = xssc;
c.uss = ussc;
c.deltaYprs = deltaYprs;
%% Controller info

% prediction horizon
N = 10;

% soften constraints? (1=yes, 0=no)
soft = 0;

% penalty for soft constraints
% soft_penalty = 10000;
soft_penalty = 1e+10;

% warm start? (1=yes, 0=no)
warm_start = 1;

% stage cost
x = SX.sym('x', c.nx);
u = SX.sym('u', c.nu);
xss = SX.sym('xss', c.nx);
uss = SX.sym('uss', c.nu);
% Q = eye(c.nx);
% R = 2*eye(c.nu);
Q = [100 0; 0 100];
R = [10 0; 0 0.01];
lstage = Function('lstage', {x,u,xss,uss}, {(x(1:2)-xss(1:2))'*Q*(x(1:2)-xss(1:2)) + (u-uss)'*R*(u-uss)});

% terminal cost
% P = zeros(c.nx);
P = 0*1e+3*eye(c.nyc);
lterm = Function('lterm', {x,xss}, {(x(1:2)-xss(1:2))'*P*(x(1:2)-xss(1:2))});

% terminal equality constraint? (1=yes, 0=no)
% term_equality_cons = 0;
term_equality_cons = 1;

% backoff values for controlled output
% backoff = 0*ones(c.nyc,1);
backoff = 0.1*ones(c.ny,1);

% store information
c.mpc.N = N;
c.mpc.soft = soft;
c.mpc.soft_penalty = soft_penalty;
c.mpc.warm_start = warm_start;
c.mpc.lstage = lstage;
c.mpc.lterm = lterm;
c.mpc.term_equality_cons = term_equality_cons;
c.mpc.backoff = backoff;


%% Simulator info

% number of repeat simulations
Nrep = 1;

% number of simulation steps
Nsim = 100;

% random seed for disturbance (set to 0 for disturbances=0)
random_dist_seed = 100;

% specify function for target changes
% Nchange = 50;
% Ntotal = Nsim;
% ytar_now = getRandomSignal(c.bounds.yc_min, c.bounds.yc_max, c.Ts, Nchange, Ntotal);
ytar_now = @(t) myReferenceSignal(t,Ts, xssc, Tref, Iref);

% specify bounds on initial states for simulation
x0_min = x0;
x0_max = x0;

% store information
c.sim.Nrep = Nrep;
c.sim.Nsim = Nsim;
c.sim.random_seed = random_dist_seed;
c.sim.ytar_now = ytar_now;
c.sim.x0_min = x0_min;
c.sim.x0_max = x0_max;

c.sim.w_thick = w_thick;
c.sim.PTref = PTref;

%% Observer info

% initial covariance, process noise covariance, and measurement noise covariance
% c.obs.P0 = 1*eye(nx+nd);
% c.obs.Q = 1e-2*eye(nx+nd);
% c.obs.R = 1e-4*eye(ny);
% c.obs.P0 = 1*eye(nx+nd);
% c.obs.Q = 1e+1*eye(nx+nd);
% c.obs.R = 1e-1*eye(ny);
% c.obs.P0 = 1*eye(nx+nd);
% c.obs.Q = 50*eye(nx+nd);
% c.obs.R = 0.05*eye(ny);
c.obs.P0 = 1*eye(nx+nd);
c.obs.Q = 1000*eye(nx+nd);
c.obs.R = 0.005*eye(ny);


% error in initial state and disturbance estimates (xhat \in x + err)
c.obs.xhat0_err = 0.1*ones(nx,1);

% guess for bounds on state and disturbance estimate
% c.bounds.xhat_min = [-2 ; -1];
% c.bounds.xhat_max = [ 1 ;  1];
% c.bounds.dhat_min = -0.2*ones(nd,1);
% c.bounds.dhat_max =  0.2*ones(nd,1);
c.bounds.xhat_min = x_min;
c.bounds.xhat_max = x_max;
c.bounds.dhat_min = -0.5*ones(nd,1)-1000;
c.bounds.dhat_max =  0.5*ones(nd,1)+1000;


%% Neural network info

% set properties for target tracker dnn
if flags.ttDNN > 0
    % number of nodes and layers (max if hyperparameters are optimized)
    c.tt.nodes = 8; %8
    c.tt.layers = 4; %4

    % number of samples used to fit neural network
    c.tt.Nsamp = 200; %200
end

% set properties for mpc dnn
if flags.mpcDNN > 0
    % number of nodes and layers (max if hyperparameters are optimized)
    c.mpc.nodes = 10; %10
    c.mpc.layers = 5; %5

    % number of samples used to fit neural network
    c.mpc.Nsamp = 200; %200

    % performance verification parameters
    if flags.performanceVerification > 0
        c.mpc.perfVer.Nrep = 20;
        c.mpc.perfVer.Nsim = 100;
        c.mpc.perfVer.random_seed = 10;
    end
end

% set hyperparmeter optimization parameters
if flags.hyperParamOpt > 0
    c.bo.opts.maxObjectiveEvaluations = 30;
end

end
