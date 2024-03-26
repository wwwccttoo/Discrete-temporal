function c = runClosedLoopSimulation(c, flags, varargin, Tref, Iref, PTref, Aref, dA)

% import casadi
import casadi.*

% default inputs
print_on = 0;
rand_ytar = 0;

% parse inputs
if nargin > 2
    % print_on = varargin{1};
    print_on = 1;
end
if nargin > 3
    % rand_ytar = varargin{2};
    rand_ytar = 0;
end

% extract information
Nrep = c.sim.Nrep;
Nsim = c.sim.Nsim;
random_dist_seed = c.sim.random_seed;
ytar_now = c.sim.ytar_now;
nx = c.nx;
nu = c.nu;
nd = c.nd;
nw = c.nw;
ny = c.ny;
nyc = c.nyc;
nv = c.nv;
w_min = c.bounds.w_min;
w_max = c.bounds.w_max;
v_min = c.bounds.v_min;
v_max = c.bounds.v_max;
deltaYprs = c.deltaYprs;

% fix seed if not using random
if random_dist_seed ~= 0
    rng(random_dist_seed, 'twister')
end

% get state samples
X0 = getSamples(c.sim.x0_min, c.sim.x0_max, Nrep);

% get initial observer values
Ehat0 = getSamples(-c.obs.xhat0_err, c.obs.xhat0_err, Nrep);
Xhat0 = X0 + Ehat0;
Dhat0 = zeros(nd,Nrep);

% initialize variables
X = zeros(nx,Nsim+1,Nrep);
Xhat = zeros(nx,Nsim+1,Nrep);
Dhat = zeros(nd,Nsim+1,Nrep);
U = zeros(nu,Nsim,Nrep);
W = zeros(nw,Nsim,Nrep);
Y = zeros(ny,Nsim+1,Nrep);
Yc = zeros(nyc,Nsim+1,Nrep);
Ytar = zeros(nyc,Nsim,Nrep);
V = zeros(nv,Nsim,Nrep);
Xss = zeros(nx,Nsim,Nrep);
Uss = zeros(nu,Nsim,Nrep);
Time = zeros(1,Nsim,Nrep);
Yprs = zeros(1,Nsim,Nrep);

% print start statement
if print_on == 1
    fprintf('MODEL TYPE = %g; OFFSET = %g; MPC TYPE = %g; MPC DNN = %g; TT DNN = %g\n', flags.model, flags.offset, flags.mpcType, flags.mpcDNN, flags.ttDNN)
end

% run repeat simiulations for every initial state
for n = 1:Nrep
    % reset initial condition for mpc and tt
    c = resetMPCInitialGuess(c, flags);
    c = resetTargetInitialGuess(c);

    % initial states
    X(:,1,n) = X0(:,n);

    % correct if outside bounds
    Xhat0(:,n) = min(Xhat0(:,n),c.sim.x0_max);
    Xhat0(:,n) = max(Xhat0(:,n),c.sim.x0_min);

    % initial observer values
    Xhat(:,1,n) = Xhat0(:,n);
    Dhat(:,1,n) = Dhat0(:,n);
    c.obs.xhat = Xhat(:,1,n);
    c.obs.dhat = Dhat(:,1,n);
    c.obs.P = c.obs.P0;

    % get initial measurements
    if random_dist_seed ~= 0
        V(:,1,n) = (v_max-v_min).*rand(nv,1) + v_min;
    end
    Y(:,1,n) = full(c.hp(X(:,1,n),V(:,1,n)));
    Yc(:,1,n) = full(c.r(Y(:,1,n)));

    % draw random target, if specified
    if rand_ytar == 1
        ytar_curr = (c.bounds.yc_max - c.bounds.yc_min).*rand(c.nyc,1) + c.bounds.yc_min;
    end

    % print start statement
    if print_on == 1
        fprintf('Simulation number %g...',n)
        startTicN = tic;
    end

    % run loop over time
    for k = 1:Nsim
        % get current target value
        if rand_ytar == 0
            Ytar(:,k,n) = ytar_now( c.Ts*k );
        else
            Ytar(:,k,n) = ytar_curr;
        end

        % select the proper controller
        if flags.mpcType == 0
            tic
            if flags.ttDNN == 0
                % solve target tracker
                c = setTargetParameters(c, Ytar(:,k,n), Dhat(:,k,n));
                c = solveTargetOptimization(c, 1);
                [Xss(:,k,n), Uss(:,k,n)] = returnSteadyStateValues(c);

            elseif flags.ttDNN == 1
                % evaluate the neural network
                res = full(c.tt.netca([Ytar(:,k,n) ; Dhat(:,k,n)]));
                Xss(:,k,n) = res(1:c.nx);
                Uss(:,k,n) = res(c.nx+1:c.nx+c.nu);

            end

            if flags.mpcDNN == 0
                % solve mpc problem
                c = setMPCParameters(c, flags.mpcType, Xhat(:,k,n), Dhat(:,k,n), Xss(:,k,n), Uss(:,k,n));
                c = solveMPCOptimization(c);
                U(:,k,n) = returnMPCSolution(c);

            elseif flags.mpcDNN == 1
                % evaluate the neural network
                U(:,k,n) = full(c.mpc.netca([Xhat(:,k,n) ; Dhat(:,k,n) ; Xss(:,k,n) ; Uss(:,k,n)]));

                % correct with the target if specified
                if flags.ttCorrection == 1
                    uss_net = full(c.mpc.netca([Xss(:,k,n) ; Dhat(:,k,n) ; Xss(:,k,n) ; Uss(:,k,n)]));
                    U(:,k,n) = Uss(:,k,n) + U(:,k,n) - uss_net;
                end

            end
            Time(:,k,n) = toc;

        elseif flags.mpcType == 1
            tic
            if flags.mpcDNN == 0
                % solve mpc problem
                c = setMPCParameters(c, flags.mpcType, Xhat(:,k,n), Dhat(:,k,n), Ytar(:,k,n));
                c = solveMPCOptimization(c);
                U(:,k,n) = returnMPCSolution(c);

            elseif flags.mpcDNN == 1
                % evaluate the neural network
                U(:,k,n) = full(c.mpc.netca([Xhat(:,k,n) ; Dhat(:,k,n) ; Ytar(:,k,n)]));

                % correct with the target if specified
                if flags.ttCorrection == 1
                    if flags.ttDNN == 0
                        % solve target tracker
                        c = setTargetParameters(c, Ytar(:,k,n), Dhat(:,k,n));
                        c = solveTargetOptimization(c, 1);
                        [Xss(:,k,n), Uss(:,k,n)] = returnSteadyStateValues(c);

                    elseif flags.ttDNN == 1
                        % evaluate the neural network
                        res = full(c.tt.netca([Ytar(:,k,n) ; Dhat(:,k,n)]));
                        Xss(:,k,n) = res(1:c.nx);
                        Uss(:,k,n) = res(c.nx+1:c.nx+c.nu);

                    end
                    uss_net = full(c.mpc.netca([Xss(:,k,n) ; Dhat(:,k,n) ; Ytar(:,k,n)]));
                    U(:,k,n) = Uss(:,k,n) + U(:,k,n) - uss_net;
                end

            end
            Time(:,k,n) = toc;

        else
            error('mpc type not supported!')
        end

        % make sure control inputs is within bounds
        for iu = 1:c.nu
            if U(iu,k,n) > c.bounds.u_max(iu)
                U(iu,k,n) = c.bounds.u_max(iu);
            elseif U(iu,k,n) < c.bounds.u_min(iu)
                U(iu,k,n) = c.bounds.u_min(iu);
            end
        end

        % get disturbance
        if random_dist_seed ~= 0
            W(:,k,n) = (w_max-w_min).*rand(nw,1) + w_min;
        end

        % provide values to plant
        X(:,k+1,n) = full(c.fp(X(:,k,n),U(:,k,n),W(:,k,n)));
        if random_dist_seed ~= 0
            V(:,k+1,n) = (v_max-v_min).*rand(nv,1) + v_min;
        end
        Y(:,k+1,n) = full(c.hp(X(:,k+1,n),V(:,k+1,n)));
        Yc(:,k+1,n) = full(c.r(Y(:,k+1,n)));

        Yprs(:,k+1,n) = Yprs(:,k,n) + full(deltaYprs(Y(:,k+1,n)));

        if k >= PTref
            break;
        end


        % give input and measurement to observer and get next state estimate
        c = updateEKF(c, U(:,k,n), Y(:,k+1,n));
        Xhat(:,k+1,n) = c.obs.xhat;
        Dhat(:,k+1,n) = c.obs.dhat;
    end

    % print end statement
    if print_on == 1
        fprintf('took %g seconds\n',toc(startTicN))
    end

end
% disp(c.mpc.sol)
% disp(c.mpc.sol.stats().return_status)
% store data
c.data.X = X;
c.data.U = U;
c.data.W = W;
c.data.Y = Y;
c.data.V = V;
c.data.Yc = Yc;
c.data.Ytar = Ytar;
c.data.Xhat = Xhat;
c.data.Dhat = Dhat;
c.data.Xss = Xss;
c.data.Uss = Uss;
c.data.Time = Time;
c.data.Yprs = Yprs;

end
