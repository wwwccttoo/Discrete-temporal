function c = getMPCTrainingData(c, flags)

if flags.mpcType == 0
    % random samples for initial states, disturbances, steady-state
    % states and inputs
    Ns = c.mpc.Nsamp;
    X0 = (c.bounds.xhat_max - c.bounds.xhat_min).*rand(c.nx,Ns) + c.bounds.xhat_min;
    Dhat = (c.bounds.dhat_max - c.bounds.dhat_min).*rand(c.nd,Ns) + c.bounds.dhat_min;
    Xss = (c.bounds.x_max - c.bounds.x_min).*rand(c.nx,Ns) + c.bounds.x_min;
    Uss = (c.bounds.u_max - c.bounds.u_min).*rand(c.nu,Ns) + c.bounds.u_min;

    % solvempc at each sample
    Uopt = zeros(c.nu,Ns);
    for i = 1:Ns
        c = setMPCParameters(c, flags.mpcType, X0(:,i), Dhat(:,i), Xss(:,i), Uss(:,i));
        c0 = solveMPCOptimization(c);
        Uopt(:,i) = returnMPCSolution(c0);
    end

    % return data
    if ~isfield(c.mpc,'rawData')
        c.mpc.rawData.X0 = X0;
        c.mpc.rawData.Dhat = Dhat;
        c.mpc.rawData.Xss = Xss;
        c.mpc.rawData.Uss = Uss;
        c.mpc.rawData.Uopt = Uopt;
        c.mpc.data = [X0 ; Dhat ; Xss ; Uss];
        c.mpc.target = [Uopt];
    else
        c.mpc.rawData.X0 = [c.mpc.rawData.X0, X0];
        c.mpc.rawData.Dhat = [c.mpc.rawData.Dhat, Dhat];
        c.mpc.rawData.Xss = [c.mpc.rawData, Xss];
        c.mpc.rawData.Uss = [c.mpc.rawData, Uss];
        c.mpc.rawData.Uopt = [c.mpc.rawData.Uopt, Uopt];
        c.mpc.data = [c.mpc.data, [X0 ; Dhat ; Xss ; Uss]];
        c.mpc.target = [c.mpc.target, Uopt];
    end

elseif flags.mpcType == 1
    % random samples for initial states, disturbances, steady-state
    % states and inputs
    Ns = c.mpc.Nsamp;
    X0 = (c.bounds.xhat_max - c.bounds.xhat_min).*rand(c.nx,Ns) + c.bounds.xhat_min;
    Dhat = (c.bounds.dhat_max - c.bounds.dhat_min).*rand(c.nd,Ns) + c.bounds.dhat_min;
    Ytar = (c.bounds.yc_max - c.bounds.yc_min).*rand(c.nyc,Ns) + c.bounds.yc_min;

    % solve mpc at each sample
    Uopt = zeros(c.nu,Ns);
    for i = 1:Ns
        c = resetMPCInitialGuess(c, flags);
        c = setMPCParameters(c, flags.mpcType, X0(:,i), Dhat(:,i), Ytar(:,i));
        c = solveMPCOptimization(c);
        Uopt(:,i) = returnMPCSolution(c);
    end

    % return data
    if ~isfield(c.mpc,'rawData')
        c.mpc.rawData.X0 = X0;
        c.mpc.rawData.Dhat = Dhat;
        c.mpc.rawData.Ytar = Ytar;
        c.mpc.rawData.Uopt = Uopt;
        c.mpc.data = [X0 ; Dhat ; Ytar];
        c.mpc.target = [Uopt];
    else
        c.mpc.rawData.X0 = [c.mpc.rawData.X0, X0];
        c.mpc.rawData.Dhat = [c.mpc.rawData.Dhat, Dhat];
        c.mpc.rawData.Ytar = [c.mpc.rawData.Ytar, Ytar];
        c.mpc.rawData.Uopt = [c.mpc.rawData.Uopt, Uopt];
        c.mpc.data = [c.mpc.data, [X0 ; Dhat ; Ytar]];
        c.mpc.target = [c.mpc.target, Uopt];
    end

end

end
