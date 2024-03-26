function c = performanceVerificationDNN(c, flags)

% set performance verification simulation parameters
c0 = c;
c0.sim.Nrep = c.mpc.perfVer.Nrep;
c0.sim.Nsim = c.mpc.perfVer.Nsim;
c0.sim.random_seed = c.mpc.perfVer.random_seed;

% run closed-loop simulations
c0 = runClosedLoopSimulation(c0, flags, 1, 1);

% extract information
c.mpc.perfVer.data = c0.data;

end
