function c = getTargetTrainingData(c, flags)

% extract information
Ns = c.tt.Nsamp;
yc_min = c.bounds.yc_min;
yc_max = c.bounds.yc_max;
nyc = c.nyc;
nd = c.nd;
dhat_min = c.bounds.dhat_min;
dhat_max = c.bounds.dhat_max;

% random target samples
Yss = (yc_max - yc_min).*rand(nyc,Ns) + yc_min;

% random disturbance samples
Dhat = (dhat_max - dhat_min).*rand(nd,Ns) + dhat_min;

% solve target tracker at each sample
Xss = zeros(c.nx,Ns);
Uss = zeros(c.nu,Ns);
for i = 1:Ns
    c = setTargetParameters(c, Yss(:,i), Dhat(:,i));
    c = solveTargetOptimization(c);
    [Xss(:,i), Uss(:,i)] = returnSteadyStateValues(c);
end

% return data
c.tt.rawData.Y = Yss;
c.tt.rawData.X = Xss;
c.tt.rawData.U = Uss;
c.tt.rawData.Dhat = Dhat;
c.tt.data = [Yss ; Dhat];
c.tt.target = [Xss ; Uss];

end
