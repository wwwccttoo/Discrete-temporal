function c = setMPCParameters(c, mpc_type, varargin)

if length(varargin) == 4 && mpc_type == 0
    % extract from inputs
    xcurr = varargin{1};
    dcurr = varargin{2};
    xss = varargin{3};
    uss = varargin{4};

    % update parameters of the optimization
    c.mpc.opti.set_value(c.mpc.X{1},xcurr)
    c.mpc.opti.set_value(c.mpc.D,dcurr)
    c.mpc.opti.set_value(c.mpc.Xss,xss)
    c.mpc.opti.set_value(c.mpc.Uss,uss)

elseif length(varargin) == 3 && mpc_type == 1
    % extract from inputs
    xcurr = varargin{1};
    dcurr = varargin{2};
    ytar = varargin{3};

    % update parameters of the optimization
    c.mpc.opti.set_value(c.mpc.X{1},xcurr)
    c.mpc.opti.set_value(c.mpc.D,dcurr)
    c.mpc.opti.set_value(c.mpc.Ytar,ytar)

else
    error('input arguments mismatched')
end


end
