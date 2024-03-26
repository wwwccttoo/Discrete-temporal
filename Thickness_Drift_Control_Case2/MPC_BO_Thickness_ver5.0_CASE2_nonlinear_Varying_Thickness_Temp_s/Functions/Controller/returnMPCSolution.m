function [ucurr] = returnMPCSolution(c)

% extract solution
ucurr = c.mpc.sol.value(c.mpc.U{1});

end
