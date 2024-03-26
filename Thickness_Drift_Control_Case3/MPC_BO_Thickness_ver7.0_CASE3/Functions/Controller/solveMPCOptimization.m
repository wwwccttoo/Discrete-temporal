function [c] = solveMPCOptimization(c)

try
    % call solver
    sol = c.mpc.opti.solve();
    c.mpc.sol = sol;
catch e
    c.mpc.sol = c.mpc.opti.debug;

% warm start
if c.mpc.warm_start == 1
    c.mpc.opti.set_initial(c.mpc.sol.value_variables());
end

end
