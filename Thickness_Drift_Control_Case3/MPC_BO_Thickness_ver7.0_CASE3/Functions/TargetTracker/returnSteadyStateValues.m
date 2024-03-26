function [xss, uss] = returnSteadyStateValues(c)

% extract solution
xss = c.tt.sol.value(c.tt.X);
uss = c.tt.sol.value(c.tt.U);

end
