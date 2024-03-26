function [c] = solveTargetOptimization(c, varargin)

warmstart = 0;
if nargin > 1
    warmstart = varargin{1};
end

% call solver
sol = c.tt.opti.solve();
c.tt.sol = sol;

% warm start
if warmstart == 1
    c.tt.opti.set_initial(sol.value_variables());
end

end
