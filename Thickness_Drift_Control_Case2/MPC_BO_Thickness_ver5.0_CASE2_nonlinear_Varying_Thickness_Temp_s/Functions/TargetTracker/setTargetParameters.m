function c = setTargetParameters(c, ytar, dcurr)

% update parameters of the optimization
c.tt.opti.set_value(c.tt.Ytar,ytar)
c.tt.opti.set_value(c.tt.D,dcurr)

end
