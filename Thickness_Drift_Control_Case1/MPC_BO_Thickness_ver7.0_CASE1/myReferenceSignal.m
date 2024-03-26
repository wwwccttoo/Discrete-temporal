function yref = myReferenceSignal(t, Ts, xssc, Tref, Iref)

yref = [Tref; Iref] - xssc(1:2);

% Nstep = 20;
%
% if ceil(t/Ts) < Nstep
%     yref = 1;
% elseif ceil(t/Ts) < 2*Nstep
%     yref = -1;
% elseif ceil(t/Ts) < 3*Nstep
%     yref = 0.5;
% elseif ceil(t/Ts) < 4*Nstep
%     yref = -0.5;
% else
%     yref = 0;
% end

% Nstep = 20;
%
% if ceil(t/Ts) < Nstep
%     yref = -0.3;
% elseif ceil(t/Ts) < 2*Nstep
%     yref = 0;
% elseif ceil(t/Ts) < 3*Nstep
%     yref = 0.3;
% elseif ceil(t/Ts) < 4*Nstep
%     yref = 0.6;
% else
%     yref = 0.9;
% end

end
