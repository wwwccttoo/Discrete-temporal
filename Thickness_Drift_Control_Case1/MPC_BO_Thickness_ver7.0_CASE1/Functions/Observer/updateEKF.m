function c = updateEKF(c, ucurr, ymeas)

% extract information
xhat = c.obs.xhat;
dhat = c.obs.dhat;

% get predicted state
x_next = full(c.f(xhat, ucurr, dhat));
d_next = dhat;

% get predicted measurement
y_next = full(c.h(x_next, d_next));

% get predicted covariance
A1 = full(c.obs.A1(xhat, ucurr, dhat));
H1 = full(c.obs.H1(xhat, dhat));
phi = A1;
P_aug_next = phi * c.obs.P * phi' + c.obs.Q;

% get kalman gain
K_aug_next = P_aug_next*H1'/(H1*P_aug_next*H1' + c.obs.R);

% update state estimate and covariance
x_aug_next = [x_next ; d_next];
x_aug_update = x_aug_next + K_aug_next*(ymeas - y_next);
P_aug_update = (eye(c.nx+c.nd) - K_aug_next*H1)*P_aug_next;

% update observer attributes
c.obs.xhat = x_aug_update(1:c.nx);
c.obs.dhat = x_aug_update(c.nx+1:c.nx+c.nd);
c.obs.P = P_aug_update;

end
