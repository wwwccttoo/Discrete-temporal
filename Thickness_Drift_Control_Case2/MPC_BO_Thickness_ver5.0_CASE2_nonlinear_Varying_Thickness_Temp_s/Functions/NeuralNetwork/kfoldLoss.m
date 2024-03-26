function rmse = kfoldLoss(x, y, cv, numHid, numLay)
% Train net.
net = feedforwardnet(numHid*ones(1,numLay), 'trainlm');
for l = 1:numLay
    net.layers{l}.transferFcn = 'poslin';
end
net = train(net, x(:,cv.training), y(:,cv.training));
% Evaluate on validation set and compute rmse
ypred = net(x(:, cv.test));
rmse = sqrt(mean(mean((ypred - y(:,cv.test)).^2)));
end
