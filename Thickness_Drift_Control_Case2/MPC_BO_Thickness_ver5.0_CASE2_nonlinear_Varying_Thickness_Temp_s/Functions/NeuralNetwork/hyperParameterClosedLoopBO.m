function [net, H, L, finalrmse] = hyperParameterClosedLoopBO(data, target, nodesMax, layersMax, opts)

% extract information
Ntr = size(data,2);
nx = size(data,1);

% scale data
data_min = min(data,[],2);
data_max = max(data,[],2);
target_min = min(target,[],2);
target_max = max(target,[],2);
data_s = 2*(data-repmat(data_min,[1,Ntr]))./repmat(data_max-data_min,[1,Ntr])-1;
target_s = 2*(target-repmat(target_min,[1,Ntr]))./repmat(target_max-target_min,[1,Ntr])-1;

% define hyperparameters to optimize
vars = [optimizableVariable('hiddenLayerSize', [5,nodesMax], 'Type', 'integer');
        optimizableVariable('numberLayer', [2,layersMax], 'Type', 'integer')];

% specify the objective to optimize
minfn = @(T)minFcnBO(T, s, X0, Nsim);

% Optimize
minfn = @(T)kfoldLoss(XTrain', YTrain', cv, T.hiddenLayerSize, T.numberLayer);
results = bayesopt(minfn, vars,'IsObjectiveDeterministic', false,...
    'AcquisitionFunctionName', 'expected-improvement-plus', 'MaxObjectiveEvaluations', opts.maxObjectiveEvaluations);

% return optimal hyperparameter
T = bestPoint(results);
H = T.hiddenLayerSize;
L = T.numberLayer;

% Train final model on full training set using the best hyperparameters
net = feedforwardnet(H*ones(1,L), 'trainlm');
for l = 1:L
    net.layers{l}.transferFcn = 'poslin';
end
net = train(net, XTrain', YTrain');

% Evaluate on test set and compute final rmse
ypred = net(XTest');
finalrmse = sqrt(mean(mean((ypred - YTest').^2)));

end
