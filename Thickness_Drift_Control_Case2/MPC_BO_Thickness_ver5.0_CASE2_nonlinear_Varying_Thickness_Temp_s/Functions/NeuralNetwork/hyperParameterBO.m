function [net, H, L, finalrmse] = hyperParameterBO(data, target, nodesMax, layersMax, opts)

% Combine data and target into single matrix
ntarget = size(target,1);
data = data';
target = target';
Daten = [data, target];
[m,n] = size(Daten);

% Split into train and test
P = 0.7;
Training = Daten(1:round(P*m),:) ;
Testing = Daten(round(P*m)+1:end,:);
XTrain = Training(:,1:n-ntarget);
YTrain = Training(:,n-ntarget+1:end);
XTest = Testing(:,1:n-ntarget);
YTest = Testing(:,n-ntarget+1:end);

% Define a train/validation split to use inside the objective function
cv = cvpartition(numel(YTrain(:,1)), 'Holdout', 1/3);

% Define hyperparameters to optimize
vars = [optimizableVariable('hiddenLayerSize', [5,nodesMax], 'Type', 'integer');
        optimizableVariable('numberLayer', [2,layersMax], 'Type', 'integer')];

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
