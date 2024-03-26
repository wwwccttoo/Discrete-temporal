function [net, netca] = trainNeuralNetwork(data, target, H, L, varargin)

bo_opts = [];
if nargin > 4
    bo_opts = varargin{1};
end

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

% train the network
if isempty(bo_opts)
    net = feedforwardnet(H*ones(1,L), 'trainlm');
    for l = 1:L
        net.layers{l}.transferFcn = 'poslin';
    end
    [net,tr] = train(net, data_s, target_s);
else
    [net, Hopt, Lopt, finalrmse] = hyperParameterBO(data_s, target_s, H, L, bo_opts);
    H = Hopt;
    L = Lopt;
end

% extract weights and biases from neural network
W = cell(L+1,1);
b = cell(L+1,1);
W{1} = net.IW{1};
b{1} = net.b{1};
for i = 1:L
    W{i+1} = net.LW{i+1,i};
    b{i+1} = net.b{i+1};
end

% create casadi evaluation of neural network
import casadi.*
x = MX.sym('x', nx);
xs = 2*(x-data_min)./(data_max-data_min)-1;
z = max(W{1}*xs+b{1},0);
for k = 1:L-1
    z = max(W{k+1}*z+b{k+1},0);
end
us = W{L+1}*z+b{L+1};
u = (us+1)/2.*(target_max-target_min)+target_min;
%u = min(max(u,s.u_min'),s.u_max');
netca = Function('netca', {x}, {u});

% % test if results are same
% input = data(:,1);
% input_s = 2*(input-data_min)./(data_max-data_min)-1;
% output_s = net(input_s);
% output = (output_s+1)/2.*(target_max-target_min)+target_min
% output_ca = full(netca(input))

end
