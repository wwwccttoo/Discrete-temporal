
clear all; close all; clc;


%% Set Flags for Various Options

% MODEL = exact form of model specified in loadProblemData()
%0 = standard model

% OFFSET = whether or not to use offset-free disturbances
%0 = no
%1 = yes

% MPCTYPE = exact form of mpc to use
%0 = Rawlings style wherein target tracker is separate from mpc
%1 = Limon style wherein target tracker is embedded in mpc

% MPCDNN = whether or not to use neural network approximation for mpc
%0 = no
%1 = yes

% TTDNN = whether or not to use neural network approximation for target tracker
%0 = no
%1 = yes

% TTCORRECTION = whether or not to correct offset using target tracker
%0 = no
%1 = yes

% HYPERPARAMOPT = whether or not to optimize hyperparameters in neural network
%0 = no
%1 = yes

% PERFORMANCEVERIFICATION = whether or not to run performance verification step for neural network
%0 = no
%1 = yes

% first configuration
i = 1;
flags{i}.model = 0;
flags{i}.offset = 1;
flags{i}.mpcType = 1;
flags{i}.mpcDNN = 0;
flags{i}.ttDNN = 0;
flags{i}.ttCorrection = 0;
flags{i}.hyperParamOpt = 0;
flags{i}.performanceVerification = 0;

% second configuration
% i = 2;
% flags{i}.model = 0;
% flags{i}.offset = 1;
% flags{i}.mpcType = 1;
% flags{i}.mpcDNN = 1;
% flags{i}.ttDNN = 0;
% flags{i}.ttCorrection = 1;
% flags{i}.hyperParamOpt = 0;
% flags{i}.performanceVerification = 0;

% % third configuration
% i = 3;
% flags{i}.model = 0;
% flags{i}.offset = 1;
% flags{i}.mpcType = 1;
% flags{i}.mpcDNN = 1;
% flags{i}.ttDNN = 0;
% flags{i}.ttCorrection = 0;
% flags{i}.hyperParamOpt = 0;
% flags{i}.performanceVerification = 0;

% % fourth configuration
% i = 4;
% flags{i}.model = 0;
% flags{i}.offset = 0;
% flags{i}.mpcType = 0;
% flags{i}.mpcDNN = 0;
% flags{i}.ttDNN = 0;
% flags{i}.ttCorrection = 0;
% flags{i}.hyperParamOpt = 0;
% flags{i}.performanceVerification = 0;

% flags for plotting
plots.phase = 1;
plots.controlledOutput = 1;
plots.input = 1;
plots.stateEstimate = 1;
plots.disturbanceEstimate = 1;
plots.performanceVerification = 1;
plots.processVerification = 1;

% flags for saving [empty for no]
saveConfig = [];

% flags for loading [empty for no]; flags overwritten if loaded
loadConfig = [];

%% Control inputs from B.O.
Tref = 50;
Iref = 2000;
PTref = 93;
dA = 1;
Aref = [1.4016 -14.4395 12.5570]';

%% Construct the Controllers Specified by Flags

% add the internal functions to the path
addpath(genpath('./Functions'))

% load previous configuration, if specified
if ~isempty(loadConfig)
    [c, flags] = loadConfigurations(loadConfig);
    nc = length(flags);

% otherwise run the code to construct all values
else
    % get number of unique configurations that must be built
    nc = length(flags);

    % loop over the various configuration options
    c = cell(nc,1);
    for i = 1:nc

        % set initial configuration to baseline
        c{i} = loadProblemData(c{i}, flags{i}, Tref, Iref, PTref, Aref, dA);

        % get target tracker
        c{i} = getTargetTracker(c{i});

        % get EKF
        c{i} = getEKF(c{i});

        % construct the corresponding mpc type
        c{i} = getMPC(c{i}, flags{i}.mpcType);

        % build a neural network approximation of target tracker
        if flags{i}.ttDNN == 1
            % get training data for tracker
            c{i} = getTargetTrainingData(c{i}, flags{i});

            % train the neural network approximation
            [ttnet, ttnetca] = trainNeuralNetwork(c{i}.tt.data, c{i}.tt.target, c{i}.tt.nodes, c{i}.tt.layers);
            c{i}.tt.net = ttnet;
            c{i}.tt.netca = ttnetca;
        end

        % build a neural network approximation of mpc
        if flags{i}.mpcDNN == 1
            % get training data for mpc
            c{i} = getMPCTrainingData(c{i}, flags{i});

            % train the neural network approximation
            if flags{i}.hyperParamOpt == 1
                [mpcnet, mpcnetca] = trainNeuralNetwork(c{i}.mpc.data, c{i}.mpc.target, c{i}.mpc.nodes, c{i}.mpc.layers, c{i}.bo.opts);
            else
                [mpcnet, mpcnetca] = trainNeuralNetwork(c{i}.mpc.data, c{i}.mpc.target, c{i}.mpc.nodes, c{i}.mpc.layers);
            end
            c{i}.mpc.net = mpcnet;
            c{i}.mpc.netca = mpcnetca;

            % performance verification
            if flags{i}.performanceVerification == 1
                c{i} = performanceVerificationDNN(c{i}, flags{i});
            end
        end

        % run closed-loop simulation
        % c{i} = runClosedLoopSimulation(c{i}, flags{i}, 1);
        c{i} = runClosedLoopSimulation(c{i}, flags{i}, 1, Tref, Iref, PTref, Aref, dA);
    end
end

% save information, if specified
if ~isempty(saveConfig)
    saveConfigurations(c, flags, saveConfig)
end


%% Plot Data

% plotting options
colors = {'-b', '--r', '-.g', ':m'};
lineWidth = {2, 2, 2, 2};

% % plot state phase plot (first 2 states)
% if plots.phase == 1
%     plotPhase(c, colors, lineWidth)
%     axis([-1.2, 1.2, -1.2, 1.2])
% end

% plot controlled outputs
if plots.controlledOutput == 1
    plotControlledOutput(c, colors, lineWidth)
end

% plot inputs
if plots.input == 1
    plotInput(c, colors, lineWidth)
end

% plot state estimates
if plots.stateEstimate == 1
    plotStateEstimate(c, colors, lineWidth)
end

% plot disturbance estimates
if plots.disturbanceEstimate == 1
    plotDisturbanceEstimate(c, colors, lineWidth)
end

% plot process results
if plots.processVerification == 1
    plotProcessResults(c, colors, lineWidth)
end

% % plot performance verification results
% if plots.performanceVerification == 1
%     plotPerformanceVerification(c, flags)
% end

% remove the internal functions to the path
rmpath(genpath('./Functions'))
