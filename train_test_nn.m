clc; close all;
%%
% create_feature_space_ucm;
 
mse_x_nn = [];
mse_y_nn = [];
mse_nn = [];

inputs = trainingset_normalized';
targets = pos_train';

% Create a Fitting Network
hiddenLayerSize = [20, 10, 5, 3];
net = fitnet(hiddenLayerSize);
% net.layers{1}.transferFcn = 'logsig';
% net.layers{2}.transferFcn = 'tansig';
% net.layers{3}.transferFcn = 'elliotsig';
% net.layers{4}.transferFcn = 'satlins';
%%
% Set up Division of Data for Training, Validation, Testing
net.divideFcn = 'dividerand';
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;

net.trainFcn = 'trainbr';
net.trainParam.epochs = 200000;
net.trainParam.showCommandLine = 1;
net.trainParam.max_fail = 0;
net.trainParam.goal = 1*10^-2; 
net.trainParam.show	= 1;

net.trainParam.mu_max = 1*10^40;
net.trainParam.min_grad = 1*10^-6;

net.performFcn='msereg';

%% % Train the Network
[net,tr] = train(net,inputs,targets,'useParallel','yes','showResources','yes');
% [net, tr] = train(net, inputs, targets);
%% % Test the Network
nn_results = test_localization_nn(net, testingset_normalized);
%% % Evaluate the Network
[mse_x_nn, mse_y_nn, mse_nn] = residual_analysis(nn_results, pos_testing)
nBins = 50;
e_nn = sqrt((nn_results(:,1)-pos_testing(:,1)).^2 + (nn_results(:,2)-pos_testing(:,2)).^2);

[cdf_nn, ~, bins_nn] = localization_cdf(e_nn, nBins);
figure;
plot(bins_nn, cdf_nn, 'b'); grid on;