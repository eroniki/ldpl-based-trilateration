%% Train SVM-Regressors
mdl_x_svm = svm_train(trainingset_normalized, pos_train(:,1), '-s 4 -t 2 -c 100 -n 0.5');
mdl_y_svm = svm_train(trainingset_normalized, pos_train(:,2), '-s 4 -t 2 -c 100 -n 0.5');


%% Train Linear Regressor
mdl_x_lr = fitlm(trainingset_normalized, pos_train(:,1), 'linear');
mdl_y_lr = fitlm(trainingset_normalized, pos_train(:,2), 'linear');

%% Train Neural Network
inputs = trainingset_normalized';
targets = pos_train';
 
% Create a Fitting Network
hiddenLayerSize = 10;
net = fitnet(hiddenLayerSize);

% Set up Division of Data for Training, Validation, Testing
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;
net.trainFcn = 'trainbr';
net.trainParam.showCommandLine = 1;
net.performFcn='msereg';
% net.performParam.ratio=0.5; 
% net.trainParam.goal=1*10^-6; 

% Train the Network
[net,tr] = train(net,inputs,targets,'useParallel','yes','showResources','yes');
% Test the Network
outputs = net(testingset_normalized');
% [r,m,b] = regression(pos_testing',testingset_normalized')
% plotregression(pos_testing',testingset_normalized')
% performance = perform(net,testingset_normalized',outputs)
 
% View the Network
% view(net)