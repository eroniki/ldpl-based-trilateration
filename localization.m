clear all, close all, clc;
%%
% Usage: svm-train [options] training_set_file [model_file]
% options:
% -s svm_type : set type of SVM (default 0)
% 	0 -- C-SVC
% 	1 -- nu-SVC
% 	2 -- one-class SVM
% 	3 -- epsilon-SVR
% 	4 -- nu-SVR
% -t kernel_type : set type of kernel function (default 2)
% 	0 -- linear: u'*v
% 	1 -- polynomial: (gamma*u'*v + coef0)^degree
% 	2 -- radial basis function: exp(-gamma*|u-v|^2)
% 	3 -- sigmoid: tanh(gamma*u'*v + coef0)
% 	4 -- precomputed kernel (kernel values in training_set_file)
% -d degree : set degree in kernel function (default 3)
% -g gamma : set gamma in kernel function (default 1/num_features)
% -r coef0 : set coef0 in kernel function (default 0)
% -c cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)
% -n nu : set the parameter nu of nu-SVC, one-class SVM, and nu-SVR (default 0.5)
% -p epsilon : set the epsilon in loss function of epsilon-SVR (default 0.1)
% -m cachesize : set cache memory size in MB (default 100)
% -e epsilon : set tolerance of termination criterion (default 0.001)
% -h shrinking : whether to use the shrinking heuristics, 0 or 1 (default 1)
% -b probability_estimates : whether to train a SVC or SVR model for probability estimates, 0 or 1 (default 0)
% -wi weight : set the parameter C of class i to weight*C, for C-SVC (default 1)
% -v n: n-fold cross validation mode
% -q : quiet mode (no outputs)
%%  Test regressors
nTest = 1;

mse_x_nn = zeros(nTest,1);
mse_y_nn = zeros(nTest,1);
mse_nn = zeros(nTest,1);


mse_x_rnn = zeros(nTest,1);
mse_y_rnn = zeros(nTest,1);
mse_rnn = zeros(nTest,1);

mse_x_svm = zeros(nTest,1);
mse_y_svm = zeros(nTest,1);
mse_svm = zeros(nTest,1);

mse_x_lr= zeros(nTest,1);
mse_y_lr = zeros(nTest,1);
mse_lr = zeros(nTest,1);

rnn_results = [];
nn_results = [];
svm_results = [];
lr_results = [];

%%
for cnt=1:nTest
    disp(cnt);
    %%
%     create_feature_space_bcc
%     create_feature_space_rgu
	create_feature_space_ucm
    %%
    % visualize_feature_space
    %% Space Reduction
    % [coeff, score] = pca(featureSpace_normalized);
    % reducedDimension = coeff(:,1:12);
    % trainingset_normalized_lr = trainingset_normalized * reducedDimension;
    % testingset_normalized_lr = testingset_normalized * reducedDimension;
    %% Train regressors
    train
    %% Test
    output_nn = test_localization_nn(net, testingset_normalized);
%     output_rnn = test_localization_rnn(netRNN, testingset_normalized);
%     output_svm = test_localization_svm(mdl_x_svm, mdl_y_svm, pos_testing, testingset_normalized);
%     output_lr = test_localization_lr(mdl_x_lr, mdl_y_lr, testingset_normalized);
   
    [mse_x_nn(cnt), mse_y_nn(cnt), mse_nn(cnt)] = residual_analysis(output_nn, pos_testing);
%     [mse_x_rnn(cnt), mse_y_rnn(cnt), mse_rnn(cnt)] = residual_analysis(output_rnn, pos_testing);
%     [mse_x_svm(cnt), mse_y_svm(cnt), mse_svm(cnt)] = residual_analysis(output_svm, pos_testing);
%     [mse_x_lr(cnt), mse_y_lr(cnt), mse_lr(cnt)] = residual_analysis(output_lr, pos_testing);

    nn_results = [nn_results; output_nn];
%     rnn_results = [rnn_results; output_rnn];
%     svm_results = [svm_results; output_svm];
%     lr_results = [lr_results; output_lr];
end
%% Plot Results
if nTest == 1
    figure; 
    str = {'MSE_{RNN}';'MSE_{LR}'; 'MSE_{SVM}'; 'MSE_{NN}'};
	bar([mse_rnn, mse_lr, mse_svm, mse_nn]);
    title('Mean Squared Error for Different R2egressors');
    ylabel('Localization Error MSE (m)');
    set(gca, 'XTickLabel',str, 'XTick',1:numel(str));
    grid on;
    
    figure;
    str = {'MSE_{RNN,X}'; 'MSE_{RNN,Y}'; 'MSE_{LR,X}'; 'MSE_{LR,Y}'; 'MSE_{SVM,X}'; 'MSE_{SVM,Y}'; 'MSE_{NN,X}'; 'MSE_{NN,Y}'};
    bar([mse_x_rnn, mse_y_rnn, mse_x_lr, mse_y_lr, mse_x_svm, mse_y_svm, mse_x_nn, mse_y_nn]);
    set(gca, 'XTickLabel', str, 'XTick',1:numel(str));
    grid on;
else
    figure(66); boxplot([mse_rnn, mse_lr, mse_svm, mse_nn], 'labels', {'RNN', 'LR', 'SVM' , 'NN'}); grid on;
    title('Mean Squared Error Attained After n=20 Repetition'); 
    ylabel('MSE (m)');
    xlabel('Regressors');    
    drawnow;
    
    figure(99); boxplot([mse_x_rnn, mse_y_rnn, mse_x_lr, mse_y_lr, mse_x_svm, mse_y_svm, mse_x_nn, mse_y_nn], ...
        'labels', {'x,RNN', 'y,RNN', 'x,LR', 'y,LR', 'x,SVM', 'y,SVM', 'x,NN', 'y,NN'}); grid on;
    title('MSE for Different Regressors Along X- and Y- Axis n=20');
    ylabel('MSE (m)');
    xlabel('Regressors');
    drawnow;
    
end

figure; plot(pos_train(:, 1), pos_train(:,2), 'r*'); grid on;
figure; plot(pos_testing(:,1), pos_testing(:,2), 'g*'); grid on; xlabel('X'), ylabel('Y'); grid on;

%%
nBins = 100;
e_lr = sqrt((lr_results(:,1)-pos_testing(:,1)).^2 + (lr_results(:,2)-pos_testing(:,2)).^2);
e_svm = sqrt((svm_results(:,1)-pos_testing(:,1)).^2 + (svm_results(:,2)-pos_testing(:,2)).^2);
e_rnn = sqrt((rnn_results(:,1)-pos_testing(:,1)).^2 + (rnn_results(:,2)-pos_testing(:,2)).^2);
e_nn = sqrt((nn_results(:,1)-pos_testing(:,1)).^2 + (nn_results(:,2)-pos_testing(:,2)).^2);
[cdf_lr, ~, bins_lr] = localization_cdf(e_lr, nBins);
[cdf_svm, ~, bins_svm] = localization_cdf(e_svm, nBins);
[cdf_rnn, ~, bins_rnn] = localization_cdf(e_rnn, nBins);
[cdf_nn, ~, bins_nn] = localization_cdf(e_nn, nBins);
figure;
plot(bins_rnn, cdf_rnn, 'k', bins_lr, cdf_lr, 'r', bins_svm, cdf_svm, 'g', bins_nn, cdf_nn, 'b');
legend('RNN', 'LR', 'SVM', 'NN'); grid on;