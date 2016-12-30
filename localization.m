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

mse_x_svm = zeros(nTest,1);
mse_y_svm = zeros(nTest,1);
mse_svm = zeros(nTest,1);

mse_x_lr= zeros(nTest,1);
mse_y_lr = zeros(nTest,1);
mse_lr = zeros(nTest,1);
nn_results = [];
svm_results = [];
lr_results = [];
% create_feature_space_rgu
% create_feature_space_ucm
% train
for cnt=1:nTest
    cnt
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
    output_nn = test_localization_nn(net, testingset_normalized);
    output_svm = test_localization_svm(mdl_x_svm, mdl_y_svm, pos_testing, testingset_normalized);
    output_lr = test_localization_lr(mdl_x_lr, mdl_y_lr, testingset_normalized);
    [mse_x_nn(cnt), mse_y_nn(cnt), mse_nn(cnt)] = residual_analysis(output_nn, pos_testing);
    [mse_x_svm(cnt), mse_y_svm(cnt), mse_svm(cnt)] = residual_analysis(output_svm, pos_testing);
    [mse_x_lr(cnt), mse_y_lr(cnt), mse_lr(cnt)] = residual_analysis(output_lr, pos_testing);

    nn_results = [nn_results; output_nn];
    svm_results = [svm_results; output_svm];
    lr_results = [lr_results; output_lr];
end
%% Plot Results
if nTest == 1
    figure; 
    str = {'MSE_{LR}'; 'MSE_{SVM}'; 'MSE_{NN}'};
	bar([mse_lr, mse_svm, mse_nn]);
    title('Mean Squared Error for Different Regressors');
    ylabel('Localization Error MSE (m)');
%     set(gca, 'XTickLabel',str, 'XTick',1:numel(str));
    grid on;
    
    figure;
    str = {'MSE_{LR,X}'; 'MSE_{LR,Y}'; 'MSE_{SVM,X}'; 'MSE_{SVM,Y}'; 'MSE_{NN,X}'; 'MSE_{NN,Y}'};
    bar([mse_x_lr, mse_y_lr, mse_x_svm, mse_y_svm, mse_x_nn, mse_y_nn]);
%     set(gca, 'XTickLabel', str, 'XTick',1:numel(str));
    grid on;
else
    figure(66); boxplot([mse_lr, mse_svm, mse_nn], 'labels', {'LR', 'SVM' , 'NN'}); grid on;
    title('Mean Squared Error Attained After n=20 Repetition'); 
    ylabel('MSE (m)');
    xlabel('Regressors');    
    drawnow;
    
    figure(99); boxplot([mse_x_lr, mse_y_lr, mse_x_svm, mse_y_svm, mse_x_nn, mse_y_nn], ...
        'labels', {'x,LR', 'y,LR', 'x,SVM', 'y,SVM', 'x,NN', 'y,NN'}); grid on;
    title('MSE for Different Regressors Along X- and Y- Axis n=20');
    ylabel('MSE (m)');
    xlabel('Regressors');
    drawnow;
    
end

figure; plot(pos_train(:,1), pos_train(:,2), 'r*'); grid on;
figure; plot(pos_testing(:,1), pos_testing(:,2), 'g*'); grid on; xlabel('X'), ylabel('Y'); grid on;
    
%%
% %% Kalman     
% [~,I] = sort(pos_testing(:,1));
% path = create_path(pos_testing(I,:), length(pos_testing));
% 
% kalmanFilter_svm = configureKalmanFilter('ConstantVelocity',...
%           path(1,:), [1 1]*1e5, [3, 3], 10);
% kalmanFilter_lr = configureKalmanFilter('ConstantVelocity',...
%           path(1,:), [1 1]*1e5, [3, 3], 10);
% 
% sse_w_kalman_svm = 0;
% sse_wo_kalman_svm = 0;
%       
% sse_w_kalman_lr = 0;
% sse_wo_kalman_lr = 0;
% 
% pos_testing_smooth = pos_testing(I,:);
% testingset_normalized_smooth = testingset_normalized(I,:);
% for i=1:length(pos_testing)
%     predictedLocation_svm = predict(kalmanFilter_svm);
%     predictedLocation_lr = predict(kalmanFilter_lr);
%     
%     [pos_predict_x, accuracy_x, prob_x] = svmpredict(pos_testing_smooth(i,1), testingset_normalized_smooth(i,:), mdl_x_svm);
%     [pos_predict_y, accuracy_y, prob_y] = svmpredict(pos_testing_smooth(i,2), testingset_normalized_smooth(i,:), mdl_y_svm);
%     pos_predict_x_lr = predict(mdl_x_lr, testingset_normalized_smooth(i,:));
%     pos_predict_y_lr = predict(mdl_y_lr, testingset_normalized_smooth(i,:));
%     
%     corrected_svm = correct(kalmanFilter_svm, [pos_predict_x, pos_predict_y]);
%     corrected_lr = correct(kalmanFilter_lr, [pos_predict_x_lr, pos_predict_y_lr]);
%     
%     figure(1);
%     clf;
%     scatter(path(i,1),path(i,2), 'k+'); hold on;
%     scatter(predictedLocation_svm(1),predictedLocation_svm(2), 'bd'); hold on;
%     scatter(pos_predict_x, pos_predict_y, 'ro'); hold on;
%     scatter(corrected_svm(1), corrected_svm(2), 'g*');
%     grid on; 
%     axis([-10 30 -10 30]);
%     legend('Ground Truth', 'Kalman Prediction', 'Observation (SVM)', 'Kalman Correction');
%     
%     error_w_kf = norm(pos_testing_smooth(i,:) - corrected_svm)
%     error_wo_kf = norm(pos_testing_smooth(i,:) - [pos_predict_x, pos_predict_y]) 
%     sse_w_kalman_svm = sse_w_kalman_svm + error_w_kf;
%     sse_wo_kalman_svm = sse_wo_kalman_svm + error_wo_kf;
%     title(['Error w/ KF: ', num2str(error_w_kf) , ' Error w/o KF: ', num2str(error_wo_kf)]);
%     
%     figure(2);
%     clf;
%     scatter(path(i,1),path(i,2), 'k+'); hold on;
%     scatter(predictedLocation_lr(1),predictedLocation_lr(2), 'bd'); hold on;
%     scatter(pos_predict_x, pos_predict_y, 'ro'); hold on;
%     scatter(corrected_svm(1), corrected_svm(2), 'g*');
%     grid on; 
%     axis([-10 30 -10 30]);
%     legend('Ground Truth', 'Kalman Prediction', 'Observation (LR)', 'Kalman Correction');
%     
%     error_w_kf_lr = norm(pos_testing_smooth(i,:) - corrected_lr)
%     error_wo_kf_lr = norm(pos_testing_smooth(i,:) - [pos_predict_x_lr, pos_predict_y_lr]) 
%     sse_w_kalman_lr = sse_w_kalman_lr + error_w_kf_lr;
%     sse_wo_kalman_lr = sse_wo_kalman_lr + error_wo_kf_lr;
%     title(['Error w/ KF: ', num2str(error_w_kf_lr) , ' Error w/o KF: ', num2str(error_wo_kf_lr)]);
% end
% %%
% mse_wo_kalman_svm = sse_wo_kalman_svm/68;
% mse_w_kalman_svm = sse_w_kalman_svm/68;
% mse_wo_kalman_lr = sse_wo_kalman_lr/68;
% mse_w_kalman_lr = sse_w_kalman_lr/68;
% disp(mse_wo_kalman_svm);
% disp(mse_w_kalman_svm);
% disp(mse_wo_kalman_lr);
% disp(mse_w_kalman_lr);