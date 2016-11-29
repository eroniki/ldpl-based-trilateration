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
%%
create_feature_space_bcc
% create_feature_space_rgu
%%
visualize_feature_space
%% Space Reduction
[coeff, score] = pca(featureSpace_normalized);
reducedDimension = coeff(:,1:12);
trainingset_normalized_lr = trainingset_normalized * reducedDimension;
testingset_normalized_lr = testingset_normalized * reducedDimension;
%% Train SVM-Regressors
mdl_x_svm = svm_train(trainingset_normalized, pos_train(:,1), '-s 4 -t 2 -c 100 -n 0.5');
mdl_y_svm = svm_train(trainingset_normalized, pos_train(:,2), '-s 4 -t 2 -c 100 -n 0.5');

%% Regression
[pos_predict_x, accuracy_x, prob_x] = svmpredict(pos_testing(:,1), testingset_normalized, mdl_x_svm);
[pos_predict_y, accuracy_y, prob_y] = svmpredict(pos_testing(:,2), testingset_normalized, mdl_y_svm);
error_x_svm = pos_predict_x-pos_testing(:,1);
error_y_svm = pos_predict_y-pos_testing(:,2);
%% Linear Regression
mdl_x_lr = fitlm(trainingset_normalized, pos_train(:,1), 'linear');
mdl_y_lr = fitlm(trainingset_normalized, pos_train(:,2), 'linear');
%% Regression
pos_predict_x = predict(mdl_x_lr, testingset_normalized);
pos_predict_y = predict(mdl_y_lr, testingset_normalized);
error_x_lr = pos_predict_x-pos_testing(:,1);
error_y_lr = pos_predict_y-pos_testing(:,2);

%% Residual Analysis
e_x_bar_svm = mean(sqrt(error_x_svm.^2));
e_y_bar_svm = mean(sqrt(error_y_svm.^2));
mse_svm = mean(sqrt(error_x_svm.^2+error_y_svm.^2));

e_x_bar_lr = mean(sqrt(error_x_lr.^2));
e_y_bar_lr = mean(sqrt(error_y_lr.^2));
mse_lr = mean(sqrt(error_x_lr.^2+error_y_lr.^2));

errors = [error_x_svm, error_y_svm, error_x_lr, error_y_lr]
mses = [mse_svm, mse_lr]


%% Kalman     
[~,I] = sort(pos_testing(:,1));
path = create_path(pos_testing(I,:), length(pos_testing));

kalmanFilter_svm = configureKalmanFilter('ConstantVelocity',...
          path(1,:), [1 1]*1e5, [3, 3], 10);
kalmanFilter_lr = configureKalmanFilter('ConstantVelocity',...
          path(1,:), [1 1]*1e5, [3, 3], 10);

sse_w_kalman_svm = 0;
sse_wo_kalman_svm = 0;
      
sse_w_kalman_lr = 0;
sse_wo_kalman_lr = 0;

pos_testing_smooth = pos_testing(I,:);
testingset_normalized_smooth = testingset_normalized(I,:);
for i=1:length(pos_testing)
    predictedLocation_svm = predict(kalmanFilter_svm);
    predictedLocation_lr = predict(kalmanFilter_lr);
    
    [pos_predict_x, accuracy_x, prob_x] = svmpredict(pos_testing_smooth(i,1), testingset_normalized_smooth(i,:), mdl_x_svm);
    [pos_predict_y, accuracy_y, prob_y] = svmpredict(pos_testing_smooth(i,2), testingset_normalized_smooth(i,:), mdl_y_svm);
    pos_predict_x_lr = predict(mdl_x_lr, testingset_normalized_smooth(i,:));
    pos_predict_y_lr = predict(mdl_y_lr, testingset_normalized_smooth(i,:));
    
    corrected_svm = correct(kalmanFilter_svm, [pos_predict_x, pos_predict_y]);
    corrected_lr = correct(kalmanFilter_lr, [pos_predict_x_lr, pos_predict_y_lr]);
    
    figure(1);
    clf;
    scatter(path(i,1),path(i,2), 'k+'); hold on;
    scatter(predictedLocation_svm(1),predictedLocation_svm(2), 'bd'); hold on;
    scatter(pos_predict_x, pos_predict_y, 'ro'); hold on;
    scatter(corrected_svm(1), corrected_svm(2), 'g*');
    grid on; 
    axis([-10 30 -10 30]);
    legend('Ground Truth', 'Kalman Prediction', 'Observation (SVM)', 'Kalman Correction');
    
    error_w_kf = norm(pos_testing_smooth(i,:) - corrected_svm)
    error_wo_kf = norm(pos_testing_smooth(i,:) - [pos_predict_x, pos_predict_y]) 
    sse_w_kalman_svm = sse_w_kalman_svm + error_w_kf;
    sse_wo_kalman_svm = sse_wo_kalman_svm + error_wo_kf;
    title(['Error w/ KF: ', num2str(error_w_kf) , ' Error w/o KF: ', num2str(error_wo_kf)]);
    
    figure(2);
    clf;
    scatter(path(i,1),path(i,2), 'k+'); hold on;
    scatter(predictedLocation_lr(1),predictedLocation_lr(2), 'bd'); hold on;
    scatter(pos_predict_x, pos_predict_y, 'ro'); hold on;
    scatter(corrected_svm(1), corrected_svm(2), 'g*');
    grid on; 
    axis([-10 30 -10 30]);
    legend('Ground Truth', 'Kalman Prediction', 'Observation (LR)', 'Kalman Correction');
    
    error_w_kf_lr = norm(pos_testing_smooth(i,:) - corrected_lr)
    error_wo_kf_lr = norm(pos_testing_smooth(i,:) - [pos_predict_x_lr, pos_predict_y_lr]) 
    sse_w_kalman_lr = sse_w_kalman_lr + error_w_kf_lr;
    sse_wo_kalman_lr = sse_wo_kalman_lr + error_wo_kf_lr;
    title(['Error w/ KF: ', num2str(error_w_kf_lr) , ' Error w/o KF: ', num2str(error_wo_kf_lr)]);
end
%%
mse_wo_kalman_svm = sse_wo_kalman_svm/68;
mse_w_kalman_svm = sse_w_kalman_svm/68;
mse_wo_kalman_lr = sse_wo_kalman_lr/68;
mse_w_kalman_lr = sse_w_kalman_lr/68;
disp(mse_wo_kalman_svm);
disp(mse_w_kalman_svm);
disp(mse_wo_kalman_lr);
disp(mse_w_kalman_lr);