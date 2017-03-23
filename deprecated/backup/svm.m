%% Clean up and initialize
clear all, close all, clc;
%% Load data and visualize it
load('PosData.mat');
[errMean, errMedian] = test(calibAvgRSS, calibAvgXY, testRSS, testXY);


featureSpace = calibAvgRSS;
mean_meas = mean(featureSpace(:));

featureSpace_normalized = featureSpace - mean_meas;
% max_meas = max(featureSpace_normalized (:));
% min_meas = min(featureSpace_normalized (:));
% diff = max_meas-min_meas;

% featureSpace_normalized = featureSpace_normalized / std(featureSpace_normalized(:));

testRSS_normalized = testRSS - mean_meas;
% testRSS_normalized = testRSS_normalized /std(featureSpace_normalized(:));

figure(1); imagesc(featureSpace); title('Raw RSSI Measurements'); xlabel('RSSI Readings n_{AP} = 50'); ylabel('Positions n_{pos} = 82'); colorbar();
figure(2); imagesc(featureSpace_normalized);  title('Centered RSSI Measurements'); xlabel('RSSI Readings n_{AP} = 50'); ylabel('Positions n_{pos} = 82'); colormap(jet); colorbar;
saveas(1,'outputs/raw.jpg');
saveas(2,'outputs/centered.jpg');

%% Support Vector Regression
svm_object_x = svmtrain(calibAvgXY(:,1), featureSpace_normalized, '-s 4 -t 2 -n 0.5 -c 1');
pos_predict_x =svmpredict(testXY(:,1), testRSS_normalized, svm_object_x);
error_x = pos_predict_x-testXY(:,1);

svm_object_y = svmtrain(calibAvgXY(:,2), featureSpace_normalized, '-s 4 -t 2 -n 0.5 -c 1');
pos_predict_y =svmpredict(testXY(:,2), testRSS_normalized, svm_object_x);
error_y = pos_predict_y-testXY(:,2);

[testXY, pos_predict_x, pos_predict_y];
e_x_bar = mean(sqrt(error_x.^2))
e_y_bar = mean(sqrt(error_y.^2))
mse = mean(sqrt(error_x.^2+error_y.^2))

%% Linear Regression Fit

mdl_x = fitlm(featureSpace_normalized, calibAvgXY(:,1), 'linear');
pos_predict_x = predict(mdl_x, testRSS_normalized);

mdl_y = fitlm(featureSpace_normalized, calibAvgXY(:,2), 'linear');
pos_predict_y = predict(mdl_y, testRSS_normalized);

error_x = pos_predict_x-testXY(:,1);
error_y = pos_predict_y-testXY(:,2);
[testXY, pos_predict_x, pos_predict_y];
e_x_bar = mean(sqrt(error_x.^2))
e_y_bar = mean(sqrt(error_y.^2))
mse = mean(sqrt(error_x.^2+error_y.^2))

%% PCA
[coeff, score] = pca(featureSpace);
reducedDimension = coeff(:,1:10);
featureSpace_reduced = featureSpace * reducedDimension;
testRSS_reduced = testRSS * reducedDimension;
%% Support Vector Regression w/ PCA
svm_object_x = svmtrain(calibAvgXY(:,1), featureSpace_reduced, '-s 4 -t 2 -n 0.5 -c 1');
pos_predict_x =svmpredict(testXY(:,1), testRSS_reduced, svm_object_x);
error_x = pos_predict_x-testXY(:,1);

svm_object_y = svmtrain(calibAvgXY(:,2), featureSpace_reduced, '-s 4 -t 2 -n 0.5 -c 1');
pos_predict_y =svmpredict(testXY(:,2), testRSS_reduced, svm_object_x);
error_y = pos_predict_y-testXY(:,2);

[testXY, pos_predict_x, pos_predict_y];
e_x_bar = mean(sqrt(error_x.^2))
e_y_bar = mean(sqrt(error_y.^2))
mse = mean(sqrt(error_x.^2+error_y.^2))

%% Linear Regression Fit w/ PCA

mdl_x = fitlm(featureSpace_reduced, calibAvgXY(:,1), 'linear');
pos_predict_x = predict(mdl_x, testRSS_reduced);

mdl_y = fitlm(featureSpace_reduced, calibAvgXY(:,2), 'linear');
pos_predict_y = predict(mdl_y, testRSS_reduced);

error_x = pos_predict_x-testXY(:,1);
error_y = pos_predict_y-testXY(:,2);
[testXY, pos_predict_x, pos_predict_y];
e_x_bar = mean(sqrt(error_x.^2))
e_y_bar = mean(sqrt(error_y.^2))
mse = mean(sqrt(error_x.^2+error_y.^2))

% %%  Sort Testing Points
% pos_testing = testXY;
% start = pos_testing(1,:);
% start_vec = repmat(start, [length(pos_testing),1]);
% dist = sum((((start_vec - pos_testing).^2))');
% [dist_sorted, I] = sort(dist);
% pos_testing_sorted = pos_testing(I,:);
% testingset_sorted = testRSS(I,:);
% %% Tracking with Kalman
% kalmanFilter = configureKalmanFilter('ConstantVelocity',...
%           start, [1 1]*1e5, [std(pos_predict_x), std(pos_predict_y)], 15);
% % StateTransitionModel = eye(4,4);
% % MeasurementModel = [1 0 0 0; 0 0 1 0];
% % % ControlModel = [1 1 0 0; 0 1 0 0; 0 0 1 1; 0 0 0 1];
% % MeasurementNoise = [std(pos_predict_x), std(pos_predict_y)];
% % ProcessNoise = 5*eye(4);
% % kalmanFilter = vision.KalmanFilter(StateTransitionModel,MeasurementModel,ControlModel);   
% 
% regress_result = [];
% predictedLocation = [];
% trackedLocation = [];
% 
% for jj=2:length(pos_testing)  
% %     u_x = pos_testing_sorted(jj,1) - pos_testing_sorted(jj-1,1);
% %     u_y = pos_testing_sorted(jj,1) - pos_testing_sorted(jj-1,1);
% %     u = [0; u_x; 0; u_y];
% %     pos_predict_x = svmpredict(pos_testing_sorted(jj,1), testingset_sorted(jj,:), svm_object_x);
% %     pos_predict_y = svmpredict(pos_testing_sorted(jj,2), testingset_sorted(jj,:), svm_object_y);
%     pos_predict_x = predict(mdl_x, testingset_sorted(jj,:));
%     pos_predict_y = predict(mdl_y, testingset_sorted(jj,:));
%     regress_result = [regress_result;pos_predict_x, pos_predict_y];
% 
%     predictedLocation = [predictedLocation; predict(kalmanFilter)];
% %     predictedLocation = [predictedLocation; predict(kalmanFilter, u)];
%     
%     trackedLocation = [trackedLocation; correct(kalmanFilter, [pos_predict_x, pos_predict_y])];
% end
% mse_svm = 0;
% mse_kalman = 0;
% for jj=2:length(pos_testing)-1    
%     figure(jj);
%     plot(pos_testing_sorted(jj,1), pos_testing_sorted(jj,2), 'c>', regress_result(jj,1), regress_result(jj,2), 'bd', predictedLocation(jj,1), predictedLocation(jj,2), 'rs', trackedLocation(jj,1), trackedLocation(jj,2), 'g*'); grid on; hold on;
%     axis([-10, 100, -10, 40]);
%     legend('Ground Truth', 'Regression Result', 'Prediction (Kalman)', 'Correction (Kalman)');
%     error_kalman = sqrt(sum(((trackedLocation(jj,:) - pos_testing_sorted(jj,:)).^2)'));
%     error_svm = sqrt(sum(((regress_result(jj,:) - pos_testing_sorted(jj,:)).^2)'));
%     mse_svm = mse_svm + error_svm;
%     mse_kalman = mse_kalman + error_kalman;
%     title(['Error (w/ Kalman): ', num2str(error_kalman), ' m', 'Error (w/o Kalman): ', num2str(error_svm), ' m']);
%     drawnow;   
% %     pause;
% end
% figure; plot(pos_testing_sorted(:,1), pos_testing_sorted(:,2), 'r-*')
% disp(mse_svm/67);
% disp(mse_kalman/67);
% %% Visualize the results
% % 
% % for jj=1:length(testXY)
% %     figure(jj); plot(testXY(jj,1), testXY(jj,2), 'b*', pos_predict_x(jj), pos_predict_y(jj), 'r*'); grid on; title(['Error: ', num2str(sqrt(error_x(jj)^2+error_y(jj)^2)), ' m']);
% %     legend('Testing Location', 'Predicted Location');
% %     drawnow;    
% % end
% % % 
% % %% Save Results 
% % h =  findobj('type','figure');
% % % n = length(h);
% % 
% % for jj=1:n
% %     saveas(jj, ['outputs/', num2str(jj), '.jpg']);
% % end