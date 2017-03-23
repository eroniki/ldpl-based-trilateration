n_dimensions = 65;
mse_lr = zeros(n_dimensions,1); 
mse_svm = zeros(n_dimensions,1); 
for i=1:n_dimensions
    create_feature_space_rgu
    [coeff, score] = pca(featureSpace_normalized);
    reducedDimension = coeff(:,1:i);
    trainingset_normalized = trainingset_normalized * reducedDimension;
    testingset_normalized = testingset_normalized * reducedDimension;
    %% Linear Regression
    mdl_x_lr = fitlm(trainingset_normalized, pos_train(:,1), 'linear');
    mdl_y_lr = fitlm(trainingset_normalized, pos_train(:,2), 'linear');
    %% Regression
    pos_predict_x = predict(mdl_x_lr, testingset_normalized);
    pos_predict_y = predict(mdl_y_lr, testingset_normalized);
    error_x_lr = pos_predict_x-pos_testing(:,1);
    error_y_lr = pos_predict_y-pos_testing(:,2);
    e_x_bar_lr = mean(sqrt(error_x_lr.^2));
    e_y_bar_lr = mean(sqrt(error_y_lr.^2));
    mse_lr(i) = mean(sqrt(error_x_lr.^2+error_y_lr.^2));
end

figure(1); plot(1:n_dimensions, mse_lr); grid on;

for i=1:n_dimensions
    create_feature_space_rgu
    [coeff, score] = pca(featureSpace_normalized);
    reducedDimension = coeff(:,1:i);
    trainingset_normalized = trainingset_normalized * reducedDimension;
    testingset_normalized = testingset_normalized * reducedDimension;
    %% Train SVM-Regressors
    mdl_x_svm = svm_train(trainingset_normalized, pos_train(:,1), '-s 4 -t 2 -c 100 -n 0.5');
    mdl_y_svm = svm_train(trainingset_normalized, pos_train(:,2), '-s 4 -t 2 -c 100 -n 0.5');
    %% Regression
    [pos_predict_x, accuracy_x, prob_x] = svmpredict(pos_testing(:,1), testingset_normalized, mdl_x_svm);
    [pos_predict_y, accuracy_y, prob_y] = svmpredict(pos_testing(:,2), testingset_normalized, mdl_y_svm);
    error_x_svm = pos_predict_x-pos_testing(:,1);
    error_y_svm = pos_predict_y-pos_testing(:,2);
    e_x_bar_svm = mean(sqrt(error_x_svm.^2));
    e_y_bar_svm = mean(sqrt(error_y_svm.^2));
    mse_svm(i) = mean(sqrt(error_x_svm.^2+error_y_svm.^2));
end

figure(2); plot(1:n_dimensions, mse_svm); grid on;