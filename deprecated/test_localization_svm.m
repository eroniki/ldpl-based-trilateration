function [pos_predict] = test_localization_svm(mdl_x, mdl_y, pos_testing, testingdata)
    [pos_predict_x, ~, ~] = svmpredict(pos_testing(:,1), testingdata, mdl_x);
    [pos_predict_y, ~, ~] = svmpredict(pos_testing(:,2), testingdata, mdl_y);
    pos_predict = [pos_predict_x, pos_predict_y];
end

