function [pos_predict] = test_localization_lr(mdl_x, mdl_y, testingdata)
    pos_predict_x = predict(mdl_x, testingdata);
    pos_predict_y = predict(mdl_y, testingdata);
    pos_predict = [pos_predict_x, pos_predict_y];
end

