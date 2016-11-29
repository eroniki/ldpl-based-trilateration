function [mdl] = svm_train(features, labels, parameters)
    mdl = svmtrain(labels, features, parameters);
end

