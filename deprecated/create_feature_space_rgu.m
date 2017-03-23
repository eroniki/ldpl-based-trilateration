load('datasets/rtu/PosData.mat');

featureSpace = calibAvgRSS;
mean_meas = mean(featureSpace(:));

featureSpace_normalized = featureSpace - mean_meas;

testingset = testRSS;
testingset_normalized  = testRSS - mean_meas;

pos_train = calibAvgXY;
pos_testing = testXY;
trainingset = calibAvgRSS;
trainingset_normalized = featureSpace_normalized;