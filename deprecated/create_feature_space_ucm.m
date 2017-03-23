load('datasets/UCM/feature_space.mat');
%%
% [~,I] = sort(pos(:,1));
% [pos, idx] = create_path(pos(I,:), length(pos));
% featureSpace = featureSpace(idx,:);

mean_meas = mean(featureSpace(:));
featureSpace(featureSpace==0) = -200;
featureSpace_normalized = featureSpace - mean_meas;
max_meas = max(featureSpace_normalized (:));
min_meas = min(featureSpace_normalized (:));
diff = max_meas-min_meas;

featureSpace_normalized = featureSpace_normalized / std(featureSpace_normalized(:));

mean_meas = mean(featureSpace(:));

% featureSpace_normalized = featureSpace;
% featureSpace_normalized = featureSpace - mean_meas;
% max_meas = max(featureSpace_normalized (:));
% min_meas = min(featureSpace_normalized (:));
% diff = max_meas-min_meas;

% featureSpace_normalized = featureSpace_normalized / std(featureSpace_normalized(:));
% featureSpace_normalized_image = uint8(255*(featureSpace_normalized  - min(featureSpace_normalized(:)))/(max(featureSpace_normalized(:))-min(featureSpace_normalized(:)))); 

db__ = featureSpace_normalized;
pos__ = pos;
[rows, ~] = size(featureSpace_normalized);

training_id = randsample(1:rows, 2500);

trainingset = db__(training_id,:);
pos_train = pos__(training_id,:);
db__(training_id,:) = [];
pos__(training_id,:) = [];
[rows, ~] = size(db__);
test_id = randsample(1:rows, 620);
testingset = db__(test_id,:);
pos_testing = pos__(test_id,:);

trainingset_normalized = trainingset;
testingset_normalized = testingset;
