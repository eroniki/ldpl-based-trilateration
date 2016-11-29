db = importfile_no_amcl('datasets/bcc/1478808114.txt', 1, 3236);
pos_file = importfile_positions('datasets/bcc/positions.txt', 2, 129);


[nPos,~] = size(pos_file);
T = table();
for i=1:nPos
    rows = db.id == i;
    [nAP, ~] = size(db(rows,:));
    T = [T; db(rows, {'timestamp', 'mac', 'rssi', 'ssid'}), repmat(pos_file(i,2:3),[nAP,1])];
end

% append timestamps
db_(:,1) = table2cell(T(:,1));
% append mac addresses 
db_(:,2) = table2array(T(:,2));
% append rssi 
db_(:,3) = table2cell(T(:,3));
% append ssid
db_(:,4) = table2cell(T(:,4));
% append x position
db_(:,5) = table2cell(T(:,5));
% append y position
db_(:,6) = table2cell(T(:,6));

[nRows, ~] = size(db);
% loop over the cell

%% Find unique AP's in the environment and the timestamps
% The MATLAB function unique returns an ordered vector.
uniqueAP = unique(db_(:,2));
uniqueTimeStamps = unique(int64(cell2mat(db_(:,1))));
    
nAP = numel(uniqueAP);
nTimeStamp = numel(uniqueTimeStamps);
% Create an empty feature space
featureSpace = zeros(nTimeStamp, nAP); 
% Construct the feature space

% for timestamp=1:nTimeStamp
%     ts = int64(cell2mat(db_(:,1)));
%     db__ = db_(ts == uniqueTimeStamps(timestamp), :);
%     [nObs, ~] = size(db__);
%     for ap=1:nObs
%         ap_index = uniqueAP == db__(timestamp, 2)
%         featureSpace(timestamp, ap_index) = db__(time);
%         
%     end
%     
% end
%% Find Unique Locations
pos = zeros(nTimeStamp,2);
for jj=1:nTimeStamp
    rows = T.timestamp == uniqueTimeStamps(jj);
    cols = {'X','Y'};
    tmp = table2array(T(rows,cols));
    pos(jj,:) = tmp(1,:);
end

for row = 1:nRows
    mac = T(row,'mac');
    rssi = table2array(T(row, 'rssi'));

    ts = int64(table2array(T(row, 'timestamp')));
    
    featureSpace(ts == uniqueTimeStamps, find(ismember(uniqueAP, table2array(mac)))) = rssi; 
end
%%
mean_meas = mean(featureSpace(:));

featureSpace_normalized = featureSpace - mean_meas;
max_meas = max(featureSpace_normalized (:));
min_meas = min(featureSpace_normalized (:));
diff = max_meas-min_meas;

featureSpace_normalized = featureSpace_normalized / std(featureSpace_normalized(:));
featureSpace_normalized_image = uint8(255*(featureSpace_normalized  - min(featureSpace_normalized(:)))/(max(featureSpace_normalized(:))-min(featureSpace_normalized(:)))); 

db__ = featureSpace_normalized;
pos__ = pos;
[rows, ~] = size(featureSpace_normalized);

training_id = randsample(1:rows, 100);

trainingset = db__(training_id,:);
pos_train = pos__(training_id,:);
db__(training_id,:) = [];
pos__(training_id,:) = [];
[rows, ~] = size(db__);
test_id = randsample(1:rows, 25);
testingset = db__(test_id,:);
pos_testing = pos__(test_id,:);

trainingset_normalized = trainingset;
testingset_normalized = testingset;
