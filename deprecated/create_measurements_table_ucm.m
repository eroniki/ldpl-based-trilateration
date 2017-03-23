load('datasets/UCM/T.mat');
%%
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

[nRows, ~] = size(T);
% loop over the cell

%% Find unique AP's in the environment and the timestamps
% The MATLAB function unique returns an ordered vector.
uniqueAP = unique(T.mac);
uniqueTimeStamps = unique(unique(T.timestamp));
    
nAP = numel(uniqueAP);
nTimeStamp = numel(uniqueTimeStamps);
% Create an empty feature space
featureSpace = zeros(nTimeStamp, nAP); 
% Construct the feature space

%% Find Unique Locations
pos = zeros(nTimeStamp,2);
for jj=1:nTimeStamp
    rows = T.timestamp == uniqueTimeStamps(jj);
    cols = {'x','y'};
    tmp = table2array(T(rows,cols));
    pos(jj,:) = tmp(1,:);
end
%% 
for row = 1:nRows
    mac = T(row,'mac');
    rssi = table2array(T(row, 'rssi'));

    ts = int64(table2array(T(row, 'timestamp')));
    
    featureSpace(ts == uniqueTimeStamps, find(ismember(uniqueAP, table2array(mac)))) = rssi; 
end