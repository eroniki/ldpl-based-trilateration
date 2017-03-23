clear all, close all, clc;
% profile on
%% Map files 
% map.origin.x = -35.681483;
% map.origin.y = -9.290859;
% map.res = 0.05;
% map.location = 'map/goodwin2dfloor.pgm';
% map.map = imread(map.location);
% map.map = flipdim(map.map ,1);
% map.map = map.map';
%% Parse the data
db = importfile_no_amcl('measurements/1478808114.txt', 1, 3236);
pos_file = importfile_positions('measurements/positions.txt', 2, 129);


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

% %% Visualization
% figure(1); imagesc(featureSpace); title('Raw RSSI Measurements'); xlabel('RSSI Readings n_{AP} = 50'); ylabel('Positions n_{pos} = 82'); colorbar();
% figure(2); imagesc(featureSpace_normalized);  title('Centered RSSI Measurements'); xlabel('RSSI Readings n_{AP} = 50'); ylabel('Positions n_{pos} = 82'); colormap(jet); colorbar;
% figure(3); imshow(featureSpace_normalized_image);
% % saveas(1,'outputs/raw.jpg');
% % saveas(2, 'outputs/centered.jpg');
% % imwrite(featureSpace_normalized_image, 'outputs/featureSpace.tif');
% goodwin
% %% Analysis of the APs
% % Find 10 the strongest APs
% [Y, I] = sort(sum(featureSpace));
% for(jj=1:10)
%     strongMAC = uniqueAP(I(jj));
%     vars = {'x', 'y', 'rssi', 'ssid', 'mac'};
%     [i,~]=ind2sub(size(db.mac), strmatch(strongMAC, db.mac, 'exact'));
%     spliceddata = db(i, vars);
%     [uniqueRows,ia,ic] = unique(spliceddata);
%     x = table2array(uniqueRows(:,'x'));
%     y = table2array(uniqueRows(:,'y'));
%     rssi = table2array(uniqueRows(:, 'rssi'));
% %     figure(jj); scatter3(x,y,rssi,repmat(15, 1, numel(x)), rssi, 'filled'); title([spliceddata.ssid{1}, ' ' , spliceddata.mac{1}]); grid on; colorbar; hold on;
% %     view(120,34);
%     px.x = round((x-map.origin.x)/map.res);
%     py.y = round((y-map.origin.y)/map.res);
%     figure(jj); scatter3(py.y, px.x, -rssi, repmat(15, 1, numel(x)), rssi, 'filled');
%     grid on; colorbar; title([spliceddata.ssid{1}, ' ' , spliceddata.mac{1}]);  hold on;
%     ax = gca();
%     imshow(map.map, 'Parent', ax); title([spliceddata.ssid{1}, ' ' , spliceddata.mac{1}]); 
%     
%     view(32,48);
% %     ax = gca();
% end

%% Split Data into Training, Test and Validation Sets

db__ = featureSpace_normalized;
pos__ = pos;
[rows, ~] = size(featureSpace_normalized);

training_id = randsample(1:rows, 90);

trainingset = db__(training_id,:);
pos_train = pos__(training_id,:);
db__(training_id,:) = [];
pos__(training_id,:) = [];
[rows, ~] = size(db__);
test_id = randsample(1:rows, 35);
testingset = db__(test_id,:);
pos_testing = pos__(test_id,:);

% training_id = randsample(1:rows, 60);
% 
% trainingset = db__(training_id,:);
% pos_train = pos__(training_id,:);
% [rows, ~] = size(db__);
% test_id = randsample(1:rows, 82);
% testingset = db__(test_id,:);
% pos_testing = pos__(test_id,:);


% db__(test_id,:) = [];
% pos__(test_id,:) = [];
% validset = db__;
% pos_valid= pos__;




%% Support Vector Regression
% m=3;
% b=5;
% x=-10:0.01:10;
% y = m*x+b;
% svm_object_x = svmtrain(y,x,'-s 3 -t 3 -n 0.5 -c 1');
% svmpredict(y, 0, svm_object_x)
svm_object_x = svmtrain(pos_train(:,1), trainingset, '-s 3 -t 3 -n 0.5 -c 1');
pos_predict_x =svmpredict(pos_testing(:,1), testingset, svm_object_x);
error_x = pos_predict_x-pos_testing(:,1);

svm_object_y = svmtrain(pos_train(:,2), trainingset, '-s 3 -t 3 -n 0.5 -c 1');
pos_predict_y =svmpredict(pos_testing(:,2), testingset,svm_object_y);
error_y = pos_predict_y-pos_testing(:,2);
[pos_testing, pos_predict_x, pos_predict_y]
T = table(pos_testing(:,1), pos_testing(:,2), pos_predict_x, pos_predict_y, ...
    error_x, error_y, ...
    'VariableNames',{'X' 'Y' ... 
    'Predicted_X' 'Predicted_Y', 'Error_X', 'Error_Y'});
e_x_bar = mean(sqrt(error_x.^2))
e_y_bar = mean(sqrt(error_y.^2))
mse = mean(sqrt(error_x.^2+error_y.^2))
%% Visualize the results

for jj=1:length(pos_testing)
    figure(jj); plot(pos_testing(jj,1), pos_testing(jj,2), 'b*', pos_predict_x(jj), pos_predict_y(jj), 'r*'); grid on; title(['Error: ', num2str(sqrt(error_x(jj)^2+error_y(jj)^2)), ' m']);
    legend('Testing Location', 'Predicted Location');
    drawnow;    
end

%% Save Results 
h =  findobj('type','figure');
n = length(h);

for jj=1:n
    saveas(jj, ['outputs/bcc/', num2str(jj), '.jpg']);
end

% plot(py(jj), px(jj), 'LineStyle', 'none', 'Marker', 'd', 'MarkerFaceColor', colors(jj,:), 'MarkerEdgeColor', 'none'); hold on;
% plot(p_hat_y(jj), p_hat_x(jj), 'LineStyle', 'none', 'Marker', 'd', 'MarkerFaceColor', colors(jj,:), 'MarkerEdgeColor', 'none');
 
%% project vectors to higher level spaces 
% % kernel function
% kern = @(x,a,c,d) ((a*x'*x+c).^d);
% kern_rbf = @(x,a) (exp(x'*x./a));
% kern_conv = @(x) (exp(conv2(x,x')))
% % project observations to higher dimensional spaces
% projectedFeatures = zeros(nTimeStamp, nAP, nAP);
% posList = zeros(nTimeStamp, 2);
% px.list = [];
% for i=1:nTimeStamp
%     projectedFeatures(i,:,:) = kern_conv(featureSpace_normalized(i,:));
% %     projectedFeatures(i,:,:) = kern(featureSpace_normalized(i,:),1,1000,2);
% %     projectedFeatures(i,:,:) = kern_rbf(featureSpace_normalized(i,:),5);
%     rows = db.timestamp == uniqueTimeStamps(i);
%     cols = {'x','y'};
%     pos = db(rows,cols);
%     pos = pos(1,:);
%     pos = table2array(pos);
%     posList(i,:) = pos;
%     px.x = round((pos(1)-map.origin.x)/map.res);
%     px.y = round((pos(2)-map.origin.y)/map.res);
%     px.list = [px.list; px.y, px.x];
%     center = [px.y, px.x];
%     sup = reshape(projectedFeatures(i,:,:), [nAP, nAP]);
%     figure(4); imagesc(sup); title(['Projected Measurements (Location=', num2str(i), 'X=', num2str(pos(1)) , 'Y=', num2str(pos(2)), ')']); colorbar();
% %     figure(5); imshow(map.map); hold on;
% %     viscircles(px.list,repmat(10,i,1));
% %     imwrite(sup,['location', num2str(i), '.jpg'])
% %     saveas(4, ['outputs/features_rbf', num2str(i), '.jpg']);
% %     saveas(5, ['outputs/location', num2str(i), '.jpg']);
%     pause;
% end
% 
% profile off
% profsave(profile('info'),'poly')

%% 
% for i=1:uniqueTimeStamps
% row
% end

% sup = [projectedFeatures, db.x, db.y];