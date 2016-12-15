load('datasets/UCM/Gorkem_data.mat');
[nPos,~] = size(data);
%%
T = table();
% for i=1:nPos
%     rows = db.id == i;
%     [nAP, ~] = size(db(rows,:));
%     T = [T; db(rows, {'timestamp', 'mac', 'rssi', 'ssid'}), repmat(pos_file(i,2:3),[nAP,1])];
% end
%%
for i=1:nPos
     nAP = length(data(i).ap);
     for j=1:nAP
        timestamp = data(i).time;
        mac = data(i).ap(j).mac;
        rssi = data(i).ap(j).signal;
        ssid = data(i).ap(j).essid;
        x = data(i).x;
        y = data(i).y;
%         whos timestamp
%         whos mac
%         whos rssi
%         whos ssid
%         whos x
%         whos y
        tmp = table(timestamp, {mac}, rssi, {ssid}, x, y);
        T = [T; tmp];
     end
%     tmp_x = data(i).x;
%     tmp_y = data(i).y;
%     tmp_timestamp = data(i).time;
%     tmp_meas
    
end