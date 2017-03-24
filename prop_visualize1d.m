close all; clc;
addpath('freespace/');


Pt = [21, 10, 24]; %% WiFi, BT, LoRa
Pt_w = dbm2watts(Pt);
f= [2.4, 2.4, 0.9] * 10^9; %% WiFi, BT, LoRa
node_str = {'Node 0', 'Node 1', 'Node 2', 'Node 3', 'Node 4', 'Node 5', 'Node 6', 'Node 7'};
rf_str = {'WiFi', 'BT', 'LoRa'};
Gt = [0.0056, 0.0025, 0.083];
rf_ind = reshape(1:24, [8, 3])';

lambda = ftolambda(f);
ref_grid = [1,1; 9,1; 17,1; 1,24; 7, 25; 7, 16; 7, 8; 7,0];

pos_n = pos_node *0.9;
x = 1.35:0.9:7.2;
y = 0.45:0.9:22.5;

[xx, yy] = meshgrid(x,y);
grid_centers = [xx(:),yy(:)];

distq = 0.1:0.2:25;

friis =@(Pt, lambda, d, Gt)(Pt+20*log10(lambda)+15.9636-20*log10(d)+20*log10(Gt));



for rf_type = 1:numel(rf_str)
    for ap=1:8
        d = pdist2(grid_centers, pos_node(ap,:));
        [d_sorted, ind] = sort(d,'Ascend');
        gx = 1:8;
        gy = 1:25;
        rf_ind(rf_type, ap)
        m = propagation_maps(:, :, rf_ind(rf_type, ap));
        m = m(:);
        m = m(ind);

        m(m==0) = NaN;

        mea = nanmean(m);
        
        figure;
%         subplot(2,4,ap);
        
        plot(d_sorted, m, 'b-.', ...
            d_sorted, smooth(m), 'r', ...            
            d_sorted, repmat(mea, size(d_sorted)), 'k');
        
        hold on;
        
        Pr_t = friis(Pt(rf_type), lambda(rf_type), distq, Gt(rf_type));
        mea = mean(Pr_t);     
        plot(distq, Pr_t, 'b', distq, repmat(mea, size(distq)), 'k');
        grid on; grid minor;
        xlabel('Distance [m]');
        ylabel('RSS [dBm]');
        title([rf_str{rf_type},' ', node_str{ap}]);
    end
end

if save_figures
    n = get(gcf,'Number');

    for i=1:n
        saveas(i, ['output_prop/', num2str(i), '.png'],'png');
    end
end


% 
% dfromFriis = @(Pr, Pt, f, K) 10.^((Pt-Pr+K+20*log10(ftolambda(f))-15.9636)/20);
% Friis = @(d, Pt, f, K) (Pt+K+20*log10(ftolambda(f))-15.9636-20*log10(d));
% 
% Pr_theory = Friis(dist, Pt, 9*10^8, 22/2);
% 
% 
% 
% x = 1:7;
% y = 1:25;
% [xx,yy] = meshgrid(x,y);
% grid_cells = [xx(:), yy(:)];

% 
% for i=1:8
%     d = pdist2(grid_cells,pos_node(i,:));
%     [d_sorted, ind] = sort(d,'Ascend');
%     mean_measurements = propagation_maps(:,:,i+16);
%     mean_measurements = mean_measurements(:);
%     mean_measurements = mean_measurements(ind);
%     qq = find(mean_measurements == 0);
%     mean_measurements(qq) = [];
%     d_sorted(qq) = [];
%     figure; plot(d_sorted*0.9, mean_measurements); hold on;
%     plot(dist, Pr_theory);
%     grid on;
%     grid minor;
%     xlabel('Distance [m]');
%     ylabel('RSSI [dBm]');
% 
% end

