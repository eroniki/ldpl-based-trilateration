close all; clc;
addpath('~/Desktop/freespace/');
Pt = 24;
dist = 0:0.02:23.365358974;

dfromFriis = @(Pr, Pt, f, K) 10.^((Pt-Pr+K+20*log10(ftolambda(f))-15.9636)/20);
Friis = @(d, Pt, f, K) (Pt+K+20*log10(ftolambda(f))-15.9636-20*log10(d));

Pr_theory = Friis(dist, Pt, 9*10^8, 22/2);



x = 1:7;
y = 1:25;
[xx,yy] = meshgrid(x,y);
grid_cells = [xx(:), yy(:)];


for i=1:8
    d = pdist2(grid_cells,pos_node(i,:));
    [d_sorted, ind] = sort(d,'Ascend');
    mean_measurements = propagation_maps(:,:,i+16);
    mean_measurements = mean_measurements(:);
    mean_measurements = mean_measurements(ind);
    qq = find(mean_measurements == 0);
    mean_measurements(qq) = [];
    d_sorted(qq) = [];
    figure; plot(d_sorted*0.9, mean_measurements); hold on;
    plot(dist, Pr_theory);
    grid on;
    grid minor;
    xlabel('Distance [m]');
    ylabel('RSSI [dBm]');

end

