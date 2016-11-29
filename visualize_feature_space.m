[n_pos, n_ap] = size(featureSpace);

figure(1); imagesc(featureSpace);
title('Raw RSSI Measurements');
xlabel(['RSSI Readings (n_{AP} = ', num2str(n_ap), ')']);
ylabel(['Positions (n_{pos} =  ', num2str(n_pos), ')']);
colorbar();

figure(2); imagesc(featureSpace_normalized);
title('Centered RSSI Measurements');
xlabel(['RSSI Readings (n_{AP} = ', num2str(n_ap), ')']);
ylabel(['Positions (n_{pos} =  ', num2str(n_pos), ')']);
colormap(jet);
colorbar;

saveas(1,'outputs/raw.jpg');
saveas(2,'outputs/centered.jpg');