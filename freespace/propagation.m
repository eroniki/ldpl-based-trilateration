clc; clear all; close all;

cd
sensivity_xbee = -101;
getDistance_xbee = @(rssi) 10.^((24-rssi-31.5266)/20);
xbee_distance = getDistance_xbee(dbm_xbee);

maxCoverage_xbee = getDistance_xbee(sensivity_xbee);

dbm_wifi = -20:-1:-30;
sensivity_wifi = -98;
getDistance_wifi = @(rssi) 10.^((17-rssi-40.0460)/20);
wifi_distance = getDistance_wifi(dbm_wifi);
maxCoverage_wifi = getDistance_wifi(sensivity_wifi);
%% Results
figure(1); stem(dbm_xbee, xbee_distance); hold on;
stem(dbm_wifi, wifi_distance);
legend('900 mhz', '2.4ghz');
ylabel('Distance (m)');
xlabel('Received Signal Strength (dBm)');
title('Received Signal Strength vs. Distance');
grid on; grid minor;

figure(2); bar(xbee_distance, dbm_xbee); hold on;
bar(wifi_distance, dbm_wifi);
legend('900 mhz', '2.4ghz');
xlabel('Distance (m)');
ylabel('Received Signal Strength (dBm)');
title('Distance vs. Received Signal Strength');
grid on; grid minor;

disp('Max coverage 900MHz:');
disp(maxCoverage_xbee);

disp('Max coverage 2.4GHz:');
disp(maxCoverage_wifi);