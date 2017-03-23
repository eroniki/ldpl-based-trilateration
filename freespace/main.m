clc; clear all; close all;

dbm = -25:-1:-45;

x_ref = [5,5];
x_AP = [0,10; 10, 10; 10, 0];
Pr_ref = [-30, -30, -30];
r = zeros(3,1);
measurement = [-80, -75, -80];

for i=1:3
    r(i) = getDistanceFromRef(Pr_ref(i), x_ref, measurement(i));
end

figure(1); plot(x_AP(:,1),x_AP(:,2), 'r*'); hold on;
plot(x_ref(1), x_ref(2), 'b*');
for i=1:numel(r)
    [xunit, yunit] = circle(x_AP(i,1), x_AP(i,2), r(i));
    figure(1); plot(xunit, yunit); grid on;
end
legend('AP', 'Ref', 'Measurement_1', 'Measurement_2', 'Measurement_3');
% hold off;
% legend(cellstr(num2str(dbm')));
% 
% distanceDiff1 = getDistanceDifferenceFromRef(-20, 1,-25);
% distanceDiff2 = getDistanceDifferenceFromRef(-40, 5,-45);
% 
% [xunit1, yunit1] = circle(0,1,distanceDiff1);
% [xunit2, yunit2] = circle(0,5,distanceDiff2);
% [xout,yout] = circcirc(0,1,distanceDiff1,0,5,distanceDiff2);
% figure(2); plot(xunit1, yunit1, 'r', xunit2, yunit2,'b'); grid on;
