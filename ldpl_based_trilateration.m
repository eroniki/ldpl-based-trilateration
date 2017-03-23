% This script calls necessary scripts to run a localization test on Hancock
% dataset. The localization is based on LDPL model and least squares
% estimation.
%% Clear up and close all existing figures
clear all; close all; clc;
%% Add path in order to use functions resides in other packages
addpath('/home/murat/wifi_localization/wifi_regression','-frozen');
verbose = 1;
save_figures = 1;
% Function Handles
loss_regularization = @(n)(n)^1.2;
loss_localization = @(d, d_hat) (d-d_hat).^2;
loss_total = @(l1, l2, w) w*l1 + (1-w)*l2;
localization_error = @(d, d_hat) sqrt(sum((d - d_hat).^2,2));
z_score = @(x) (x-mean(x,'omitnan')) ./ std(x, 'omitnan');
circle = @(x,y,r,ang) deal(x+r*cos(ang), y+r*sin(ang));
%% Preprocess the data
preprocess
%%
center = [4, 13];
pt = 24;
lora_propagation = propagation_maps(:,:,17:24);
lora_propagation(~lora_propagation) = NaN;
pr_at_center = propagation_maps(13,4,17:24);
pl = zeros(25,7,8);
pr_at_center = pr_at_center(:);
dist = pdist2(pos_node,center);
pl = pt - lora_propagation;
pl_at_center = pt - pr_at_center;
%% Path Loss Exponent Estimation
% Collective
collective_path_loss
% Individual
individual_path_loss
%% Estimate Radial Distances
estimate_radial_distance
%% Trilateration
least_square_trilateration
%% Normality Check for the normalized error
[h, p] = kstest(z_score(error_map(:)))
[h, p] = kstest(z_score(error_map_ind(:)))
%% Visualize and save the resulting figures
visualize
