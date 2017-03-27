% This script calls necessary scripts to run a localization test on Hancock
% dataset. The localization is based on LDPL model and least squares
% estimation.
%% Clear up and close all existing figures
clear all; close all; clc;
% Set verbosity level and the flag for saving the figures
verbose = 1;
save_figures = 1;
% Function Handles
loss_regularization = @(n)(n)^1.2;
loss_localization = @(d, d_hat) bsxfun(@minus,d_hat,d).^2;
loss_total = @(l1, l2, w) w*l1 + (1-w)*l2;
localization_error = @(d, d_hat) sqrt(sum(bsxfun(@minus, d_hat,d).^2,2));
z_score = @(x) (x-mean(x,'omitnan')) ./ std(x, 'omitnan');
circle = @(x,y,r,ang) deal(x+r*cos(ang), y+r*sin(ang));
%% Preprocess the data
preprocess
%%
% Transmitted power
pt = 24;
% Section the propagation map to acquire relevant data such that
% the measurements will contain one source of information (wifi, bt, or
% lora)
% lora = 17:24
% bt   =  9:16
% wifi =  1:8
measurement_set = 17:24;
lora_propagation = propagation_maps(:,:, measurement_set);
% No measurements = NaN
lora_propagation(~lora_propagation) = NaN;
% the label of the reference grid
center = [13, 4];
[cx_m, cy_m] = grid_label_to_grid_center(center(1), center(2));
pr_at_center = propagation_maps(center(1),center(2), measurement_set);
pr_at_center = pr_at_center(:);
dist = pdist2(pos_node_m, [cx_m, cy_m]);
pl = pt - lora_propagation;
pl_at_center = pt - pr_at_center;
std_measurement = std(data_lora);
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
