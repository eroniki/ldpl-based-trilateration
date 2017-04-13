%% Radial distance estimation with LDPL
error_map = zeros(25,7,8);
error_map_ind = zeros(25,7,8);
for i=1:25
    for j=1:7               
        test_point = [i,j];
        pl_vector = pl(test_point(1), test_point(2), :);
        pl_vector = pl_vector(:);

        d_hat = ldpl(dist, pl_vector, pl_at_center, path_loss_exp, std_measurement);
        d_hat_ind = ldpl(dist, pl_vector, pl_at_center, path_loss_exp_ind, std_measurement);
        d = pdist2(pos_node_m, [gcx(i), gcy(j)]);
        e = localization_error(d, d_hat);
        e_ind = localization_error(d_hat_ind, d);
        error_map(i,j,:) = e;
        error_map_ind(i,j,:) = e_ind;
    end
end