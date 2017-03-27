%% Collective Path Loss Exponent Estimation
n_test = 15:0.02:18;
e_bar = [];
n_tested = [];
for n=n_test
    n_tested = [n_tested, n];
    loss_map = zeros(25,7,8,401);
    for i=1:25
        for j=1:7               
            test_point = [i,j];
            pl_vector = pl(test_point(1), test_point(2), :);
            pl_vector = pl_vector(:);
            d_hat = ldpl(dist, pl_vector, pl_at_center, n, std_measurement);
            d = pdist2(pos_node_m, [gcx(i), gcy(j)]);
            ll = loss_localization(d,d_hat);
            loss_map(i,j,:,:) = ll;
        end
    end
    rl = loss_regularization(n);
    ll = mean(loss_map(:), 'omitnan');
    buff = loss_total(ll, rl, 1);
    e_bar = [e_bar, buff];
    if(verbose)
        disp(['n: ', num2str(n), ' ll: ', num2str(ll), ' rl:', num2str(rl), ' sum: ', num2str(buff)]);
    end
    figure(1);
    plot(n_tested, e_bar); grid on; grid minor;
    xlim([min(n_test), max(n_test)]);
    ylim([0, 1000]);
    xlabel('Path Loss Exponent (n)');
    ylabel('Loss');
    drawnow;
end
if(save_figures)
    saveas(1, 'output/path_loss_exponent_joint.png','png');
end
close(1);
[min_val, min_id] = min(e_bar);
path_loss_exp = n_test(min_id);
disp('Path Loss Exponent:');
disp(vpa(path_loss_exp, 8));