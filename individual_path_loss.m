%% Ind. Path Loss Exponent
n_test = 1:0.2:8;
e_bar = [];
n_tested = [];
for n=n_test
    n_tested = [n_tested, n];
    loss_map = zeros(25,7,8);
    e = [];
    for i=1:25
        for j=1:7               
            test_point = [i,j];
            pl_vector = pl(test_point(1), test_point(2), :);
            pl_vector = pl_vector(:);
            d_hat = ldpl(dist, pl_vector, pl_at_center, n, std_lora);
            d = pdist2(pos_node, test_point);
            ll = loss_localization(d,d_hat);
            e =  [e, ll];
%             loss_map(i,j,:) = e;
        end
    end
    ll = mean(e, 2, 'omitnan');
    rl = loss_regularization(n);
    buff = loss_total(ll, rl, 0.9);
    e_bar = [e_bar, buff];
    if(verbose)
        disp(['n: ', num2str(n), ' e_bar: ', num2str(buff'), num2str(size(e_bar))]);
    end
    figure(1);
    plot(n_tested, e_bar(1,:), n_tested, e_bar(2,:), n_tested, e_bar(3,:), ...
        n_tested, e_bar(4, :), n_tested, e_bar(5, :), n_tested, e_bar(6,:), ...
        n_tested, e_bar(7, :), n_tested, e_bar(8, :)); grid on; grid minor;
    legend('0', '1', '2', '3', '4', '5', '6', '7');
    xlim([min(n_test), max(n_test)]);
    ylim([0, 1000]);
    xlabel('Path Loss Exponent (n)');
    ylabel('Loss');
    drawnow;
end
if(save_figures)
    saveas(1, 'output/path_loss_exponent_ind.png','png');
end
close(1);
[min_val, min_id] = min(e_bar,[],2);
path_loss_exp_ind = n_test(min_id)';
disp('Path Loss Exponent:');
disp(vpa(path_loss_exp_ind, 8));