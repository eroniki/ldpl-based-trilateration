function p_rx = friis(p_tx, gt, gr, lambda, d, l)
    p_rx = p_tx * gt * gr * lambda^2 / (157.9137 * d^2 *l);
    % 157.9137 = (4pi)^2
end

