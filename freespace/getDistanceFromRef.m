function distance = getDistanceFromRef(p_ref, x_ref, p)
    gamma_inv = 10^((p_ref-p)/20);
    distance = norm(x_ref) * (gamma_inv);
end

