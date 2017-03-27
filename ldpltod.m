function [ d ] = ldpltod(n)
    n_data = evalin('base', 'n_data');
    pl_d = evalin('base', 'pl_d');
    pl_d0 = evalin('base', 'pl_at_center');
    d0 = evalin('base', 'dist');
    d = 0;
    for ap=1:8
        for i=1:n_data
            d_ = d0(ap)*10.^((pl_d(i,ap)-pl_d0(ap))./(10*n));
            d = d + d_;
        end
    end
end

