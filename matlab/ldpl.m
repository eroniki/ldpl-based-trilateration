function [d_hat] = ldpl(d0, pl_d, pl_d0, n, std)
%     rnd_= normrnd(zeros([numel(pl_d), 1]), std');
    xx = -4:0.02:4;
    rnd_ = zeros(numel(pl_d0), numel(xx));
    for i=1:numel(pl_d)
        rnd_(i,:) = normpdf(xx);
    end
%     rnd_ = 0;
%     d_hat = 10.^((pl_d - pl_d0 - rnd_)./(10*n)).*d0;
    qq = bsxfun(@plus, -rnd_, pl_d-pl_d0);
    pp = bsxfun(@rdivide, qq, 10*n);
    d_hat = bsxfun(@times, 10.^pp, d0);
end

