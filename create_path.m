function [path] = create_path(list, ntest)
    idx = [1];

    path(1,:) = list(1,:);
    tmp = list;
    for i=2:ntest
        tmp(idx,:) = NaN;
        [nRows, ~] = size(tmp);
        rep = repmat(list(idx(end),:), nRows,1);
        diff = rep-tmp;
        lastPoint = list(idx(end),:);
        dist = diff(:,1).^2 + diff(:,2).^2;
        [minDist, minID] = min(dist);
        idx = [idx, minID];
        tmp(minID, :) = NaN;
        path(i,:) = list(minID,:);
    end
end

