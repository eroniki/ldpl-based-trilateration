function [errMean, errMedian] = test(RSS, XY, testRSS, testXY, testOrient)
% For tests without orientation information use matrices: calibAvgRSS, calibAvgXY
% Examples:
% [errMean, errMedian] = test(calibAvgRSS, calibAvgXY, testRSS, testXY)
% [errMean, errMedian] = test(calibAvgRSS(:,ind5AP_All), calibAvgXY, testRSS(:,ind5AP_All), testXY)
%
% For tests with orientation information use matrices: calibRSS, calibXY
% And provide orientation information for test data.
% Examples:
% [errMean, errMedian] = test(calibRSS, calibXY, testRSS, testXY, testOrient)
% [errMean, errMedian] = test(calibRSS(:,ind5AP_All), calibXY, testRSS(:,ind5AP_All), testXY, testOrient)

% Copyright (c) 2009 Gints Jekabsons
%
% Permission is hereby granted, free of charge, to any person obtaining a
% copy of this software and associated documentation files (the "Software"),
% to deal in the Software without restriction, including without limitation
% the rights to use, copy, modify, merge, publish, distribute, sublicense,
% and/or sell copies of the Software, and to permit persons to whom the
% Software is furnished to do so, subject to the following conditions:
%
% The above copyright notice and this permission notice shall be included
% in all copies or substantial portions of the Software.
%
% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
% OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
% MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
% NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
% DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
% OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
% USE OR OTHER DEALINGS IN THE SOFTWARE.

if nargin < 4
    error('Too few input arguments.');
elseif nargin == 4
    testOrient = [];
end

[n, d] = size(RSS);
[nXY, dXY] = size(XY);
if (n < 1) || (d < 1) || (nXY ~= n) || (dXY ~= 2)
    error('Wrong input data sizes.');
end

[tn, td] = size(testRSS);
[tnXY, tdXY] = size(testXY);
if (tn < 1) || (td ~= d) || (tnXY ~= tn) || (tdXY ~= 2)
    error('Wrong input data sizes.');
end

err = zeros(tn,1);
for i = 1 : tn
    if isempty(testOrient)
        predictedXY = knn(RSS, XY, testRSS(i,:), 2, true);
    else
        ind = testOrient(i,1):4:n;
        predictedXY = knn(RSS(ind,:), XY(ind,:), testRSS(i,:), 2, true);
    end
    err(i) = norm(predictedXY - testXY(i,:));
end
errMean = mean(err);
errMedian = median(err);
return

function testXY = knn(RSS, XY, testRSS, k, weighting)
testXY = zeros(size(testRSS,1),2);
for i = 1 : size(testRSS,1)
    dists = sqrt(sum(bsxfun(@minus, RSS, testRSS(i,:)).^2, 2));
    [distClosest, ind] = sort(dists);
    indClosest = ind(1:k);
    if weighting
        if distClosest(1) <= 0
            testXY(i,1) = XY(indClosest(1),1);
            testXY(i,2) = XY(indClosest(1),2);
        else
            w = 1 ./ distClosest(1:k);
            wsum = sum(w);
            testXY(i,1) = sum(XY(indClosest,1) .* w) / wsum;
            testXY(i,2) = sum(XY(indClosest,2) .* w) / wsum;
        end
    else
        testXY(i,:) = mean(XY(indClosest,:));
    end
end
return
