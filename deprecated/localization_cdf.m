function [cdf, countsError, binsError] = localization_cdf(error, nBins)
    [countsError, binsError] = hist(error, nBins);
    cdf = cumsum(countsError) / sum(countsError);
end

