function [mse_x, mse_y, mse] = residual_analysis(prediction, groundTruth)
    error = prediction - groundTruth;
    mse_x = mean(sqrt(error(:,1).^2));
    mse_y = mean(sqrt(error(:,2).^2));
    mse = mean(sqrt(error(:,1).^2+error(:,2).^2));
%     mses= ;
end

