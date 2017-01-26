function outputs = test_localization_nn(net, testingdata)
    outputs = sim(net,testingdata');
    outputs = outputs';
end

