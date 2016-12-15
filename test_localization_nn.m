function outputs = test_localization_nn(net, testingdata)
    outputs = net(testingdata');
    outputs = outputs';
end

