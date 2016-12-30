function outputs = test_localization_rnn(net, testingdata)
    outputs = net(testingdata');
    outputs = outputs';
end

