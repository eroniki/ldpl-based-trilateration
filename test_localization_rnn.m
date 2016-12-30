function outputs = test_localization_rnn(net, testingdata)
    outputs = net(testingdata', Xi, Ai);
    outputs = outputs';
end

