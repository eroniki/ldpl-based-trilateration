function outputs = test_localization_rnn(net, testingdata)
    outputs = sim(net, con2seq(testingdata'));
    outputs = cell2mat(outputs);
    outputs = outputs';
end

