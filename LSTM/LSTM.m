%[XTrain,YTrain] = japaneseVowelsTestData; % load sequence data
%XTrain(1:5)
class_dict=train_generator.class_indices;
labels= train_generator.labels;
file_names= train_generator.filenames;

figure %Visualize the first time series in a plot. Each line corresponds to a feature.
plot(XTrain{1}')
xlabel("Time Step")
title("Training Observation 1")
numFeatures = size(XTrain{1},1);
legend("Feature " + string(1:numFeatures),'Location','northeastoutside')
numObservations = numel(XTrain);
for i=1:numObservations % get the sequence length for each observation
    sequence = XTrain{i};
    sequenceLengths(i) = size(sequence,2);
end
[sequenceLengths,idx] = sort(sequenceLengths); %load data by seq length
XTrain = XTrain(idx); 
YTrain = YTrain(idx);
figure %v view seq lengths in bar chart
bar(sequenceLengths)
ylim([0 30])
xlabel("Sequence")
ylabel("Length")
title("Sorted Data")
miniBatchSize = 27; % reduce the amount of padding

inputSize = 12; %define LSTM layer
numHiddenUnits = 100;
numClasses = 10;
layers = [ ...
    sequenceInputLayer(inputSize)
    bilstmLayer(numHiddenUnits,'OutputMode','last')
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer]
maxEpochs = 100;
miniBatchSize = 27;
options = trainingOptions('adam', ...
    'ExecutionEnvironment','cpu', ...
    'GradientThreshold',1, ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'SequenceLength','longest', ...
    'Shuffle','never', ...
    'Verbose',0, ...
    'Plots','training-progress');
net = trainNetwork(XTrain,YTrain,layers,options); % train the network
[XTest,YTest] = japaneseVowelsTestData; % test the network
XTest(1:3) 
numObservationsTest = numel(XTest); % ensure that the data is organized using mini-batches of similar length
for i=1:numObservationsTest
    sequence = XTest{i};
    sequenceLengthsTest(i) = size(sequence,2);
end
[sequenceLengthsTest,idx] = sort(sequenceLengthsTest);
XTest = XTest(idx);
YTest = YTest(idx);
miniBatchSize = 27; %classify the test data
YPred = classify(net,XTest, ...
    'MiniBatchSize',miniBatchSize, ...
    'SequenceLength','longest');
acc = sum(YPred == YTest)./numel(YTest) %calculate the accuracy
