%%%directions for use: 



imds = imageDatastore('MerchData', ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames'); 
[imdsTrain,imdsValidation] = splitEachLabel(imds,0.7);

% visualize a random image
I = readimage(imdsTrain,randi(numel(imdsTrain.Files)));
imshow(I);

net = googlenet;

inputSize = net.Layers(1).InputSize;
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain);
augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);

% pull a test sample
test_sample_table = augimdsTrain.readByIndex(1);
test_sample = test_sample_table.input{1};

% use GoogLeNet to classify image
label = classify(net,test_sample)
imshow(test_sample);


lgraph = layerGraph(net);

numClasses = numel(categories(imdsTrain.Labels));

newLearnableLayer = fullyConnectedLayer(numClasses, ...
        'Name','new_fc', ...
        'WeightLearnRateFactor',8, ...
        'BiasLearnRateFactor',8);

lgraph = replaceLayer(lgraph,'loss3-classifier',newLearnableLayer);
newClassLayer = classificationLayer('Name','new_classoutput');
lgraph = replaceLayer(lgraph,'output',newClassLayer);

options = trainingOptions('sgdm', ...
    'MiniBatchSize',10, ...
    'MaxEpochs',6, ...
    'InitialLearnRate',3e-4, ...
    'Plots','training-progress');

newNet = trainNetwork(augimdsTrain,lgraph,options);


[YPred,probs] = classify(newNet,augimdsValidation);
accuracy = mean(YPred == imdsValidation.Labels)

% And view individual images

idx = randperm(numel(imdsValidation.Files),10);
figure
for i = 1:10
    subplot(5,2,i)
    I = readimage(imdsValidation,idx(i));
    imshow(I)
    label = YPred(idx(i));
    title(string(label) + " ,  " + num2str(100*max(probs(idx(i),:)),9) + "%");
end

