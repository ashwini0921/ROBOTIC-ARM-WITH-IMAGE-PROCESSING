# ROBOTIC-ARM-WITH-IMAGE-PROCESSING
Designed robotic Arm which can separate fruits using Deep Learning

## ABSTRACT

The project involves the development of a 6-degree-of-freedom (DOF) robotic arm capable of fruit separation through image processing techniques. A classification model is designed using Deep Neural Networks (DNN) combine with Convolutional Neural Network to accurately identify and categorize fruits based on their size, shape, and color attributes. This innovative approach combines robotics and artificial intelligence to enhance fruit sorting processes, contributing to efficiency and precision in agricultural and food industries. The integration of image processing and machine learning underscores the potential for automated fruit handling, optimizing quality control and streamlining production workflows.

## ROBOT SPECIFICATION
Drive System: Electric motor(servo motor)
Programming Software: MATLAB, ARDUINO IDE
Degree of Freedom: 5
Rotational Joints: 5
Gripper: Mechanical
Speed of Movement: Adjustable(300 degree/s to 30 degree/s)

## HARDWARE COMPONENTS USED
### Robotic Arm:
A robotic arm is a sort of arm which may be mechanical, and can be programmed with
capabilities almost equivalent to an actual arm. The robotic arm can be an aggregate of a
component or a piece of a more advanced robot.

### Servo Motors
Actuator whose motion can be controlled precisely. It is a simple electric motor with a
closed loop feedback control system for specific angular rotation.

### Arduino UNO microcontroller
A microcontroller board based on the ATmega328P is a compact integrated circuit
designed to govern a specific operation in an embedded system. It includes a processor, memory
and input/output (I/O) peripherals on a single chip.

### Ultrasonic Sensors
Ultrasonic Sensors generate ultrasonic sound waves which travel with a speed of sound in
the medium. Since it is very less absorbed by surroundings, it strikes the object and reflects. The
reflected waves are sensed by ultrasonic sensors. The time interval between to and fro motion of
ultrasonic wave is used to get distance of the object. Ultrasonic sensor detect presence and coordinates of object.
Distance of object is 340 *(t/2) m where t in seconds.

### Frontech Webcam
It has 3 MP Image Resolution and USB Interface to connect it with PC and capture image of the object. It is a CMOS based camera with 640 x 480 Pixels and maximum frame rate of 30 fps.

## SOFTWARE COMPONENT USED:

### MATLAB
It contains an image processing toolbox, a deep learning toolbox which helps in making image processing more programming compatible. MATLAB is equipped with toolbox to support Arduino programming and external camera integration to the system.

### Arduino IDE
Software to test hardware component of Robotic arm using Arduino Code before implementing in MATLAB.

## MATLAB CODE FOR IMAGE CLASSIFICATION

#### SAMPLE IMAGES OF APPLES
![apple6](https://github.com/ashwini0921/ROBOTIC-ARM-WITH-IMAGE-PROCESSING/assets/111654188/c4416433-3973-43a2-8fbe-f54f3e76bfdf)
#### SAMPLE IMAGES OF ORANGES
![images11](https://github.com/ashwini0921/ROBOTIC-ARM-WITH-IMAGE-PROCESSING/assets/111654188/470edbea-3732-4301-88f9-31c5be4a9cd7)
#### SAMPLE IMAGES OF BANANA
![images8](https://github.com/ashwini0921/ROBOTIC-ARM-WITH-IMAGE-PROCESSING/assets/111654188/db59c8b2-6a33-4342-bbde-bfa4862e6d4d)

### Building the Deep Neural Network Model using GoogLeNet:

#### GoogLeNet Architecture:
![image](https://github.com/ashwini0921/ROBOTIC-ARM-WITH-IMAGE-PROCESSING/assets/111654188/8b9c471d-e02a-4a38-ac71-087524c3ee61)

#### MATLAB code for GoogLeNet Deep Learning Model for fruit Classification:
```
clear
close all
clc
imds = imageDatastore("Train_data", 'LabelSource', 'foldernames', 'IncludeSubfolders',true);
[imdsTrain,imdsValidation] = splitEachLabel(imds,0.7);
net=googlenet;
analyzeNetwork(net);
inputSize = net.Layers(1).InputSize;
lgraph = layerGraph(net);
lgraph = removeLayers(lgraph, {'loss3-classifier','prob','output'});
numClasses = numel(categories(imdsTrain.Labels));
newLayers = [
    fullyConnectedLayer(numClasses,'Name','fc','WeightLearnRateFactor',10,'BiasLearnRateFactor',10)
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classoutput')];
lgraph = addLayers(lgraph,newLayers);
%figure('Units','normalized','Position',[0.3 0.3 0.4 0.4]);
%plot(lgraph)
%ylim([0,10])
lgraph = connectLayers(lgraph,'pool5-drop_7x7_s1','fc');

layers = lgraph.Layers;
connections = lgraph.Connections;

layers(1:10) = freezeWeights(layers(1:10));
lgraph = createLgraphUsingConnections(layers,connections);

pixelRange = [-30 30];
scaleRange = [0.9 1.1];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange, ...
    'RandXScale',scaleRange, ...
    'RandYScale',scaleRange);
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
    'DataAugmentation',imageAugmenter,'ColorPreprocessing', 'gray2rgb');

augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation,'ColorPreprocessing', 'gray2rgb');
miniBatchSize = 10;

valFrequency = floor(numel(augimdsTrain.Files)/miniBatchSize);
options = trainingOptions('sgdm', ...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',6, ...
    'InitialLearnRate',3e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',valFrequency, ...
    'Verbose',false, ...
    'Plots','training-progress');

net = trainNetwork(augimdsTrain,lgraph,options);

[YPred,probs] = classify(net,augimdsValidation);
accuracy = mean(YPred == imdsValidation.Labels)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function layers = freezeWeights(layers)
for ii = 1:size(layers,1)
    props = properties(layers(ii));
    for p = 1:numel(props)
        propName = props{p};
        if ~isempty(regexp(propName, 'LearnRateFactor$', 'once'))
            layers(ii).(propName) = 0;
        end
    end
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function lgraph = createLgraphUsingConnections(layers,connections)

lgraph = layerGraph();
for i = 1:numel(layers)
    lgraph = addLayers(lgraph,layers(i));
end

for c = 1:size(connections,1)
    lgraph = connectLayers(lgraph,connections.Source{c},connections.Destination{c});
end

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
```
#### MATLAB output for above fruit classification based model

![image](https://github.com/ashwini0921/ROBOTIC-ARM-WITH-IMAGE-PROCESSING/assets/111654188/cf7e929c-b920-4bcb-8985-11b77e2e228a)

#### Conclusion from above output

After 6 epochs the validation accuracy acheived after 6 epochs is 94.12%. The validation accuracy curve closely follows the training accuracy curve hence it can be concluded that there is less overfitting.


### Building the Deep Neural Network Model using SqueezeNet:

#### SqueezNet Architecture:

![image](https://github.com/ashwini0921/ROBOTIC-ARM-WITH-IMAGE-PROCESSING/assets/111654188/7b99573a-4d30-4237-b981-38df209222c7)
![image](https://github.com/ashwini0921/ROBOTIC-ARM-WITH-IMAGE-PROCESSING/assets/111654188/31a3bdb9-4ec9-4b05-b6df-b4c198b5257d)
![image](https://github.com/ashwini0921/ROBOTIC-ARM-WITH-IMAGE-PROCESSING/assets/111654188/dddd9074-0206-45f8-a510-1486a66cfc09)

#### MATLAB code for SqueezeNet Deep Learning Model for fruit Classification:

```
clear
close all
clc
imds = imageDatastore("Train_data", 'LabelSource', 'foldernames', 'IncludeSubfolders',true);
[imdsTrain,imdsValidation] = splitEachLabel(imds,0.7);
net = squeezenet;
analyzeNetwork(net);
inputSize = net.Layers(1).InputSize;
lgraph = layerGraph(net);
[learnableLayer,classLayer] = findLayersToReplace(lgraph); 
numClasses = numel(categories(imdsTrain.Labels));
newConvLayer =  convolution2dLayer([1, 1],numClasses,'WeightLearnRateFactor',10,'BiasLearnRateFactor',10,"Name",'new_conv');
lgraph = replaceLayer(lgraph,'conv10',newConvLayer);
newClassificatonLayer = classificationLayer('Name','new_classoutput');
lgraph = replaceLayer(lgraph,'ClassificationLayer_predictions',newClassificatonLayer);

pixelRange = [-30 30];
scaleRange = [0.9 1.1];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange, ...
    'RandXScale',scaleRange, ...
    'RandYScale',scaleRange);
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
    'DataAugmentation',imageAugmenter,'ColorPreprocessing', 'gray2rgb');

augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation,'ColorPreprocessing', 'gray2rgb');
miniBatchSize = 10;

valFrequency = floor(numel(augimdsTrain.Files)/miniBatchSize);
options = trainingOptions('sgdm', ...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',6, ...
    'InitialLearnRate',3e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',valFrequency, ...
    'Verbose',false, ...
    'Plots','training-progress');

net = trainNetwork(augimdsTrain,lgraph,options);

[YPred,probs] = classify(net,augimdsValidation);
accuracy = mean(YPred == imdsValidation.Labels)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [learnableLayer,classLayer] = findLayersToReplace(lgraph)
% Copyright 2018-2020 The MathWorks, Inc.
%
% Function Description
% findLayersToReplace(lgraph) finds the single classification layer and the
% preceding learnable (fully connected or convolutional) layer of the layer
% graph lgraph.

if ~isa(lgraph,'nnet.cnn.LayerGraph')
    error('Argument must be a LayerGraph object.')
end

% Get source, destination, and layer names.
src = string(lgraph.Connections.Source);
dst = string(lgraph.Connections.Destination);
layerNames = string({lgraph.Layers.Name}');

% Find the classification layer. The layer graph must have a single
% classification layer.
isClassificationLayer = arrayfun(@(l) ...
    (isa(l,'nnet.cnn.layer.ClassificationOutputLayer')|isa(l,'nnet.layer.ClassificationLayer')), ...
    lgraph.Layers);

if sum(isClassificationLayer) ~= 1
    error('Layer graph must have a single classification layer.')
end
classLayer = lgraph.Layers(isClassificationLayer);


% Traverse the layer graph in reverse starting from the classification
% layer. If the network branches, throw an error.
currentLayerIdx = find(isClassificationLayer);
while true
    
    if numel(currentLayerIdx) ~= 1
        error('Layer graph must have a single learnable layer preceding the classification layer.')
    end
    
    currentLayerType = class(lgraph.Layers(currentLayerIdx));
    isLearnableLayer = ismember(currentLayerType, ...
        ['nnet.cnn.layer.FullyConnectedLayer','nnet.cnn.layer.Convolution2DLayer']);
    
    if isLearnableLayer
        learnableLayer =  lgraph.Layers(currentLayerIdx);
        return
    end
    
    currentDstIdx = find(layerNames(currentLayerIdx) == dst);
    currentLayerIdx = find(src(currentDstIdx) == layerNames);
    
end

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
```
#### MATLAB output for above fruit classification based model

![image](https://github.com/ashwini0921/ROBOTIC-ARM-WITH-IMAGE-PROCESSING/assets/111654188/15763868-638c-4012-af4f-719532522925)

#### Conclusion from above output

After 6 epochs the validation accuracy acheived after 6 epochs is 100%. The validation accuracy curve closely follows the training accuracy curve hence it can be concluded that there is a less overfitting.

### Conclusion
Both pre-trained model GoogLeNet and SqueezeNet were implemented. It is found that SqueezeNet acheived 100% accuracy for above application.

### Create a function to classify the fruits directly from trained network

#### MATLAB CODE FOR THE FUNCTION

```
disp(classifier(imread("Test_data/apple18.jpg"),net))
function label=classifier(image,net)
inputSize = net.Layers(1).InputSize
pixelRange = [-30 30];
scaleRange = [0.9 1.1];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange, ...
    'RandXScale',scaleRange, ...
    'RandYScale',scaleRange);
augimdsimage = augmentedImageDatastore(inputSize(1:2),image, ...
    'DataAugmentation',imageAugmenter,'ColorPreprocessing', 'gray2rgb');
[label,score] = classify(net,augimdsimage);
end
```
#### FIGURE OF APPLE18
![apple18](https://github.com/ashwini0921/ROBOTIC-ARM-WITH-IMAGE-PROCESSING/assets/111654188/dc62704b-9de3-4d7e-a010-9b8bf8496c85)

#### MATLAB OUTPUT
![image](https://github.com/ashwini0921/ROBOTIC-ARM-WITH-IMAGE-PROCESSING/assets/111654188/1d8d5af5-2985-4172-b397-1ebf966f9134)












