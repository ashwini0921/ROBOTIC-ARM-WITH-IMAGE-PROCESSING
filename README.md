# ROBOTIC-ARM-WITH-IMAGE-PROCESSING
Designed robotic Arm which can separate fruits using Deep Learning

## ABSTRACT

The project involves the development of a 6-degree-of-freedom (DOF) robotic arm capable of fruit separation through image processing techniques. A classification model is designed using Deep Neural Networks (DNN) combine with Convolutional Neural Network to accurately identify and categorize fruits based on their size and shape attributes. This innovative approach combines robotics and artificial intelligence to enhance fruit sorting processes, contributing to efficiency and precision in agricultural and food industries. The integration of image processing and machine learning underscores the potential for automated fruit handling, optimizing quality control and streamlining production workflows.

## ROBOT SPECIFICATION
Drive System: Electric motor(servo motor)
Programming Software: MATLAB
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
![1](https://github.com/ashwini0921/ROBOTIC-ARM-WITH-IMAGE-PROCESSING/assets/111654188/2b070fb9-2a2c-4afc-b0da-2abaf26219ae)
#### SAMPLE IMAGES OF ORANGES
![2](https://github.com/ashwini0921/ROBOTIC-ARM-WITH-IMAGE-PROCESSING/assets/111654188/26cad09a-abf4-4f21-96d9-67dd45759bea)
#### SAMPLE IMAGES OF BANANA
![3](https://github.com/ashwini0921/ROBOTIC-ARM-WITH-IMAGE-PROCESSING/assets/111654188/c263d5fc-023e-4e83-b728-6ecac685165e)

### Building the Deep Neural Network Model using GoogLeNet:

#### GoogLeNet Architecture:
![4](https://github.com/ashwini0921/ROBOTIC-ARM-WITH-IMAGE-PROCESSING/assets/111654188/57b7cdad-c1f7-46cd-b8de-dfd5cf4bc538)

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

![5](https://github.com/ashwini0921/ROBOTIC-ARM-WITH-IMAGE-PROCESSING/assets/111654188/6d4ce425-2178-4e87-986a-7dab2e41cf8f)

#### Conclusion from above output

After 6 epochs the validation accuracy acheived after 6 epochs is 94.12%. The validation accuracy curve closely follows the training accuracy curve hence it can be concluded that there is less overfitting.


### Building the Deep Neural Network Model using SqueezeNet:

#### SqueezNet Architecture:

![6](https://github.com/ashwini0921/ROBOTIC-ARM-WITH-IMAGE-PROCESSING/assets/111654188/32ab8505-8a55-42df-801b-94a8f5c36004)
![7](https://github.com/ashwini0921/ROBOTIC-ARM-WITH-IMAGE-PROCESSING/assets/111654188/f5d2464e-05cd-409f-b93d-461bbd0d04e2)
![8](https://github.com/ashwini0921/ROBOTIC-ARM-WITH-IMAGE-PROCESSING/assets/111654188/76a83ee1-ea40-4962-997a-894352713e78)

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

![9](https://github.com/ashwini0921/ROBOTIC-ARM-WITH-IMAGE-PROCESSING/assets/111654188/0daa2ab8-1ff6-42dc-9215-c03342445794)

#### Conclusion from above output

After 6 epochs the validation accuracy acheived after 6 epochs is 100%. The validation accuracy curve closely follows the training accuracy curve hence it can be concluded that there is a less overfitting.

### Conclusion
Both pre-trained model GoogLeNet and SqueezeNet were implemented. It is found that SqueezeNet acheived 100% accuracy for above application.

### Create a function to classify the fruits directly from trained network

#### MATLAB CODE FOR THE FUNCTION

```
disp(classifier(imread("Test_data/apple18.jpg"),net))
function label=classifier(image1,net)
inputSize = net.Layers(1).InputSize
pixelRange = [-30 30];
scaleRange = [0.9 1.1];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange, ...
    'RandXScale',scaleRange, ...
    'RandYScale',scaleRange);
augimdsimage = augmentedImageDatastore(inputSize(1:2),image1, ...
    'DataAugmentation',imageAugmenter,'ColorPreprocessing', 'gray2rgb');
[label,score] = classify(net,augimdsimage);
end
```
#### FIGURE OF APPLE18
![10](https://github.com/ashwini0921/ROBOTIC-ARM-WITH-IMAGE-PROCESSING/assets/111654188/4d70fa91-f92e-4e4f-92b9-637ac58c7e73)

#### MATLAB OUTPUT
![11](https://github.com/ashwini0921/ROBOTIC-ARM-WITH-IMAGE-PROCESSING/assets/111654188/9ba6e092-668b-47bb-b96f-f86f46f10508)

## HARDWARE SETUP AND IMPLEMENTAION

### Move the endeffector to object location

#### MATLAB CODE FOR CONTROLING ROBOT SERVO MOTOR
```
a = arduino('COM4', 'Uno', 'Libraries', 'Servo','Ultrasonic');
s1 = servo(a, 'D8')
s2 = servo(a, 'D9')
s3 = servo(a, 'D10')
s4 = servo(a, 'D11')
s5 = servo(a, 'D12')
s6 = servo(a, 'D13')
distance_sensor= ultrasonic(a,'D2','D3')
r=0.0;
r1=11.8;
r2=23.1836;
while 1
setInitial(s1,s2,s3,s4,s5,s6);
xc=0;yc=readDistance(distance_sensor);zc=0;
yc=yc+9.3;
if(yc<23)
    r=sqrt(xc.^2+yc.^2+zc.^2);
    theta3=atan(yc./xc)*57.2957795;
    theta2=acos((r.^2-r1.^2-r2.^2)/(2*r1*r2));
    theta1=(acos((r1+r2*cos(theta2))./r)+acos(sqrt((xc.^2+yc.^2)./(r.^2))))*57.2957795;
    theta2=theta2*57.2957795;
    pause(0.5);
    writePosition(s1, theta1);
    pause(0.5);
    writePosition(s2, theta2);
    pause(0.5);
    writePosition(s3, theta3);
    activateendeffector(s6);
    pause(10)
end
pause(10)
end

function setInitial(s1,s2,s3,s4,s5,s6)
    pause(0.5);
    writePosition(s1, 100);
    pause(0.5);
    writePosition(s2, 60);
    pause(0.5);
    writePosition(s3, 0);
    pause
    writePosition(s4, 90);
    pause(0.5);
    writePosition(s5, 60);
    pause(0.5);
    writePosition(s6, 120);
    pause(0.5);
end
function activateendeffector(s6)
    pause(0.5);
    writePosition(s6, 0);
    pause(0.5);
end
```
#### HARDWARE SETUP

![12](https://github.com/ashwini0921/ROBOTIC-ARM-WITH-IMAGE-PROCESSING/assets/111654188/79bf524b-f783-43c7-85aa-1fecf42d51dc)

#### Video Demonstration

https://github.com/ashwini0921/ROBOTIC-ARM-WITH-IMAGE-PROCESSING/assets/111654188/26f681d1-f6d3-48db-98a4-d8c177dfbedc

## INTEGRATING CAMERA TO ABOVE HARDWARE SETUP AND DEPLOY CLASSIFICATION MODEL

### MATLAB CODE:
```
a = arduino('COM4', 'Uno', 'Libraries', 'Servo','Ultrasonic');
s1 = servo(a, 'D8')
s2 = servo(a, 'D9')
s3 = servo(a, 'D10')
s4 = servo(a, 'D11')
s5 = servo(a, 'D12')
s6 = servo(a, 'D13')
camera = webcam;
distance_sensor= ultrasonic(a,'D2','D3')
r=0.0;
r1=11.8;
r2=23.1836;
xc_orange=input("Enter x coordinate of orange box")
yc_orange=input("Enter y coordinate of orange box")
xc_apple=input("Enter x coordinate of apple box")
yc_apple=input("Enter y coordinate of apple box")
xc_banana=input("Enter x coordinate of banana box")
yc_banana=input("Enter y coordinate of banana box")
while 1
setInitial(s1,s2,s3,s4,s5,s6);
xc=0;yc=readDistance(distance_sensor);zc=0;
yc=yc+9.3;
im = snapshot(camera);
image(im)
label=classifier(im,net)
movecoordinate(xc,yc,zc,r,r1,r2);
pause(10)
if(label=='orange')
    movecoordinate(xc_orange,yc_orange,0,r,r1,r2);
    pause(10)
elseif(label=='apple')
    movecoordinate(xc_apple,yc_apple,0,r,r1,r2);
    pause(10)
elseif(label=='banana')
    movecoordinate(xc_banana,yc_banana,0,r,r1,r2);
    pause(10)
end
end

function setInitial(s1,s2,s3,s4,s5,s6)
    pause(0.5);
    writePosition(s1, 100);
    pause(0.5);
    writePosition(s2, 60);
    pause(0.5);
    writePosition(s3, 0);
    pause
    writePosition(s4, 90);
    pause(0.5);
    writePosition(s5, 60);
    pause(0.5);
    writePosition(s6, 120);
    pause(0.5);
end
function activateendeffector(s6)
    pause(0.5);
    writePosition(s6, 0);
    pause(0.5);
end
function movecoordinate(xc,yc,zc,r,r1,r2)
if(yc<23 && xc<23 && zc<23 && yc>-23 && xc>-23 && zc>-23)
    r=sqrt(xc.^2+yc.^2+zc.^2);
    theta3=atan(yc./xc)*57.2957795;
    theta2=acos((r.^2-r1.^2-r2.^2)/(2*r1*r2));
    theta1=(acos((r1+r2*cos(theta2))./r)+acos(sqrt((xc.^2+yc.^2)./(r.^2))))*57.2957795;
    theta2=theta2*57.2957795;
    pause(0.5);
    writePosition(s1, theta1);
    pause(0.5);
    writePosition(s2, theta2);
    pause(0.5);
    writePosition(s3, theta3);
    activateendeffector(s6);
    pause(10)
end
end

function label=classifier(image1,net)
inputSize = net.Layers(1).InputSize
pixelRange = [-30 30];
scaleRange = [0.9 1.1];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange, ...
    'RandXScale',scaleRange, ...
    'RandYScale',scaleRange);
augimdsimage = augmentedImageDatastore(inputSize(1:2),image1, ...
    'DataAugmentation',imageAugmenter,'ColorPreprocessing', 'gray2rgb');
[label,score] = classify(net,augimdsimage);
end
```

## REFERENCES

[1] https://arxiv.org/ftp/arxiv/papers/2201/2201.07882.pdf

[2] https://medium.com/xrpractices/how-to-train-your-robot-arm-fbf5dcd807e1

[3] https://www.researchgate.net/publication/341202448_Robotic_Arm_Control_and_Task_








