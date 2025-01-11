%% Visualization of the Testing Process for Digit-Recognizing Neural Network
% Neural Network data is saved by Training program in a .mat file and
% loaded here.

%% Clearing Environment
clc; clear; close all;
format long;

%% Loading MNIST Data
load("mnist.mat");
clear training

%% Loading Neural Network
load("neuralNetwork3.mat");

%% Activation Functions
relu = {
    @(x) x.*(x >= 0) 
    @(x) 1.*(x >= 0)
    };
sfmx = {
    @(x) exp(x)./sum(exp(x),1)
    @(x) (exp(x)./sum(exp(x),1)).*(1-(exp(x)./sum(exp(x),1)))
    };

%% Testing Input and Output

% Create an Empty Arrange to Store the Showed Image
img = ones([test.height test.width 3]);

% Initialize Figure
figure
k = image(img);
title("Tested Image")
xlim([0, 28]); ylim([0, 28]);
daspect([1 1 1]);

tests_performed = 50;

% Cycle Testing for Multiple Test Images
for image_index = 1:tests_performed
    % Input and Expected Output Definition
    testing_image = test.images(:,:,image_index);
    testing_input = reshape(testing_image,test.height*test.width,[]);
    
    target_output = zeros([10 1]);
    target_output(test.labels(image_index) + 1,1) = 1;
    
    % Forward Propagation
    [neuron_in, neuron_out] = forwardPropagation(testing_input, weights, bias, relu, sfmx);
    
    % Classification Results
    [M,I] = max(neuron_out{end});
    
    clc;
    confianza = M
    prediccion = I-1
    expected = test.labels(image_index)
    
    % Visualization
    img(:,:,1) = 1-testing_image;
    img(:,:,2) = 1-testing_image;
    img(:,:,3) = 1-testing_image;
    
    k.CData = img;
    drawnow;
    pause(3);
end