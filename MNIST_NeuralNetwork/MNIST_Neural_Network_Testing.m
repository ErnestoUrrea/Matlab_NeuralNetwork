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
%image_index = 30;

figure

for image_index = 1:50
testing_image = round(test.images(:,:,image_index));
testing_input = reshape(testing_image,test.height*test.width,[]);

target_output = zeros([10 1]);
target_output(test.labels(image_index) + 1,1) = 1;

%% Forward Propagation
[neuron_in, neuron_out] = forwardPropagation(testing_input, weights, bias, relu, sfmx);

[M,I] = max(neuron_out{end});

clc
confianza = M
prediccion = I-1

%% Visualization
empty_img = ones([test.height test.width 3]);

empty_img(:,:,1) = 1-testing_image;
empty_img(:,:,2) = 1-testing_image;
empty_img(:,:,3) = 1-testing_image;

image(empty_img)

title(num2str(test.labels(image_index)))

xlim([0, 28])
ylim([0, 28])

daspect([1 1 1])

drawnow

pause(3)
end