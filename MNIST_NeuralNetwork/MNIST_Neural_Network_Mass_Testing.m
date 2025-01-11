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
testing_input = zeros([test.height*test.width test.count]);
target_output = zeros([10 test.count]);

for i = 1:test.count
    testing_image = test.images(:,:,i);
    testing_input(:,i) = reshape(testing_image,test.height*test.width,[]);
    target_output(test.labels(i) + 1,i) = 1;
end

clear test testing_image i

%% Forward Propagation
[~, neuron_out] = forwardPropagation(testing_input, weights, bias, relu, sfmx);

[~,I1] = max(target_output);
[~,I2] = max(neuron_out{end});

coincidencias = (I1 == I2);

aciertos = sum(target_output(:,coincidencias),2)./sum(target_output,2)

mean(aciertos)

mean(coincidencias)