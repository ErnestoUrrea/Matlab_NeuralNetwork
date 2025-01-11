%% Clearing Environment
clc; clear; close all;
format long;

%% MNIST Data
load("mnist.mat");

%% Activation Functions
relu = {
    @(x) x.*(x >= 0) 
    @(x) 1.*(x >= 0)
    };
tanh = {
    @(x) (1 - exp(-2.*x))./(1 + exp(-2.*x)) 
    @(x) 1 - ((1 - exp(-2.*x))./(1 + exp(-2.*x))).^2 
    };
sigm = {
    @(x) 1./(1 + exp(-x)) 
    @(x) (1./(1 + exp(-x))).*(1 - (1./(1 + exp(-x))))
    };

%% Cost Functions
msqe = {
    @(out, tar) 0.5.*sum((out - tar).^2, 1)./size(out,1) 
    @(out, tar) (out - tar)./size(out,1) 
    };

%% Used Functions
activationFunction = relu;
activationFunctionOut = relu;
costFunction = msqe;

%% Neural Network Memory Allocation

% Considerations:
% - Input is not defined as a Layer
% - Last Layer is Output

% Neural Network Size Definition
input_size = training.height*training.width;
layer_sizes = [50; 50; 10];

% Neurons and Links Memory Allocation
neuron_in = cell(size(layer_sizes));
neuron_out = cell(size(layer_sizes));
weights = cell(size(layer_sizes));
bias = cell(size(layer_sizes));

% Derivatives Memory Allocation
neuron_in_der = cell(size(layer_sizes));
neuron_out_der = cell(size(layer_sizes));
error_der = cell(size(layer_sizes));

for layer_ind = 1:size(layer_sizes,1)
    rows = layer_sizes(layer_ind);

    if layer_ind > 1
        cols = layer_sizes(layer_ind - 1);
    else
        cols = input_size;
    end

    weights{layer_ind} = rand([rows cols])-1;
    bias{layer_ind} = rand([rows 1])-1;
    neuron_in{layer_ind} = zeros([rows 1]);
    neuron_out{layer_ind} = zeros([rows 1]);
end

clear rows cols layer_ind

%% Learning Test

for index = 1:(training.count)
    % Training Input and Target Output
    training_input = reshape(training.images(:,:,index),[],1);
    target_output = zeros([10 1]);
    target_output(training.labels(index) + 1) = 1;
    
    % Forward Propagation
    [neuron_in, neuron_out] = forwardPropagation(training_input, weights, bias, activationFunction, activationFunctionOut);

    if(mod(index,1000) == 0)
        index
        neuron_out{end}'
        costFunction{1}(neuron_out{end},target_output)'
    end

    % Backward Propagation
    [neuron_in_der, neuron_out_der, error_der] = backwardPropagation(training_input, neuron_in, neuron_out, target_output, weights, activationFunction, activationFunctionOut, costFunction);

    % Updating Neural Network Weights and Biases
    [weights, bias] = updateNeuralNetwork(neuron_in_der, error_der, weights, bias, 0.5);
end

%% Save Trained Neural Network for Testing
save neuralNetwork.mat weights bias sigm activationFunction activationFunctionOut costFunction
