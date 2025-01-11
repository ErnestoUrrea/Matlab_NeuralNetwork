%% Training Process for Digit-Recognizing Neural Network
% Neural Network data is saved in a .mat file for testing in a different
% program.

%% Clearing Environment
clc; clear; close all;
format long;

%% Load MNIST Data
load("mnist.mat");
clear test;

%% Training Data
test_size = training.count;

% Training Input
x = reshape(training.images(:,:,1:test_size),training.height*training.width,[]);

% Target Output
f = zeros([10 test_size]);
for i = 1:test_size
    f(training.labels(i) + 1,i) = 1;
end

clear i test_size training

%% Activation Functions
relu = {
    @(x) x.*(x >= 0) 
    @(x) 1.*(x >= 0)
    };
sfmx = {
    @(x) exp(x)./sum(exp(x),1)
    @(x) (exp(x)./sum(exp(x),1)).*(1-(exp(x)./sum(exp(x),1)))
    };
axfn = @(out, tar) out-tar; 

%% Cost Functions
ccef = {
    @(out, tar) -sum(tar.*log(out), 1)
    @(out, tar) -tar./out + (1-tar)./(1-out)
    };

%% Neural Network Memory Allocation

% Considerations:
% - Input is not defined as a Layer
% - Last Layer is Output

% Neural Network Size Definition
input_size = 28*28;
% layer_sizes = [500; 300; 10];
layer_sizes = [300; 10];

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

    weights{layer_ind} = rand([rows cols]) - 0.5;
    bias{layer_ind} = 0.1*ones([rows 1]);
    neuron_in{layer_ind} = zeros([rows 1]);
    neuron_out{layer_ind} = zeros([rows 1]);
end

clear rows cols layer_ind

%% Learning Test

% Matrix for Saving Cost Values
cost = zeros([1 60000/1000]);
check_period = 1000;

% Training Process
for index = 1:size(x,2)
    % Training Input and Target Output
    training_data = x(:,index);
    target_output = f(:,index);
    
    % Forward Propagation
    [neuron_in, neuron_out] = forwardPropagation(training_data, weights, bias, relu, sfmx);

    % Backward Propagation
    % NOTE THAT Backpropagation function has an extra argument and is
    % defined different in this environment. This is because the cost
    % derivative performs a division by zero when the output value of the 
    % network for the target output is close to and rounded to 1. This can
    % be solved using a simplification of the input derivative for the
    % combination of Categorical Cross-Entropy Loss and Softmax Activation.
    % This is a temporal fix and I plan to solve it later for a more
    % seamless implementation for any cost and activation functions.
    [neuron_in_der, neuron_out_der, error_der] = backwardPropagation(training_data, neuron_in, neuron_out, target_output, weights, relu, sfmx, ccef,axfn);

    % Updating Neural Network Weights and Biases
    [weights, bias] = updateNeuralNetwork(neuron_in_der, error_der, weights, bias, 0.01);

    % Save Cost for Every 1000 Iterations
    if(mod(index,check_period) == 0)
        disp(index)
        cost(index/check_period) = ccef{1}(neuron_out{end},target_output);
    end

end

% Last Forward Propagation to see results
[~, neuron_out] = forwardPropagation(x, weights, bias, relu, sfmx);
test_output = neuron_out{end};

% Cost Graphic
figure
plot(cost)
title("Cost");
xlabel("Iteration"); ylabel("Cost");

%% Checking Percentage of Matches
% Predicted and Expected Values
[~,I1] = max(test_output);
[~,I2] = max(f);

% Match Percentage Calculation
coincidencias = (I1 == I2);
match_percentage = mean(coincidencias)

%% Save Trained Neural Network for Testing
save neuralNetwork4.mat weights bias