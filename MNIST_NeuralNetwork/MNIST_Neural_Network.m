%% Clearing Environment
clc; clear; close all;
format long;

%% MNIST Data
load("mnist.mat");
clear test;

%% Training Data
test_size = training.count;

x = reshape(training.images(:,:,1:test_size),training.height*training.width,[]);

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
% tanh = {
%     @(x) (1 - exp(-2.*x))./(1 + exp(-2.*x)) 
%     @(x) 1 - ((1 - exp(-2.*x))./(1 + exp(-2.*x))).^2 
%     };
% sigm = {
%     @(x) 1./(1 + exp(-x)) 
%     @(x) (1./(1 + exp(-x))).*(1 - (1./(1 + exp(-x))))
%     };
sfmx = {
    @(x) exp(x)./sum(exp(x),1)
    @(x) (exp(x)./sum(exp(x),1)).*(1-(exp(x)./sum(exp(x),1)))
    };
axfn = @(out, tar) out-tar; 

%% Cost Functions
% msqe = {
%     @(out, tar) 0.5.*sum((out - tar).^2, 1) 
%     @(out, tar) (out - tar)
%     };
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
layer_sizes = [500; 300; 10];

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

cost = zeros([1 60000/1000]);

for index = 1:size(x,2)
    % Training Input and Target Output
    training_data = x(:,index);
    target_output = f(:,index);
    
    % Forward Propagation
    [neuron_in, neuron_out] = forwardPropagation(training_data, weights, bias, relu, sfmx);

    % Backward Propagation
    [neuron_in_der, neuron_out_der, error_der] = backwardPropagation(training_data, neuron_in, neuron_out, target_output, weights, relu, sfmx, ccef,axfn);

    % Updating Neural Network Weights and Biases
    [weights, bias] = updateNeuralNetwork(neuron_in_der, error_der, weights, bias, 0.01);

    if(mod(index,1000) == 0)
        index
        cost(index/1000) = ccef{1}(neuron_out{end},target_output);
    end

end

% Last Forward Propagation to see results
[~, neuron_out] = forwardPropagation(x, weights, bias, relu, sfmx);
test_output = neuron_out{end};

figure
plot(cost)
%cost = mean(msqe{1}(test_output,f));
%d_cost = msqe{2}(test_output,f);

%% Checking Percentage of Coincidences
[~,I1] = max(test_output);
[~,I2] = max(f);

coincidencias = (I1 == I2);

mean(coincidencias)

%% Save Trained Neural Network for Testing
save neuralNetwork3.mat weights bias