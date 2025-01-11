%% Manual Definition and Forward and Back Propagation of a Neural Network
% Neural Network is defined with 2 input, 1 output and 1 hidden layer with
% 2 neurons. Manual forward and backpropagation is performed with the 
% objective of understanding better the process and testing the defined 
% functions.

%% Clearing Environment
clc; clear; close all;
format long;

%% Test Intput
test_input1 = [0.25*randn([2 1000]) + [ 1;  1]; 1*ones([2 1000])];
test_input2 = [0.25*randn([2 1000]) + [ 1; -1]; -1*ones([2 1000])];
test_input3 = [0.25*randn([2 1000]) + [-1; -1]; 1*ones([2 1000])];
test_input4 = [0.25*randn([2 1000]) + [-1;  1]; -1*ones([2 1000])];

test_input = [test_input1 test_input2 test_input3 test_input4];

x1 = test_input(1,:);
x2 = test_input(2,:);
z = test_input(3,:);

clear test_input test_input1 test_input2 test_input3 test_input4

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
mesq = {
    @(out, tar) 0.5.*(out - tar).^2 
    @(out, tar) (out - tar)
    };

%% Neural Network Memory Allocation

% Considerations:
% - Input is not defined as a Layer
% - Last Layer is Output

%layer_sizes = [2; 3; 4; 1]; % obsolete by new standard(?)

input_size = 2;
layer_sizes = [2; 1];

neuron_in = cell(size(layer_sizes));
neuron_out = cell(size(layer_sizes));
weights = cell(size(layer_sizes));
bias = cell(size(layer_sizes));

for layer_ind = 1:size(layer_sizes,1)
    rows = layer_sizes(layer_ind);

    if layer_ind > 1
        cols = layer_sizes(layer_ind - 1);
    else
        cols = input_size;
    end

    weights{layer_ind} = ones([rows cols]);
    bias{layer_ind} = zeros([rows 1]);
    neuron_in{layer_ind} = zeros([rows 1]);
    neuron_out{layer_ind} = zeros([rows 1]);
end

clear rows cols layer_ind

%% Manual Neural Network Definition

% Hidden Layer 1
weights{1} = [1 1; -1 -1];
bias{1} = [-1; -1];

% Hidden Layer 2
weights{2} = [1 1];
bias{2} = [0];

%% Forward Propagation Test

% Inputs
test_input = [x1;x2];
test_input = test_input(:,:);

% Forward propagation
[neuron_in, neuron_out] = forwardPropagation(test_input, weights, bias, relu);
test_output = neuron_out{end};

% Cost Function
cost = mesq{1}(test_output, z);

%% Backward Propagation Test

% Derivatives Memory Allocation
neuron_in_der = cell(size(layer_sizes));
neuron_out_der = cell(size(layer_sizes));
error_der = cell(size(layer_sizes));

% Hidden Layer 2
neuron_out_der{2} = mesq{2}(test_output, z);
neuron_in_der{2} = neuron_out_der{2}.*relu{2}(neuron_in{2});
error_der{2} = neuron_in_der{2}*neuron_out{1}';

% Hidden Layer 1
neuron_out_der{1} = weights{2}'*neuron_in_der{2};
neuron_in_der{1} = neuron_out_der{1}.*relu{2}(neuron_in{1});
error_der{1} = neuron_in_der{1}*test_input';

% Backpropagation Function
[n_in_der, n_out_der, err_der] = backwardPropagation(test_input, neuron_in, neuron_out, z, weights, relu, relu, mesq);

%% Results Comparison
all(n_in_der{1} == neuron_in_der{1},"all")
all(n_in_der{2} == neuron_in_der{2},"all")

all(n_out_der{1} == neuron_out_der{1},"all")
all(n_out_der{2} == neuron_out_der{2},"all")

all(err_der{1} == error_der{1},"all")
all(err_der{2} == error_der{2},"all")
