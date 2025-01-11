%% Visualization of a Small Neural Network with More Inputs
% Neural Network size is customizable, but 3 input and 1 output are needed 
% for proper visualization.

%% Clearing Environment
clc; clear; close all;
format long;

%% Training Data
test_size = 1000;

training_data_aux = ones([4*8 test_size]);

for group_idx = 1:8
    center_shift = zeros([3 1]);
    binary = dec2bin(group_idx-1, 3);
    expected_output = -1;

    for bit_idx = 1:3
        bit = str2double(binary(bit_idx));
        center_shift(bit_idx) = -1*(bit == 0) + bit;
    end

    if(all(center_shift == [-1; 1; -1]) || all(center_shift == [1; 1; -1]) || all(center_shift == [1; -1; 1]))
        expected_output = 1;
    end

    training_data_aux((group_idx*4-3):(group_idx*4),:) = [0.25*randn([3 test_size]) + center_shift; expected_output*ones([1 test_size])];
end

training_data = reshape(training_data_aux, 3+1, []);

x1 = training_data(1,:);
x2 = training_data(2,:);
x3 = training_data(3,:);

f = training_data(4,:);

clear binary bit bit_idx center_shift expected_output group_idx training_data_aux training_data test_size

%% Plot of Training Data
figure

S = 15;
C = [((1-f)/2)' 0*((1-f)/2)' ((f+1)/2)'];

scatter3(x1,x2,x3,S,C,"filled")

title("Expected Output");
subtitle("Color of Dots Represents Result, Red = -1, Blue = 1")
xlabel("x_1"); ylabel("x_2"); zlabel("x_3");
grid on;
xlim([-2, 2]); ylim([-2, 2]); zlim([-2, 2]);
daspect([1 1 1]);

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
    @(out, tar) 0.5.*sum((out - tar).^2, 1) 
    @(out, tar) (out - tar)
    };

%% Neural Network Memory Allocation

% Considerations:
% - Input is not defined as a Layer
% - Last Layer is Output

% Neural Network Size Definition
input_size = 3;
layer_sizes = [6; 6; 1];

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

    weights{layer_ind} = rand([rows cols]);
    bias{layer_ind} = 0.1*ones([rows 1]);
    neuron_in{layer_ind} = zeros([rows 1]);
    neuron_out{layer_ind} = zeros([rows 1]);
end

clear rows cols layer_ind

%% Learning Test

for index = 1:size(x1,2)
    % Training Input and Target Output
    training_data = [x1(:,index);x2(:,index);x3(:,index)];
    target_output = f(:,index);
    
    % Forward Propagation
    [neuron_in, neuron_out] = forwardPropagation(training_data, weights, bias, tanh);

    % Backward Propagation
    [neuron_in_der, neuron_out_der, error_der] = backwardPropagation(training_data, neuron_in, neuron_out, target_output, weights, tanh, tanh, msqe);

    % Updating Neural Network Weights and Biases
    [weights, bias] = updateNeuralNetwork(neuron_in_der, error_der, weights, bias, 0.05);

end

% Last Forward Propagation to see results
[~, neuron_out] = forwardPropagation([x1;x2;x3], weights, bias, tanh);
test_output = neuron_out{end};

% Cost Function Result After Training
cost = mean(msqe{1}(test_output,f));
%d_cost = msqe{2}(test_output,f);

%% Plot of Results
figure

S = 15;
C = [((1-test_output)/2)' 0*((1-test_output)/2)' ((test_output+1)/2)'];

scatter3(x1,x2,x3,S,C,"filled")

title("Output After Training");
subtitle("Color of Dots Represents Result, Red = -1, Blue = 1")
xlabel("x_1"); ylabel("x_2"); zlabel("x_3");
grid on;
xlim([-2, 2]); ylim([-2, 2]); zlim([-2, 2]);
daspect([1 1 1]);
