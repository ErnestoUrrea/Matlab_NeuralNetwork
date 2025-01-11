%%

% Ideas:
% - Objetos de funciones? o matrices que 

%%
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

% %Derivatives Memory Allocation
% neuron_in_der = cell(size(layer_sizes));
% neuron_out_der = cell(size(layer_sizes));
% error_der = cell(size(layer_sizes));
% 
% % Hidden Layer 2
% neuron_out_der{2} = mesq{2}(test_output, z);
% neuron_in_der{2} = neuron_out_der{2}.*relu{2}(neuron_in{2});
% error_der{2} = neuron_in_der{2}*neuron_out{1}';
% 
% % Hidden Layer 1
% neuron_out_der{1} = weights{2}'*neuron_in_der{2};
% neuron_in_der{1} = neuron_out_der{1}.*relu{2}(neuron_in{1});
% error_der{1} = neuron_in_der{1}*test_input';

[n_in_der, n_out_der, err_der] = backwardPropagation(test_input, neuron_in, neuron_out, z, weights, relu, relu, mesq);

% mean(n_in_der{1} == neuron_in_der{1},"all")
% mean(n_in_der{2} == neuron_in_der{2},"all")
% 
% mean(n_out_der{1} == neuron_out_der{1},"all")
% mean(n_out_der{2} == neuron_out_der{2},"all")
% 
% mean(err_der{1} == error_der{1},"all")
% mean(err_der{2} == error_der{2},"all")


%% Plot of Results in 3D
[X, Y] = meshgrid(-2:0.05:2,-2:0.05:2);

Xin = reshape(X,1,[]);
Yin = reshape(Y,1,[]);

[~, Zaux] = forwardPropagation([Xin; Yin], weights, bias, relu);
Z = reshape(Zaux{end},size(X));

clear Xin Yin Zaux

figure
surf(X,Y,Z) 

hold on
plot(test_input(1,test_output > 0), test_input(2,test_output > 0),'.r') 
plot(test_input(1,test_output <= 0), test_input(2,test_output <= 0),'.b') 
hold off

xlim([-2, 2])
ylim([-2, 2])

daspect([1 1 1])

%% Plot of Binary Results
% 
% figure
% 
% plot(test_input(1,layer2r > 0), test_input(2,layer2r > 0),'.r') 
% hold on
% plot(test_input(1,layer2r <= 0), test_input(2,layer2r <= 0),'.b') 
% hold off
% 
% yline(0)
% xline(0)
% 
% xlim([-2, 2])
% ylim([-2, 2])
% 
% daspect([1 1 1])