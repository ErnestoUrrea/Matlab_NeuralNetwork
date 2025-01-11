%% Clearing Environment
clc; clear; close all;
format long;

%% Training Data
test_size = 1000;

test_input1 = [0.25*randn([2 test_size]) + [ 1;  1]; 1*ones([2 test_size])];
test_input2 = [0.25*randn([2 test_size]) + [ 1; -1]; -1*ones([2 test_size])];
test_input3 = [0.25*randn([2 test_size]) + [-1; -1]; 1*ones([2 test_size])];
test_input4 = [0.25*randn([2 test_size]) + [-1;  1]; -1*ones([2 test_size])];

%training_input = [test_input1 test_input2 test_input3 test_input4];

training_input = reshape ([test_input1 ; test_input2 ; test_input3 ; test_input4], size(test_input1,1), [] );

x1 = training_input(1,:);
x2 = training_input(2,:);
z = training_input(3,:);

clear training_input test_input1 test_input2 test_input3 test_input4

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

backwardPropagation()

input_size = 2;
layer_sizes = [4; 3; 1];

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

% % Meshgrid for plotting learning process
% [X, Y] = meshgrid(-2:0.05:2,-2:0.05:2);
% 
% Xin = reshape(X,1,[]);
% Yin = reshape(Y,1,[]);
% 
% figure

for index = 1:size(x1,2)
    % Training Input and Target Output
    training_input = [x1(index);x2(index)];
    target_output = z(index);
    
    % Forward Propagation
    [neuron_in, neuron_out] = forwardPropagation(training_input, weights, bias, tanh);

    % Backward Propagation
    [neuron_in_der, neuron_out_der, error_der] = backwardPropagation(training_input, neuron_in, neuron_out, target_output, weights, tanh, tanh, mesq);

    % Updating Neural Network Weights and Biases
    [weights, bias] = updateNeuralNetwork(neuron_in_der, error_der, weights, bias, 0.3);

%     if mod(index,10) == 0
%         [~, Zaux] = forwardPropagation([Xin; Yin], weights, bias, tanh);
%         Z = reshape(Zaux{end},size(X));
%     
%         surf(X,Y,Z)
%         hold on
%         plot(x1, x2,'.b') 
%         hold off
%         drawnow
%     
%         disp(index)
%     end
end

% Close Learning Graphic
close all;

% Last Forward Propagation to see results
[~, neuron_out] = forwardPropagation([x1;x2], weights, bias, tanh);
test_output = neuron_out{end};

cost = mean(mesq{1}(test_output,z))

%% Plot of Results in 2D
[X, Y] = meshgrid(-2:0.05:2,-2:0.05:2);

Xin = reshape(X,1,[]);
Yin = reshape(Y,1,[]);

[~, Zaux] = forwardPropagation([Xin; Yin], weights, bias, tanh);
Z = reshape(Zaux{end},size(X));

clear Xin Yin Zaux

figure
[C,h] = contourf(X,Y,Z);
set(h,'LineColor','none')

hold on
plot(x1(test_output > 0), x2(test_output > 0),'.r') 
plot(x1(test_output <= 0), x2(test_output <= 0),'.b')  
hold off

grid on

xlim([-2, 2])
ylim([-2, 2])

daspect([1 1 1])

%% Plot of Results in 3D
% [X, Y] = meshgrid(-2:0.05:2,-2:0.05:2);
% 
% Xin = reshape(X,1,[]);
% Yin = reshape(Y,1,[]);
% 
% [~, Zaux] = forwardPropagation([Xin; Yin], weights, bias, tanh);
% Z = reshape(Zaux{end},size(X));
% 
% clear Xin Yin Zaux
% 
% figure
% surf(X,Y,Z)
% 
% hold on
% plot3(x1(test_output > 0), x2(test_output > 0), z(test_output > 0), '.r') 
% plot3(x1(test_output <= 0), x2(test_output <= 0), z(test_output <= 0), '.b') 
% hold off
% 
% grid on
% 
% xlim([-2, 2])
% ylim([-2, 2])
% 
% daspect([1 1 1])

%% Plot of Target Results

% figure
% 
% plot(x1(z > 0), x2(z > 0),'.r') 
% hold on
% plot(x1(z <= 0), x2(z <= 0),'.b') 
% hold off
% 
% yline(0)
% xline(0)
% 
% xlim([-2, 2])
% ylim([-2, 2])
% 
% daspect([1 1 1])