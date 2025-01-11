%% Visualization of the Training Process for a N.N. with Multiple Output
% Neural Network size is customizable, but 2 are needed for proper 
% visualization. Visualization of the Training Process is also possible,
% but not recommended for a large number of outputs.

%% Clearing Environment
clc; clear; close all;
format long;

%% Training Data
test_size = 5000;

training_data1 = [0.25*randn([2 test_size]) + [ 1;  1]; [ -1;  1; -1;  1; -1;  1; -1;  1; -1;  1; -1;  1; -1;  1; -1;  1].*ones([16 test_size])];
training_data2 = [0.25*randn([2 test_size]) + [ 1; -1]; [ -1; -1;  1;  1; -1; -1;  1;  1; -1; -1;  1;  1; -1; -1;  1;  1].*ones([16 test_size])];
training_data3 = [0.25*randn([2 test_size]) + [-1; -1]; [  1;  1;  1;  1; -1; -1; -1; -1;  1;  1;  1;  1; -1; -1; -1; -1].*ones([16 test_size])];
training_data4 = [0.25*randn([2 test_size]) + [-1;  1]; [  1;  1;  1;  1;  1;  1;  1;  1; -1; -1; -1; -1; -1; -1; -1; -1].*ones([16 test_size])];

training_data = reshape ([training_data1 ; training_data2 ; training_data3 ; training_data4], size(training_data1,1), [] );

x1 = training_data(1,:);
x2 = training_data(2,:);

z = training_data(3:18,:);

clear training_data training_data1 training_data2 training_data3 training_data4

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
input_size = 2;
layer_sizes = [4; 4; 16];

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
    target_output = z(:,index);
    
    % Forward Propagation
    [neuron_in, neuron_out] = forwardPropagation(training_input, weights, bias, tanh);

    % Backward Propagation
    [neuron_in_der, neuron_out_der, error_der] = backwardPropagation(training_input, neuron_in, neuron_out, target_output, weights, tanh, tanh, msqe);

    % Updating Neural Network Weights and Biases
    [weights, bias] = updateNeuralNetwork(neuron_in_der, error_der, weights, bias, 0.05);

%     % Update and Re-draw Actual State of the Network Every 10 Iterations
%     if mod(index,10) == 0
%         [~, Zcell] = forwardPropagation([Xin; Yin], weights, bias, tanh);
%         Zaux = Zcell{end};
%     
%         plotNeurons(X,Y,Zaux,x1,x2,layer_sizes(end));
%     
%         disp(index)
%     end

end

% Close Learning Graphic
close all;
clear X Y Z Xin Yin Zaux k training_input target_output index

% Last Forward Propagation to see results
[~, neuron_out] = forwardPropagation([x1;x2], weights, bias, tanh);
test_output = neuron_out{end};

% Cost Function Result After Training
cost = msqe{1}(test_output,z);
% d_cost = msqe{2}(test_output,z);

%% Plot of Results in 2D
[X, Y] = meshgrid(-2:0.05:2,-2:0.05:2);

Xin = reshape(X,1,[]);
Yin = reshape(Y,1,[]);

[~, Zaux] = forwardPropagation([Xin; Yin], weights, bias, tanh);

plotResults(X,Y,Zaux,x1,x2,z,layer_sizes(end));

clear Xin Yin X Y

%% Function Definitions

function plotNeurons(X,Y,Zaux,x1,x2,n)
    size1 = floor(sqrt(n));
    size2 = ceil(n/size1);

    tiledlayout(size1,size2)

    for i = 1:n
        Z = reshape(Zaux(i,:),size(X));

        nexttile
        surf(X,Y,Z)
        hold on
        plot(x1, x2,'.b') 
        hold off

        grid on
        xlim([-2, 2])
        ylim([-2, 2])
        daspect([1 1 1])
    end

    drawnow
end

function plotResults(X,Y,Zaux,x1,x2,z,n)

    for i = 1:n
        Z = reshape(Zaux{end}(i,:),size(X));

        figure
        [~,h] = contourf(X,Y,Z);
        set(h,'LineColor','none')
        
        hold on
        plot(x1(z(i,:) > 0), x2(z(i,:) > 0),'.r') 
        plot(x1(z(i,:) <= 0), x2(z(i,:) <= 0),'.b')  
        hold off
        
        grid on
        
        clim([-1.05 1.05])
        xlim([-2, 2])
        ylim([-2, 2])
        
        daspect([1 1 1])
        
        colorbar;
    end
end