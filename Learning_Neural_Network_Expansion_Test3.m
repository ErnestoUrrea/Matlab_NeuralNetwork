%% Clearing Environment
clc; clear; close all;
format long;

%% Training Data
test_size = 5000;

training_data1 = [0.25*randn([2 test_size]) + [ 1;  1]; [  1;  0;  0;  0].*ones([4 test_size])];
training_data2 = [0.25*randn([2 test_size]) + [ 1; -1]; [  0;  1;  0;  0].*ones([4 test_size])];
training_data3 = [0.25*randn([2 test_size]) + [-1; -1]; [  0;  0;  1;  0].*ones([4 test_size])];
training_data4 = [0.25*randn([2 test_size]) + [-1;  1]; [  0;  0;  0;  1].*ones([4 test_size])];

training_data = reshape ([training_data1 ; training_data2 ; training_data3 ; training_data4], size(training_data1,1), [] );

x1 = training_data(1,:);
x2 = training_data(2,:);

f = training_data(3:6,:);

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
sfmx = {
    @(x) exp(x)./sum(exp(x),1)
    @(x) (exp(x)./sum(exp(x),1)).*(1-(exp(x)./sum(exp(x),1)))
    };

%% Cost Functions
msqe = {
    @(out, tar) 0.5.*sum((out - tar).^2, 1) 
    @(out, tar) (out - tar)
    };
ccef = {
    @(out, tar) -sum(tar.*log(out), 1)
    @(out, tar) -tar./out + (1-tar)./(1-out)
    };

%% Neural Network Memory Allocation

% Considerations:
% - Input is not defined as a Layer
% - Last Layer is Output

% Neural Network Size Definition
input_size = 2;
layer_sizes = [6; 6; 4];

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
    training_data = [x1(:,index);x2(:,index)];
    %target_output = [z1(index);z2(index)];
    target_output = f(:,index);
    
    % Forward Propagation
    %[neuron_in, neuron_out] = forwardPropagation(training_data, weights, bias, tanh);
    [neuron_in, neuron_out] = forwardPropagation(training_data, weights, bias, relu, sfmx);

    % Backward Propagation
    %[neuron_in_der, neuron_out_der, error_der] = backwardPropagation(training_data, neuron_in, neuron_out, target_output, weights, tanh, tanh, msqe);
    [neuron_in_der, neuron_out_der, error_der] = backwardPropagation(training_data, neuron_in, neuron_out, target_output, weights, relu, sfmx, ccef);

    % Updating Neural Network Weights and Biases
    [weights, bias] = updateNeuralNetwork(neuron_in_der, error_der, weights, bias, 0.05);

end

% Last Forward Propagation to see results
%[~, neuron_out] = forwardPropagation([x1;x2], weights, bias, tanh);
[~, neuron_out] = forwardPropagation([x1;x2], weights, bias, relu, sfmx);
test_output = neuron_out{end};

%cost = mean(msqe{1}(test_output,f));
cost = mean(ccef{1}(test_output,f));
%d_cost = msqe{2}(test_output,f);

%% Plot of Results in 2D
[X, Y] = meshgrid(-2:0.05:2,-2:0.05:2);

Xin = reshape(X,1,[]);
Yin = reshape(Y,1,[]);

%[~, Zaux] = forwardPropagation([Xin; Yin], weights, bias, tanh);
[~, Zaux] = forwardPropagation([Xin; Yin], weights, bias, relu, sfmx);

[M,I] = max(Zaux{end});

ZI = reshape(I,size(X));
ZM = reshape(abs(M),size(X));

clear Xin Yin Zaux I

figure
[~,h] = contourf(X,Y,ZI);
set(h,'LineColor','none')

hold on
plot(x1, x2,'.k')
hold off

grid on

clim([1 4])
xlim([-2, 2])
ylim([-2, 2])

daspect([1 1 1])
colorbar;

figure
[~,h] = contourf(X,Y,ZM);
set(h,'LineColor','none')

hold on
plot(x1, x2,'.k')
hold off

grid on

clim([0 1])
xlim([-2, 2])
ylim([-2, 2])

daspect([1 1 1])
colorbar;

