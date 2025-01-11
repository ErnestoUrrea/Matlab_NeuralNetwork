%% Visualization of the Training Process for a Small Neural Network
% Neural Network size is customizable, but 2 input and 1 output are needed 
% for proper visualization.

%% Clearing Environment
clc; clear; close all;
format long;

%% Training Data
test_size = 1000;

test_input1 = [0.25*randn([2 test_size]) + [ 1;  1]; 1*ones([2 test_size])];
test_input2 = [0.25*randn([2 test_size]) + [ 1; -1]; -1*ones([2 test_size])];
test_input3 = [0.25*randn([2 test_size]) + [-1; -1]; 1*ones([2 test_size])];
test_input4 = [0.25*randn([2 test_size]) + [-1;  1]; -1*ones([2 test_size])];

training_input = reshape ([test_input1 ; test_input2 ; test_input3 ; test_input4], size(test_input1,1), [] );

x1 = training_input(1,:);
x2 = training_input(2,:);
f = training_input(3,:);

clear training_input test_input1 test_input2 test_input3 test_input4 test_size

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

input_size = 2;
layer_sizes = [4; 3; 1];

% Neurons and Links Memory Allocation
neuron_in = cell(size(layer_sizes));
neuron_out = cell(size(layer_sizes));
weights = cell(size(layer_sizes));
bias = cell(size(layer_sizes));

% Derivatives Memory Allocation
neuron_in_der = cell(size(layer_sizes));
neuron_out_der = cell(size(layer_sizes));
weights_der = cell(size(layer_sizes));

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

%% Training Test

% Meshgrid for Plotting the Training Process
[X, Y] = meshgrid(-2:0.05:2,-2:0.05:2);
Z = zeros(size(X));

Xin = reshape(X,1,[]);
Yin = reshape(Y,1,[]);

% Training Proces Plot Configuration
figure
k = surf(X,Y,Z);
hold on
plot(x1, x2,'.k') 
hold off

title("Output During Training Process"); 
xlabel("x_1"); ylabel("x_2"); zlabel("f(x_1,x_2)");
grid on;
xlim([-2, 2]); ylim([-2, 2]);
daspect([1 1 1]);

% Training Process
for index = 1:size(x1,2)
    % Training Input and Target Output
    training_input = [x1(index);x2(index)];
    target_output = f(index);
    
    % Forward Propagation
    [neuron_in, neuron_out] = forwardPropagation(training_input, weights, bias, tanh);

    % Backward Propagation
    [neuron_in_der, neuron_out_der, weights_der] = backwardPropagation(training_input, neuron_in, neuron_out, target_output, weights, tanh, tanh, mesq);

    % Updating Neural Network Weights and Biases
    [weights, bias] = updateNeuralNetwork(neuron_in_der, weights_der, weights, bias, 0.3);

    % Update and Re-draw Actual State of the Network Every 10 Iterations
    if mod(index,10) == 0
        [~, Zaux] = forwardPropagation([Xin; Yin], weights, bias, tanh);
        Z = reshape(Zaux{end},size(X));
    
        k.ZData = Z;
        drawnow
    
        disp(index)
    end
end

% Close Training Graphic and Clear Unneded Variables
close all;
clear X Y Z Xin Yin Zaux k training_input target_output index

% Last Forward Propagation to see results
[~, neuron_out] = forwardPropagation([x1;x2], weights, bias, tanh);
test_output = neuron_out{end};

% Cost Function Result After Training
cost = mean(mesq{1}(test_output,f));

%% Plot of Results in 2D
[X, Y] = meshgrid(-2:0.05:2,-2:0.05:2);

Xin = reshape(X,1,[]);
Yin = reshape(Y,1,[]);

[~, Zaux] = forwardPropagation([Xin; Yin], weights, bias, tanh);
Z = reshape(Zaux{end},size(X));

figure
contourf(X,Y,Z,'LineColor','none');

hold on
plot(x1(test_output > 0), x2(test_output > 0),'.r') 
plot(x1(test_output <= 0), x2(test_output <= 0),'.b')  
hold off

title("Colormap of Output"); 
xlabel("x_1"); ylabel("x_2");
grid on;
xlim([-2, 2]); ylim([-2, 2]);
daspect([1 1 1]);
colorbar;

clear Xin Yin Zaux X Y Z

%% Plot of Results in 3D
[X, Y] = meshgrid(-2:0.05:2,-2:0.05:2);

Xin = reshape(X,1,[]);
Yin = reshape(Y,1,[]);

[~, Zaux] = forwardPropagation([Xin; Yin], weights, bias, tanh);
Z = reshape(Zaux{end},size(X));

figure
surf(X,Y,Z)

hold on
plot3(x1(test_output > 0), x2(test_output > 0), f(test_output > 0), '.r') 
plot3(x1(test_output <= 0), x2(test_output <= 0), f(test_output <= 0), '.b') 
hold off

title("Output After Training"); 
xlabel("x_1"); ylabel("x_2"); zlabel("f(x_1,x_2)");
grid on;
xlim([-2, 2]); ylim([-2, 2]);
daspect([1 1 1]);

clear Xin Yin Zaux X Y Z

%% Plot of Target Results

figure

plot(x1(f > 0), x2(f > 0),'.r') 
hold on
plot(x1(f <= 0), x2(f <= 0),'.b') 
hold off

yline(0); xline(0);

title("Expected Results"); 
xlabel("x_1"); ylabel("x_2");
xlim([-2, 2]); ylim([-2, 2]);
daspect([1 1 1]);
