%%
clc; clear; close all;

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

%% Important Functions
relu = @(x) x.*(x > 0);

%% Neural Network Definition

% Considerations:
% - First Layer is Input (not true anymore)
% - Last Layer is Output

%layer_sizes = [2; 3; 4; 1]; % obsolete by new standard(?)

input_size = 2;
layer_sizes = [2; 1];

values = cell(size(layer_sizes));
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
    values{layer_ind} = zeros([rows 1]);
end

clear rows cols layer_ind

% Inputs
test_input = [x1;x2];

% Hidden Layer 1
weights{1} = [1 1; -1 -1];
bias{1} = [-1; -1];
values{1} = relu(weights{1}*test_input + bias{1});

% Hidden Layer 2
weights{2} = [1 1];
bias{2} = [0];
values{2} = relu(weights{2}*values{1} + bias{2});

%% Plot of Results in 3D

[X, Y] = meshgrid(-2:0.05:2,-2:0.05:2);

Xin = reshape(X,1,[]);
Yin = reshape(Y,1,[]);

Zaux = relu(weights{1}*[Xin; Yin] + bias{1});

%Z1 = reshape(Zaux(1,:),size(X));
%Z2 = reshape(Zaux(2,:),size(X));

Zaux = relu(weights{2}*Zaux + bias{2});

Z = reshape(Zaux,size(X));

figure

surf(X,Y,Z) 

% hold on
% plot(test_input(1,layer2r > 0), test_input(2,layer2r > 0),'.r') 
% plot(test_input(1,layer2r <= 0), test_input(2,layer2r <= 0),'.b') 
% hold off

% surf(X,Y,Z1) 
% hold on
% surf(X,Y,Z2) 
% hold off

yline(0)
xline(0)

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

function out = forwardPropagation(w, b, v, in)
    v{1} = relu(w{1}*in + b{1});
    v{2} = relu(w{2}*v{1} + b{2});
    out = v{2};
end