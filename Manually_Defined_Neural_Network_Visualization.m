%% Manual Definition and Propagation of a Small Neural Network
% Neural Network is defined with 2 input, 1 output and 1 hidden layer with
% 2 neurons. Manual propagation is performed with the objective of
% understanding better the process.

%% Clearing Environment
clc; clear; close all;
format long;

%% Test Intput
test_input1 = 0.25*randn([2 1000]) + [ 1;  1];
test_input2 = 0.25*randn([2 1000]) + [ 1; -1];
test_input3 = 0.25*randn([2 1000]) + [-1; -1];
test_input4 = 0.25*randn([2 1000]) + [-1;  1];

test_input = [test_input1 test_input2 test_input3 test_input4];

%% Manual Neural Network Definition and Forward Propagation

% First Layer Definition
layer1w = [1 1; -1 -1];
layer1b = [-1; -1];

% First Layer Propagation
layer1r = layer1w*test_input + layer1b;
layer1r = layer1r.*(layer1r > 0); % Relu function

% Second Layer Definition
layer2w = [1 1];
layer2b = [0];

% Second Layer Propagation
layer2r = layer2w*layer1r + layer2b;
layer2r = layer2r.*(layer2r > 0); % Relu function

%% Plot of Output in 3D

% Input Definition
[X, Y] = meshgrid(-2:0.05:2,-2:0.05:2);

Xin = reshape(X,1,[]);
Yin = reshape(Y,1,[]);

% Forward Propagation
Zaux = layer1w*[Xin; Yin] + layer1b;
Zaux = Zaux.*(Zaux > 0);

Z1 = reshape(Zaux(1,:),size(X));
Z2 = reshape(Zaux(2,:),size(X));

Zaux = layer2w*Zaux + layer2b;
Zaux = Zaux.*(Zaux > 0);

Z = reshape(Zaux,size(X));

% Plot
figure

surf(X,Y,Z) 

hold on
plot(test_input(1,layer2r > 0), test_input(2,layer2r > 0),'.r') 
plot(test_input(1,layer2r <= 0), test_input(2,layer2r <= 0),'.b') 
hold off

title("Output"); 
xlabel("x_1"); ylabel("x_2"); zlabel("f(x_1,x_2)");
grid on;
xlim([-2, 2]); ylim([-2, 2]);
daspect([1 1 1]);

%% Plot of Hidden Layer Outputs in 3D

figure
tiledlayout(1,2)

nexttile
surf(X,Y,Z1) 

title("Output of Neuron 1"); 
xlabel("x_1"); ylabel("x_2"); zlabel("f(x_1,x_2)");
grid on;
xlim([-2, 2]); ylim([-2, 2]);
daspect([1 1 1]);

nexttile
surf(X,Y,Z2)

title("Output of Neuron 2"); 
xlabel("x_1"); ylabel("x_2"); zlabel("f(x_1,x_2)");
grid on;
xlim([-2, 2]); ylim([-2, 2]);
daspect([1 1 1]);

%% Plot of Binary Results

figure

plot(test_input(1,layer2r > 0), test_input(2,layer2r > 0),'.r') 
hold on
plot(test_input(1,layer2r <= 0), test_input(2,layer2r <= 0),'.b') 
hold off

yline(0); xline(0);

title("Neural Network Classification Results"); 
xlabel("x_1"); ylabel("x_2");
xlim([-2, 2]); ylim([-2, 2]);
daspect([1 1 1]);