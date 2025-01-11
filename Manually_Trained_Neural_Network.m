%%
clc; clear; close all;

%% Test Intput
test_input1 = 0.25*randn([2 1000]) + [ 1;  1];
test_input2 = 0.25*randn([2 1000]) + [ 1; -1];
test_input3 = 0.25*randn([2 1000]) + [-1; -1];
test_input4 = 0.25*randn([2 1000]) + [-1;  1];

test_input = [test_input1 test_input2 test_input3 test_input4];

%% 2x1 1x5
input1 = [1; 4];

% layer1w = [5 3; 2 4; 8 8; 6 3; 2 1];
% layer1b = [0 1 0 2 1];

layer1w = [1 1; -1 -1];
layer1b = [-1; -1];

layer1r = layer1w*test_input + layer1b;

layer1r = layer1r.*(layer1r > 0); % Relu function

layer2w = [1 1];
layer2b = [0];

layer2r = layer2w*layer1r + layer2b;

layer2r = layer2r.*(layer2r > 0); % Relu function

%condition = layer1r > 0;

%% Plot of Results in 3D

[X, Y] = meshgrid(-2:0.05:2,-2:0.05:2);

Xin = reshape(X,1,[]);
Yin = reshape(Y,1,[]);

Zaux = layer1w*[Xin; Yin] + layer1b;
Zaux = Zaux.*(Zaux > 0);

%Z1 = reshape(Zaux(1,:),size(X));
%Z2 = reshape(Zaux(2,:),size(X));

Zaux = layer2w*Zaux + layer2b;
Zaux = Zaux.*(Zaux > 0);

Z = reshape(Zaux,size(X));

figure

surf(X,Y,Z) 

hold on
plot(test_input(1,layer2r > 0), test_input(2,layer2r > 0),'.r') 
plot(test_input(1,layer2r <= 0), test_input(2,layer2r <= 0),'.b') 
hold off

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

figure

plot(test_input(1,layer2r > 0), test_input(2,layer2r > 0),'.r') 
hold on
plot(test_input(1,layer2r <= 0), test_input(2,layer2r <= 0),'.b') 
hold off

yline(0)
xline(0)

xlim([-2, 2])
ylim([-2, 2])

daspect([1 1 1])