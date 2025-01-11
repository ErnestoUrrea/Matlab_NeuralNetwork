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

%% Plot of Results in 3D

[X, Y] = meshgrid(-2:0.05:2,-2:0.05:2);

Zaux = layer1w*[reshape(X,1,[]); reshape(Y,1,[])] + layer1b;
Zaux = Zaux.*(Zaux > 0);
Z = reshape(Zaux,size(X));

figure

surf(X,Y,Z) 
hold on
plot(test_input(1,layer1r > 0), test_input(2,layer1r > 0),'.r') 
plot(test_input(1,layer1r <= 0), test_input(2,layer1r <= 0),'.b') 
hold off

yline(0)
xline(0)

xlim([-2, 2])
ylim([-2, 2])

daspect([1 1 1])

%% Plot of Test Input

figure

plot(test_input1(1,:), test_input1(2,:),'.')
hold on
plot(test_input2(1,:), test_input2(2,:),'.')
plot(test_input3(1,:), test_input3(2,:),'.')
plot(test_input4(1,:), test_input4(2,:),'.')
hold off

yline(0)
xline(0)

xlim([-2, 2])
ylim([-2, 2])

daspect([1 1 1])