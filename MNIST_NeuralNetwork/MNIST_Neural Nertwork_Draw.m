%% Clearing Environment
clc; clear; close all;
format long;

%% Empty Image Drawing
% Initialize a blank 28x28 image
imageData = ones([28 28 3]);

% Create a figure for the drawing application
figure('Name', 'Draw on 28x28 Image', 'NumberTitle', 'off');
h = image(imageData);

grid on

xlim([0.5, 28.5])
ylim([0.5, 28.5])

daspect([1 1 1])

%% Function Callbacks
% Set up mouse events for drawing
set(gcf, 'WindowButtonDownFcn', @mouseDown);
set(gcf, 'WindowButtonMotionFcn', @mouseMove);
set(gcf, 'WindowButtonUpFcn', @mouseUp);

% Initialize drawing state
drawing = false;

% Mouse down callback function
function mouseDown(~, ~)
    global drawing;
    drawing = true;
    drawPoint(); % Draw at the initial mouse down position
end

% Mouse move callback function
function mouseMove(~, ~)
    global drawing;
    if drawing
        drawPoint(); % Draw while the mouse is moving
    end
end

% Mouse up callback function
function mouseUp(~, ~)
    global drawing;
    drawing = false;
    getPrediction()
end

% Function to draw a point on the image
function drawPoint()
    % Get the current point in axes coordinates
    point = get(gca, 'CurrentPoint');
    x = round(point(1, 2));
    y = round(point(1, 1));

    k = gca().Children;

    % Check if the point is within the image bounds
    if x >= 1 && x <= 28 && y >= 1 && y <= 28
        k.CData(x,y,1) = min(0,k.CData(x,y,1));
        k.CData(x,y,2) = min(0,k.CData(x,y,2));
        k.CData(x,y,3) = min(0,k.CData(x,y,3));

        k.CData(x+1,y,1) = min(0,k.CData(x+1,y,1));
        k.CData(x+1,y,2) = min(0,k.CData(x+1,y,2));
        k.CData(x+1,y,3) = min(0,k.CData(x+1,y,3));

        k.CData(x,y+1,1) = min(0,k.CData(x,y+1,1));
        k.CData(x,y+1,2) = min(0,k.CData(x,y+1,2));
        k.CData(x,y+1,3) = min(0,k.CData(x,y+1,3));

        k.CData(x+1,y+1,1) = min(0,k.CData(x+1,y+1,1));
        k.CData(x+1,y+1,2) = min(0,k.CData(x+1,y+1,2));
        k.CData(x+1,y+1,3) = min(0,k.CData(x+1,y+1,3));

%         k.CData(x+1,y,1) = min(0.5,k.CData(x+1,y,1));
%         k.CData(x+1,y,2) = min(0.5,k.CData(x+1,y,2));
%         k.CData(x+1,y,3) = min(0.5,k.CData(x+1,y,3));
% 
%         k.CData(x,y+1,1) = min(0.5,k.CData(x,y+1,1));
%         k.CData(x,y+1,2) = min(0.5,k.CData(x,y+1,2));
%         k.CData(x,y+1,3) = min(0.5,k.CData(x,y+1,3));
% 
%         k.CData(x-1,y,1) = min(0.5,k.CData(x-1,y,1));
%         k.CData(x-1,y,2) = min(0.5,k.CData(x-1,y,2));
%         k.CData(x-1,y,3) = min(0.5,k.CData(x-1,y,3));
% 
%         k.CData(x,y-1,1) = min(0.5,k.CData(x,y-1,1));
%         k.CData(x,y-1,2) = min(0.5,k.CData(x,y-1,2));
%         k.CData(x,y-1,3) = min(0.5,k.CData(x,y-1,3));
    
        drawnow
    end
end

% Function to guess the numbet
function getPrediction()
    load("neuralNetwork3.mat","bias","weights");

    relu = {
        @(x) x.*(x >= 0) 
        @(x) 1.*(x >= 0)
        };
    sfmx = {
        @(x) exp(x)./sum(exp(x),1)
        @(x) (exp(x)./sum(exp(x),1)).*(1-(exp(x)./sum(exp(x),1)))
        };

    k = gca().Children;
    image = 1-k.CData(:,:,1);
    testing_input = reshape(image,28*28,[]);

    [~, neuron_out] = forwardPropagation(testing_input, weights, bias, relu, sfmx);

    [M,I] = max(neuron_out{end});
    
     clc
     confianza = M
     prediccion = I-1

    pause(3)

%     clc;
    k.CData = ones([28 28 3]);
end