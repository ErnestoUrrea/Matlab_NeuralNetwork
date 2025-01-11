%% Real-Time Testing for Digit-Recognizing Neural Network
% Neural Network data is saved by Training program in a .mat file and
% loaded here.

%% Clearing Environment
clc; clear; close all;
format long;

%% Empty Image Drawing
% Blank 28x28 Image
imageData = ones([28 28 3]);

% Figure for the Drawing Application
figure('Name', 'Draw on 28x28 Image', 'NumberTitle', 'off');
h = image(imageData);

title("Number Recognition Test")
subtitle("Draw a number using your mouse.");
grid on; 
xlim([0.5, 28.5]); ylim([0.5, 28.5]);
daspect([1 1 1]);

%% Function Callbacks
% Set up Mouse Events
set(gcf, 'WindowButtonDownFcn', @mouseDown);
set(gcf, 'WindowButtonMotionFcn', @mouseMove);
set(gcf, 'WindowButtonUpFcn', @mouseUp);

% Initialize Drawing State
drawing = false;

%% Mouse Down Callback Function
function mouseDown(~, ~)
    global drawing;
    drawing = true;
    drawPoint();
end

%% Mouse Move Callback Function
function mouseMove(~, ~)
    global drawing;
    if drawing
        drawPoint();
    end
end

%% Mouse Up Callback Function
function mouseUp(~, ~)
    global drawing;
    drawing = false;
    getPrediction();
end

%% Function to Draw a Pixel on the Image
function drawPoint()
    % Current Point in Axes Coordinates
    point = get(gca, 'CurrentPoint');
    x = round(point(1, 2));
    y = round(point(1, 1));

    k = gca().Children;

    % Check if the Point is Valid
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
        drawnow
    end
end

%% Function to Guess the Number
function getPrediction()
    % Load Neural Network
    load("neuralNetwork3.mat","bias","weights");

    % Activation Functions
    relu = {
        @(x) x.*(x >= 0) 
        @(x) 1.*(x >= 0)
        };
    sfmx = {
        @(x) exp(x)./sum(exp(x),1)
        @(x) (exp(x)./sum(exp(x),1)).*(1-(exp(x)./sum(exp(x),1)))
        };

    % Get Image and Change Data
    k = gca().Children;
    image = 1-k.CData(:,:,1);

    % Forward Propagation
    testing_input = reshape(image,28*28,[]);

    [~, neuron_out] = forwardPropagation(testing_input, weights, bias, relu, sfmx);

    % Prediction Results
    [M,I] = max(neuron_out{end});
    
    clc
    confianza = M
    prediccion = I-1

    pause(3)

    % Erase Image to Guess Again
    k.CData = ones([28 28 3]);
end