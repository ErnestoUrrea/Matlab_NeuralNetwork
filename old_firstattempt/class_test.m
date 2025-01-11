nn1 = NeuralNetwork(1, 1, 1, [3, 4, 4, 2]);

nn1.getWeights();

nn1.initializeWeights();

nn1.getWeights();

nn1.forwardPropagation([1, 2, 1, 2; 2, 3, 2, 3; 3, 4, 3, 4]);

nn1.forwardPropagation([1, 2, 1, 2; 2, 3, 2, 3; 3, 4, 3, 4]);

nn1.forwardPropagation([1, 1, 1, 1; 1, 1, 1, 1; 1, 1, 1, 1]);

nn1.backPropagation([4;6;7],[0;0;1])