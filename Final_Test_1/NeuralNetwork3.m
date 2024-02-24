classdef NeuralNetwork3 < handle
    properties
        layers % an [a1,a2,...,an] form vector that contains the size of each layer
        learningRate
    end   
    properties (Access = private)
        weights
        bias
        inactiveNeurons
        activatedNeurons
        imputedNeuronErrors
    end
    methods
        function obj = NeuralNetwork3(layers, lr)
            obj.layers = layers;
            obj.initializeNetwork();
            obj.learningRate = lr;
        end
        function initializeWeights(obj)
            obj.weights{1,1} = [];
            for layer = 2:size(obj.layers,2)
            
        end
    end
end