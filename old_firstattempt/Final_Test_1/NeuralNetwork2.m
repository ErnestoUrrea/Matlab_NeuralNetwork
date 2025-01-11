classdef NeuralNetwork2 < handle
    properties
        layers
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
        function obj = NeuralNetwork2(layers, lr)
            obj.layers = layers;
            obj.initializeNetwork();
            obj.learningRate = lr;
        end
        function initializeWeights(obj)
            obj.weights{1,1} = [];
            for layer = 2:size(obj.layers,2)
                obj.weights{layer,1} = -1 + 2.*rand(obj.layers(layer),obj.layers(layer-1));
            end
        end
        function w = getWeights(obj)
            w = obj.weights;
        end
        function initializeBias(obj)
            obj.bias{1,1} = [];
            for layer = 2:size(obj.layers,2)
                obj.bias{layer,1} =  -1 + 2.*rand(obj.layers(layer),1);
            end
        end
        function b = getBias(obj)
            b = obj.bias;
        end
        function initializeNetwork(obj)
            obj.initializeWeights();
            obj.initializeBias();
        end
        function output = forwardPropagation(obj,input)
            obj.activatedNeurons = {input};
            for layer = 2:size(obj.layers,2)
                obj.inactiveNeurons{layer} = obj.weights{layer}*obj.activatedNeurons{layer-1}+obj.bias{layer};
                if layer == size(obj.layers,2)
                    obj.activatedNeurons{layer} = obj.softmaxActivation(obj.inactiveNeurons{layer});
                else 
                    obj.activatedNeurons{layer} = obj.reluActivation(obj.inactiveNeurons{layer});
                end
            end
            output = obj.activatedNeurons{end};
        end
        function backPropagation(obj, input, exp_out)
%             obj.forwardPropagation(input);
%             obj.imputedNeuronErrors{size(obj.layers,2)} = obj.lastLayerImputedError(obj.activatedNeurons{size(obj.layers,2)}, exp_out);
%             for layer = size(obj.layers,2)-1:-1:2
%                 obj.imputedNeuronErrors{layer} = obj.hiddenLayerImputedError(obj.imputedNeuronErrors{layer+1}, obj.weights{layer+1,1}, obj.reluDerivative(obj.activatedNeurons{layer}));
%             end
%             for layer = 2:size(obj.weights)
%                 obj.weights{layer,1} = obj.weights{layer,1} - obj.learningRate*obj.imputedNeuronErrors{layer}*obj.activatedNeurons{layer-1}';
%                 obj.bias{layer,1} = obj.bias{layer,1} - obj.learningRate*obj.imputedNeuronErrors{layer};
%             end
            numberOfLayers = size(obj.layers,2);
            deltaWeights = cell(numberOfLayers,1);
            deltaBias = cell(numberOfLayers,1);
            
            for layer = 2:numberOfLayers
                deltaWeights{layer,1} = zeros(size(obj.weights{layer,1}));
                deltaBias{layer,1} = zeros(size(obj.bias{layer,1}));
            end
            
            obj.forwardPropagation(input);
            for sample = 1:size(input, 2)
                obj.imputedNeuronErrors{numberOfLayers, sample} = obj.lastLayerImputedError(obj.activatedNeurons{numberOfLayers}(:,sample), exp_out(:,sample));
                for layer = numberOfLayers-1:-1:2
                    obj.imputedNeuronErrors{layer, sample} = obj.hiddenLayerImputedError(obj.imputedNeuronErrors{layer+1, sample}, obj.weights{layer+1,1}, obj.reluDerivative(obj.activatedNeurons{layer}(:,sample)));
                    deltaWeights{layer,1} = deltaWeights{layer,1} + obj.imputedNeuronErrors{layer, sample}*obj.activatedNeurons{layer-1}(:,sample)';
                    deltaBias{layer,1} = deltaBias{layer,1} + obj.imputedNeuronErrors{layer, sample};
                end
            end
            for layer = 2:numberOfLayers
                deltaWeights{layer,1} = deltaWeights{layer,1}./size(input, 2);
                deltaBias{layer,1} = deltaBias{layer,1}./size(input, 2);
            end
            for layer = 2:numberOfLayers
                obj.weights{layer,1} = obj.weights{layer,1} - obj.learningRate*deltaWeights{layer,1};
                obj.bias{layer,1} = obj.bias{layer,1} - obj.learningRate*deltaBias{layer,1};
            end
        end
        function train(obj, dataset, expectedOut, batchSize)
            for batch = 1:floor(1:size(batchSize))
                obj.backPropagation(dataset(:,(batch-1)*batchSize+1:batch*batchSize), expectedOut(:,(batch-1)*batchSize+1:batch*batchSize));
            end
        end
    end
    methods (Access = private)
        function R = softmaxActivation(~,x)
            aux = x - max(x); % Se resta el maximo para evitar overflow
            R = exp(aux)./sum(exp(aux));
        end
        function R = softmaxDerivative(~, sftmxValues)
            [f,~] = size(sftmxValues);
            R = eye(f).*sftmxValues - sftmxValues'.*sftmxValues;
        end
        function R = reluActivation(~, x)
            R = max(0,x);
        end
        function R = reluDerivative(~, x)
            R = eye(size(x,1)).*(x>0);
        end
        function loss = lossCategoricalCrossEntropy(out, expectedOut)
            out = min(1-1*10^-7, max(out, 1*10^-7));
            lossValues = -sum(expectedOut().*log(out()));
            loss = mean(lossValues);
        end
        function R = lossCategoricalCrossEntropyDerivative(~, out, exp_out)
            R = -(exp_out./out);
        end
        function R = lastLayerImputedError(~, out, exp_out)
            R = (out-exp_out)./size(out,2);
        end
        function R = hiddenLayerImputedError(~, previousImputedError, weights, activationFunctionDerivative)
            R = activationFunctionDerivative*weights'*previousImputedError;
        end
    end
end