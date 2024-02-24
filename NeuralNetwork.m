classdef NeuralNetwork < handle
    % Propiedades publicas
    properties
        activationHiddenLayers
        activationOutput
        costFunction
        layers
    end
    % Propiedades privadas
    properties (Access = private)
        weights
        bias
        inactiveNeurons
        activatedNeurons
        imputedErrors
    end
    
    % Metodos publicos
    methods
        function obj = NeuralNetwork(HL_activ, O_activ, cost_func, layers)
            obj.activationHiddenLayers = HL_activ;
            obj.activationOutput = O_activ;
            obj.costFunction = cost_func;
            obj.layers = layers;
            obj.initializeNetwork();
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
        function n = getNeurons(obj)
            n = obj.activatedNeurons();
        end
        function output = forwardPropagation(obj,input)
            mrelu = @(x) max(0,x);
            obj.activatedNeurons = {input};
            for layer = 2:size(obj.layers,2)
                if layer == size(obj.layers,2)
                    out_wo_sftmx = obj.weights{layer}*obj.activatedNeurons{layer-1}+obj.bias{layer};
                    out_wo_sftmx = out_wo_sftmx - max(out_wo_sftmx); % Se resta el maximo para evitar overflow
                    obj.inactiveNeurons{layer} = out_wo_sftmx;
                    obj.activatedNeurons{layer} = exp(out_wo_sftmx)./sum(exp(out_wo_sftmx));
                else
                    obj.activatedNeurons{layer} = double(mrelu(obj.weights{layer}*obj.activatedNeurons{layer-1}+obj.bias{layer}));
                end
            end
            output = obj.activatedNeurons{end};
        end
        function backPropagation(obj,input,exp_out)
            obj.forwardPropagation(input);
            obj.imputedErrors{size(obj.layers,1)} = obj.lastLayerImputedError(obj.activatedNeurons{size(obj.layers,1)}, exp_out);
            for layer = size(obj.layers,1)-1:-1:2
                obj.imputedErrors{layer} = hiddenLayerImputedError(obj.imputedErrors{layer+1}, obj.weights{layer+1,1}, obj.reluDerivative(obj.activatedNeurons{layer}));
            end
            obj.imputedErrors
        end
    end
    % Metodos privados
    methods (Access = private)
        function dataLoss = lossCategoricalCrossEntropy(out, trgt_out)
            out = min(1-1*10^-7, max(out, 1*10^-7));
            loss_vals = -sum(trgt_out().*log(out()));
            dataLoss = mean(loss_vals);
        end
        
        function R = softmaxDerivative(~, sftmxValues)
            [f,~] = size(sftmxValues);
            R = eye(f).*sftmxValues - sftmxValues'.*sftmxValues;
        end
        function R = lossCategoricalCrossEntropyDerivative(~, out, exp_out)
            R = -(exp_out./out);
        end
        function R = lastLayerImputedError(~, out, exp_out)
            R = (out-exp_out)/size(out,2);
        end
        function R = reluDerivative(~, x)
            R = eye(size(x,1)).*(x>0);
        end
        function R = hiddenLayerImputedError(~, previousImputedError, weights, activationFunctionDerivative)
            R = activationFunctionDerivative*weights'*previousImputedError;
        end
    end
end






