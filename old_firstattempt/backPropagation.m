function backPropagation(obj, input, exp_out)
    numberOfLayers = size(obj.layers,2);
    obj.forwardPropagation(input);
    for sample = 1:size(input, 2)
        obj.imputedNeuronErrors{numberOfLayers, sample} = obj.lastLayerImputedError(obj.activatedNeurons{numberOfLayers}(:,sample), exp_out(:,sample));
        for layer = size(numberOfLayers)-1:-1:2
            obj.imputedNeuronErrors{layer, sample} = obj.hiddenLayerImputedError(obj.imputedNeuronErrors{layer+1, sample}, obj.weights{layer+1,1}, obj.reluDerivative(obj.activatedNeurons{layer}(:,sample)));
        end
        for layer = 2:numberOfLayers
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