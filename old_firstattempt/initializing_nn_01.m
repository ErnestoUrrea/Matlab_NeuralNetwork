clc; clear;

layers_01 = [3, 4, 4, 2];

weights_01{1,1} = [];
bias_01{1,1} = [];

for layer = 2:size(layers_01,2)
    weights_01{layer,1} = -1 + 2.*rand(layers_01(layer),layers_01(layer-1));
    bias_01{layer,1} =  -1 + 2.*rand(layers_01(layer),1);
end

save("nn_init", "layers_01", "weights_01", "bias_01")