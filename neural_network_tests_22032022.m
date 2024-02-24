clc; clear;

S = load("nn_init");

layers = S.layers_01;
weights = S.weights_01;
bias = S.bias_01;

input = [1, 2, 1, 2; 2, 3, 2, 3; 3, 4, 3, 4];

R = forward_propagation(input,weights,bias,layers)
expected_output = [0, 0, 0, 0; 1, 1, 1, 1];
sum(R)
loss = loss_func(R, expected_output)

function output = forward_propagation(input, w, b, layers)
    syms x
    mrelu(x) = piecewise(x < 0,0,x > 0,x);
    neurons = {input};
    for layer = 2:size(layers,2)
        if layer == size(layers,2)
            out_wo_sftmx = w{layer}*neurons{layer-1}+b{layer};
            out_wo_sftmx = out_wo_sftmx - max(out_wo_sftmx); % Se resta el maximo para evitar overflow
            neurons{layer} = exp(out_wo_sftmx)./sum(exp(out_wo_sftmx));
        else
            neurons{layer} = double(mrelu(w{layer}*neurons{layer-1}+b{layer}));
        end
    end
    output = neurons{end};
end

function data_loss = loss_func(sftmx_out, trgt_out)
    sftmx_out = min(1-1*10^-7, max(sftmx_out, 1*10^-7));
    loss_vals = -sum(trgt_out().*log(sftmx_out()));
    data_loss = mean(loss_vals);
end
