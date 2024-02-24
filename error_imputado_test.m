softmax_outputs = [0.7, 0.1, 0.02; 0.1, 0.5, 0.9; 0.2, 0.4, 0.08];
expected_output = [1, 0, 0; 0, 1, 1; 0, 0, 0];
error_imputado_1 = zeros(size(softmax_outputs));
error_imputado_2 = zeros(size(softmax_outputs));

[~,samples] = size(softmax_outputs);
for s = 1:samples
    error_imputado_1(:,s) = softmaxDerivative(softmax_outputs(:,s))*(lossCategoricalCrossEntropyDerivative(softmax_outputs(:,s), expected_output(:,s))./samples);
end

error_imputado_2 = neuronsImputedError(softmax_outputs, expected_output);

error_imputado_1
error_imputado_2


function R = softmaxDerivative(sftmxValues)
    [f,~] = size(sftmxValues);
    R = eye(f).*sftmxValues - sftmxValues'.*sftmxValues;
end
function R = lossCategoricalCrossEntropyDerivative(out, exp_out)
    R = -(exp_out./out);
end
function R = neuronsImputedError(out, exp_out)
    R = (out-exp_out)/size(out,2);
end