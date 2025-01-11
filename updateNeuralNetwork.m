function [w, b] = updateNeuralNetwork(n_in_der, err_der, w, b, learning_rate)
%BACKWARDPROPAGATION Performs backward propagation for a neural network.
%   * Pending detailed description and parameter description and constraints. 

%     % Argument validation
%     arguments
%         in (:,:) double
%         w (:,1) cell 
%         b (:,1) cell
%         actfun (2,1) cell = {@(x) x @(x) 1}
%         outactfun (2,1) cell = actfun
%     end
    for layer_ind = 1:size(w,1)
        b{layer_ind} = b{layer_ind} - learning_rate.*n_in_der{layer_ind};
        w{layer_ind} = w{layer_ind} - learning_rate.*err_der{layer_ind};
    end
end

