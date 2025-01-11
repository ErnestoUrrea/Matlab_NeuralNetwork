function [n_in, n_out] = forwardPropagation(in, w, b, actfun, outactfun)
%FORWARDPROPAGATION Performs forward propagation for a neural network.
%   * Pending detailed description and parameter description and constraints. 

    % Argument validation
    arguments
        in (:,:) double
        w (:,1) cell 
        b (:,1) cell
        actfun (2,1) cell = {@(x) x @(x) 1}
        outactfun (2,1) cell = actfun
    end
    
    % Neuron In And Out Values Cell Memory AllocationS
    n_in = cell(size(w));
    n_out = cell(size(w));

    % Forward Propagation
    for layer_ind = 1:size(n_in,1)
        if layer_ind == 1
            n_in{layer_ind} = w{layer_ind}*in + b{layer_ind};
        else
            n_in{layer_ind} = w{layer_ind}*n_out{layer_ind-1} + b{layer_ind};
        end

        if layer_ind == size(w,1)
            n_out{layer_ind} = outactfun{1}(n_in{layer_ind});
        else
            n_out{layer_ind} = actfun{1}(n_in{layer_ind});
        end
    end
    
end
