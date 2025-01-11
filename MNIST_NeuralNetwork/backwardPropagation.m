function [n_in_der, n_out_der, err_der] = backwardPropagation(in, n_in, n_out, target, w, actfun, outactfun, costfun, auxfun)
%BACKWARDPROPAGATION Performs backward propagation for a neural network.
%   * Pending detailed description and parameter description and constraints. 

%     % Argument validation
%     arguments
%         in (:,:) double
%         n_in (:,1) cell
%         n_out (:,1) cell
%         target (:,:) double
%         w (:,1) cell
%         actfun (2,1) cell = {@(x) x @(x) 1}
%         outactfun (2,1) cell = actfun
%         costfun (2,1) cell = {@(out, tar) 0.5.*(out - tar).^2 @(out, tar) (out - tar)};
%         auxfun (1,1) function_handle = 1
%     end

    % Derivatives Memory Allocation
    n_in_der = cell(size(n_in));
    n_out_der = cell(size(n_out));
    err_der = cell(size(w));
    
    nn_size = size(n_in,1);

    % Calculation of Derivatives (Backpropagation)
    for layer_ind = nn_size:-1:1
        if layer_ind == nn_size
            %n_out_der{layer_ind} = costfun{2}(n_out{layer_ind}, target);
            %n_in_der{layer_ind} = n_out_der{layer_ind}.*outactfun{2}(n_in{layer_ind});
            n_in_der{layer_ind} = auxfun(n_out{layer_ind}, target);
        else
            n_out_der{layer_ind} = w{layer_ind + 1}'*n_in_der{layer_ind + 1};
            n_in_der{layer_ind} = n_out_der{layer_ind}.*actfun{2}(n_in{layer_ind});
        end

        if layer_ind == 1
            err_der{layer_ind} = n_in_der{layer_ind}*in';
        else
            err_der{layer_ind} = n_in_der{layer_ind}*n_out{layer_ind - 1}';
        end
    end
    
end

