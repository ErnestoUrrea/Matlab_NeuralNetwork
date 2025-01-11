% function out = forwardPropagation(w, b, in, actfun, outactfun, out_ind)
% %FORWARDPROPAGATION Performs forward propagation for a neural network.
% %   * Pending detailed description and parameter description and constraints. 
% %   * Pending implementation of multiple activation functions
% %
% %   OUT = forwardPropagation(W, B, V, IN)       Does forward propagation of 
% %                                               N.N. defined by weight cell 
% %                                               W and bias cell B using input
% %                                               IN and returns the values of
% %                                               the neurons on the last layer.
% %
% %   OUT = forwardPropagation(W, B, V, IN, IND)  Does forward propagation of 
% %                                               N.N. defined by weight cell 
% %                                               W and bias cell B using input
% %                                               IN and returns the values of
% %                                               the neurons on layers given 
% %                                               in IND.
% 
%     % Argument validation
%     arguments
%         w (:,1) cell 
%         b (:,1) cell
%         in (:,:) double
%         actfun (2,1) cell = {@(x) x @(x) 1}
%         outactfun (2,1) cell = actfun
%         out_ind (1,:) double = size(w,1)
%         
%     end
%     
%     % Neuron Value Cell
%     v_in = cell(size(w));
%     v_out = cell(size(w));
% 
%     % Forward Propagation
%     for layer_ind = 1:size(v_in,1)
%         if layer_ind == 1
%             v_in{layer_ind} = w{layer_ind}*in + b{layer_ind};
%         else
%             v_in{layer_ind} = w{layer_ind}*v_out{layer_ind-1} + b{layer_ind};
%         end
% 
%         if layer_ind == size(w,1)
%             v_out{layer_ind} = outactfun{1}(v_in{layer_ind});
%         else
%             v_out{layer_ind} = actfun{1}(v_in{layer_ind});
%         end
%     end
%     
%     % Output Assignment
%     if size(out_ind,1) == 1 && size(out_ind,2) == 1
%         out = v_out{out_ind};
%     else
%         out = v_out(out_ind);
%     end
% end
