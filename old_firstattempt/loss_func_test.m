target_output = [1, 0, 0; 0, 1, 1; 0, 0, 0];
softmax_output = [0, 0.1, 0.02; 0.1, 0.5, 0.9; 0.2, 0.4, 0.08];

loss_func(softmax_output, target_output)

% function loss_val = loss_func(sftmx_out, trgt_out)
%     [a,~] = size(trgt_out);
%     loss_val = 0;
%     for indx = 1:a
%         loss_val = loss_val - trgt_out(indx,:).*log(sftmx_out(indx,:));
%     end
% end

function loss_val = loss_func(sftmx_out, trgt_out)
    sftmx_out = min(1-1*10^-7, max(sftmx_out, 1*10^-7));
    loss_val = -sum(trgt_out().*log(sftmx_out()));
end