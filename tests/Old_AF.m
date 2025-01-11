%% Activation Functions
relu = @(x) x.*(x >= 0);
stpf = @(x) 1.*(x >= 0) - 1.*(x < 0);
tanh = @(x) (exp(x) - exp(-x))./(exp(x) + exp(-x));
sigm = @(x) 1./(1 + exp(-x));

mesq = @(out, tar) 0.5*(out - tar)^2;