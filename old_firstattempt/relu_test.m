k = [-3; 3; 4; 7; -6];


mrelu_1 = @(x) (x>0).*x;
mrelu_2 = @(x) max(0,x);

mrelu_1(k)
mrelu_2(k)

fplot(mrelu_2)