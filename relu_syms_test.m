syms x
relu(x) = piecewise(x < 0,0,x > 0,x);
fplot(relu)