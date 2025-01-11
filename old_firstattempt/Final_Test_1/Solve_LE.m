syms x y z
eqn1 = (sqrt(2)/2)*x - (sqrt(6)/6)*y + (sqrt(3)/3)*z == 4;
eqn2 = (sqrt(2)/2)*x + (sqrt(6)/6)*y - (sqrt(3)/3)*z == -2;
eqn3 = (sqrt(6)/3)*y + (sqrt(3)/3)*z == -4;

[A,B] = equationsToMatrix([eqn1, eqn2, eqn3], [x, y, z])

X = linsolve(A,B)

X = simplify(X)

simplify(A*X)