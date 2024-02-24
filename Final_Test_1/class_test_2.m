clc; clear;

% Train data: Outer cluster
[a1, b1, n1, R1, x01, y01] = deal(-1, 1, 500, 10, 0, 0);
t1 = 2*pi*rand(1,n1);
r1 = R1 + (a1 + (b1-a1).*rand(1,n1));
x1 = x01 + r1.*cos(t1);
y1 = y01 + r1.*sin(t1);
% Train data: Inner cluster
[a2, b2, n2, R2, x02, y02] = deal(-1, 1, 500, 5, 0, 0);
t2 = 2*pi*rand(1,n2);
r2 = R2 + (a2 + (b2-a2).*rand(1,n2));
x2 = x02 + r2.*cos(t2);
y2 = y02 + r2.*sin(t2);

% Test data: Outer cluster
[a3, b3, n3, R3, x03, y03] = deal(-1, 1, 200, 10, 0, 0);
t3 = 2*pi*rand(1,n3);
r3 = R3 + (a3 + (b3-a3).*rand(1,n3));
x3 = x03 + r3.*cos(t3);
y3 = y03 + r3.*sin(t3);
% Test data: Inner cluster
[a4, b4, n4, R4, x04, y04] = deal(-1, 1, 200, 5, 0, 0);
t4 = 2*pi*rand(1,n4);
r4 = R4 + (a4 + (b4-a4).*rand(1,n4));
x4 = x04 + r4.*cos(t4);
y4 = y04 + r4.*sin(t4);


plot(x1,y1,".b")
hold on
plot(x2,y2,".r")
plot(x3,y3,".g")
plot(x4,y4,".y")
hold off

trainIn = [[x1;y1],[x2;y2]];
trainOut = zeros(2,size(trainIn,2));
indices = [ones(size(x1)),zeros(size(x2))]+1;
idx = sub2ind(size(trainOut), indices, 1:size(trainOut,2));
trainOut(idx) = 1;

cols = size(trainIn,2);
P = randperm(cols);
trainDataset = trainIn(:,P);
trainExpectedOut = trainOut(:,P);

testIn = [[x3;y3],[x4;y4]];
testOut = zeros(2,size(testIn,2));
indices = [ones(size(x3)),zeros(size(x4))]+1;
idx = sub2ind(size(testOut), indices, 1:size(testOut,2));
testOut(idx) = 1;

cols = size(testIn,2);
P = randperm(cols);
testDataset = testIn(:,P);
testExpectedOut = testOut(:,P);

structure = [2, 8, 8, 2];
nn2 = NeuralNetwork2(structure,1.5);

beforeout = nn2.forwardPropagation(trainIn);

nn2.train(trainDataset, trainExpectedOut, 1)

afterout = nn2.forwardPropagation(trainIn);

equality = zeros(1,size(trainExpectedOut,2));
equality2 = zeros(1,size(trainExpectedOut,2));

[round(afterout);trainExpectedOut];

for sample = 1:size(trainExpectedOut,2)
    if isequal(round(afterout(:,sample)),trainExpectedOut(:,sample))
        equality(1,sample) = 1;
    end
    if isequal(round(afterout(:,sample)),round(beforeout(:,sample)))
        equality2(1,sample) = 1;
    end
    
end

equality;
sum(equality)
sum(equality2)

% [trainExpectedOut;round(out)]
% trainExpectedOut == round(out)


