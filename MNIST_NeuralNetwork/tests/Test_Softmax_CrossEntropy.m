%% Clearing Environment
clc; clear; close all;
format long;

%% Activation Functions
sfmx = {
    @(x) exp(x)./sum(exp(x),1)
    @(x) (exp(x)./sum(exp(x),1)).*(1-(exp(x)./sum(exp(x),1)))
    };

%% Cost Functions
ccef = {
    @(out, tar) -sum(tar.*log(out), 1)
    @(out, tar) -tar./out + (1-tar)./(1-out)
    };

%% Tests
% Last Layer Data Definitions
T = [0; 1; 0; 0; 0]            % Target Output
Z = [1.3; 5.1; 2.2; 0.7; 1.1]  % Activation Function Inputs

% Neural Network Outputs
O = sfmx{1}(Z)                 % Last Layer Outputs
C = ccef{1}(O,T)               % Cost Function Result

% Derivatives Calculation (Functions)
dCdO = ccef{2}(O,T)            % Cost Function Derivative
dOdZ = sfmx{2}(Z)              % Activation Function Derivative
dCdZ = dCdO.*dOdZ              % Last Layer Derivatives (at Input)

% Derivatives Calculation (Simplified)
O.*(1-O)                       % Act. Func. Deriv. (should be equal to dOdZ)
O-T                            % Last Layer Deriv. (should be equal to dCdZ)


