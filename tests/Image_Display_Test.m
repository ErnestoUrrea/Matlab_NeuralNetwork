%% Clearing Environment
clc; clear; close all;
format long;

%% MNIST Data
load("mnist.mat");

empty_img = ones([training.height training.width 3]);

empty_img(:,:,1) = 1-training.images(:,:,1);
empty_img(:,:,2) = 1-training.images(:,:,1);
empty_img(:,:,3) = 1-training.images(:,:,1);

image(empty_img)