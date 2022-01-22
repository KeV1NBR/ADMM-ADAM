clear all;close all;
addpath(genpath('dataset'));
addpath(genpath('function'));
%% load HSI and mask data (Ottawa)
load('Ottawa.mat')
%X3D_ref = int16(X3D_ref);
%save("inpainting192.mat", "X3D_ref");
load('mask.mat')
%mask_3D = int16(mask_3D);
load('X3DL.mat')
load('result_inpainting_2D_it05000_192.mat')
%load('X3D_rec.mat')
%% algorithm
X3D_corrupted = X3D_ref .* mask_3D;
[X3D_rec,time] = ADMMADAM(X3D_corrupted,mask_3D,double(pred));
%% plot
%plot_result(X3D_ref,X3D_DL,X3D_rec,mask_3D)
plot_result(X3D_ref,pred,X3D_rec,mask_3D)