clear all;close all;
addpath(genpath('dataset'));

load('block_3D.mat');

Kcell = cellfun(@sparse , num2cell(block_3D,[1,2]) , 'uni', 0 );
S_left=blkdiag(Kcell{:});

save('dataset/S_left.mat',"S_left");

