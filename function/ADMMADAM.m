%=====================================================================
% Programmer: Po-Wei Tang
% E-mail: q38091526@gs.ncku.edu.tw
% Date: October 19, 2021
% -------------------------------------------------------
% Reference:
% ``ADMM-ADAM: A new inverse imaging framework blending the advantages of convex optimization and deep learning,"
% accepted by IEEE Transactions on Geoscience and Remote Sensing, 2021.
%======================================================================
% [X3D_rec,time] = ADMMADAM(X3D_corrupted,mask_3D,X3D_DL)
%======================================================================
% Input
% X3D_corrupted is the corrupted image with dead pixels or stripes, whose dimension is row*col*bands.
% mask_3D is the index set of missing data X3D_corrupted, which is composed of 0 or 1.
% X3D_DL is the reconstructed image from a weak Adam optimizer-based deep learning solution.
%----------------------------------------------------------------------
% Output
% X3D_rec is the inpainting result whose dimension is the same as X3D_corrupted.
% time is the computation time (in secs).
%========================================================================
function [X3D_rec,time] = ADMMADAM(X3D_corrupted,mask_3D,X3D_DL)
t1 = clock;
%% parameters
N=10; % dimension of the hyperspectral subspace
lambda=0.01; % regularization parameter
mu=1e-3; % ADMM penalty parameter 
%% compute S_DL 
[row, col , all_bands] = size(X3D_corrupted);
spatial_len=row*col;
X2D_DL = reshape(X3D_DL,[],all_bands)';
[E_DL] = compute_basis(X3D_DL,N);
%load("E.mat")
%E_DL=E;
%load("tensor.mat")
S_DL = E_DL'*X2D_DL;
%% ADMM
mask_2D = reshape(mask_3D,spatial_len,all_bands)';
nz_idx = sparse([1;zeros(all_bands,1)]);
M_idx = sparse(kron(mask_2D,nz_idx));
M = M_idx(1:all_bands^2,:);
PtrpsP_idx = reshape(ndSparse(M),[all_bands,all_bands,spatial_len]);
PtrpsP = full(PtrpsP_idx); % Omega
RP_tensor= ttm(tensor(PtrpsP),E_DL',1);
RRtrps_tensor= ttm(RP_tensor,E_DL',2);

%RP3D=RP_tensor;
RP3D=RP_tensor.data;
RPY= [];
X2D_corrupted = reshape(X3D_corrupted,[],all_bands)'; % Y in paper
for ii = 1:spatial_len
    RPY(:,:,ii) = RP3D(:,:,ii)*X2D_corrupted(:,ii);
end
RPy = reshape(RPY,[],1);
RRtrps = RRtrps_tensor.data;
%RRtrps = RRtrps_tensor;
RRtrps_per=permute(RRtrps,[3,1,2]);
I=(mu/2)*eye(N);
block=zeros(size(RRtrps_per,1),size(RRtrps_per,2),size(RRtrps_per,3));
for i=1:size(RRtrps_per,1)
    block(i,:,:)=inv(reshape(RRtrps_per(i,:,:),10,[])+I);
end
block_3D = permute(block,[2,3,1]);
Kcell = cellfun(@sparse , num2cell(block_3D,[1,2]) , 'uni', 0 );
S_left=blkdiag(Kcell{:});
for i = 0:50
    if i==0
        S2D = zeros(N,spatial_len);
        D=zeros(N,spatial_len);
    end
    Z = (1/(mu+lambda))*(lambda*S_DL+mu*(S2D-D)); % update Z   
    DELTA = (Z+D);
    delta = reshape(DELTA,[],1);
    s_right = RPy + (mu/2)*delta;
    s = S_left*s_right;
    S2D = reshape(s,[N,65536]); % update s   
    D = D - S2D + Z; %update d
end
X2D_rec=E_DL*S2D;
X3D_rec = reshape(X2D_rec',256,256,172);
time = etime(clock, t1);